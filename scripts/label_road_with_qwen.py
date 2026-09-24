"""Generate unverified Qwen road-state proposals from a NuScenes annotation template.

The output uses `proposed_label`, never `label`, so it cannot silently become
supervised ground truth. Human review is required before exporting a training manifest.
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.data.nuscenes_road_dataset import load_nuscenes, ROAD_CLASSES

ENDPOINT = 'https://maas.qianwenaiapi.com/compatible-mode/v1/chat/completions'
DEFAULT_MODEL = 'qwen3.8-flash'

POLICY_VERSION = 'evidence_based_v3'

PROMPT = """你是自动驾驶道路状态标注员。按时间顺序查看四张 CAM_FRONT 图像，
只判断最后一张图像时自车**所在**道路状态；之前三张仅提供运动上下文。
类别：0 elevated_up=自车已经行驶在高架道路上；
1 elevated_down=自车正行驶在高架道路下方；
2 main_road=有连续通行车道、道路标线及公路通行性质的普通公共道路，且没有明确辅路特征；
3 side_road=有明确证据表明自车在与主路并行、分隔的辅路/服务道路；
4 intersection=自车此刻实际位于路口交叉区域内。
严格规则：看到前方路口、红灯、停止线或人行横道不等于“路口中”。
如果斑马线和行人贴近画面底部、车辆正在等待横穿交通，应先考虑自车尚在路口前；
只有已经驶入两条道路实际交叉的开放区域才标 4。
看到远处桥梁不等于“高架上/下”；普通街道不要仅凭宽窄猜成辅路。
停车场内部通道、私人出入口或场内车道不是公共主路；若不能合理归入五类，返回 null。
main_road 不能只因排除了其他四类就高置信度；此时最多 medium。
请优先辨认自车所在路面而不是其他车辆的位置，不要使用场景名称、文件名或外部知识。
仅输出一个 JSON 对象：
{"proposed_label":0到4的整数或null,"confidence":"high|medium|low",
"evidence":"最后一帧中支持判断的具体可见证据，简短中文",
"reason":"说明分类依据及最容易混淆的类别；若主路是排除其他类后的默认判断，请明确说明"}。
confidence 是主观审查优先级，不是校准概率。"""


def sequence_for_target(nusc, row, num_frames, stride, camera, pad_history=False):
    sample = nusc.get('sample', row['sample_token'])
    if sample['scene_token'] != row['scene_token']:
        raise ValueError(f'Scene mismatch for {row["sample_token"]}')
    chain = []
    while len(chain) < 1 + (num_frames - 1) * stride:
        sd = nusc.get('sample_data', sample['data'][camera])
        path = Path(nusc.dataroot) / sd['filename']
        if not path.is_file():
            raise FileNotFoundError(path)
        chain.append(str(path))
        if len(chain) == 1 + (num_frames - 1) * stride:
            break
        if not sample['prev']:
            if not pad_history:
                raise ValueError(f'Insufficient history for {row["sample_token"]}')
            chain.extend([chain[-1]] * (1 + (num_frames - 1) * stride - len(chain)))
            break
        sample = nusc.get('sample', sample['prev'])
    return [chain[j] for j in range((num_frames - 1) * stride, -1, -stride)]


def encode_image(path, max_width):
    with Image.open(path) as original:
        im = ImageOps.exif_transpose(original).convert('RGB')
        if im.width > max_width:
            im = im.resize((max_width, round(im.height * max_width / im.width)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=82, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def parse_answer(value):
    if not isinstance(value, str):
        raise ValueError('Response content is not text')
    value = value.strip()
    if value.startswith('```'):
        value = re.sub(r'^```(?:json)?\s*|\s*```$', '', value, flags=re.I)
    try:
        item = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', value, flags=re.S)
        if not match:
            raise ValueError('No JSON object in response')
        item = json.loads(match.group(0))
    label = item.get('proposed_label')
    if label is not None and (type(label) is not int or label not in ROAD_CLASSES):
        raise ValueError('Invalid proposed_label')
    confidence = str(item.get('confidence', 'low')).lower()
    if confidence not in ('high', 'medium', 'low'):
        confidence = 'low'
    return {'proposed_label': label, 'confidence': confidence,
            'evidence': str(item.get('evidence', ''))[:500],
            'reason': str(item.get('reason', ''))[:500]}


def propose(row, paths, api_key, model, max_width, timeout, retries):
    content = [{'type': 'text', 'text': PROMPT}]
    padded = len(paths) - len(set(paths))
    if padded:
        content.append({'type': 'text', 'text': f'本段位于 scene 开头，历史不足，最早图像重复了 {padded} 次；请只依据实际可见变化和最后一帧判断。'})
    for i, path in enumerate(paths, 1):
        content.append({'type': 'text', 'text': f'第 {i}/4 帧；第 4 帧为当前时刻。'})
        content.append({'type': 'image_url', 'image_url': {'url': encode_image(path, max_width)}})
    payload = {'model': model, 'messages': [{'role': 'user', 'content': content}],
               'enable_thinking': False, 'temperature': 0, 'max_tokens': 350}
    base = {k: row[k] for k in ('sample_token', 'scene_token', 'scene', 'timestamp', 'cam_front_path')}
    base.update({'image_paths': paths, 'model': model, 'policy_version': POLICY_VERSION,
                 'review_status': 'unverified_model_proposal', 'history_padded_count': padded})
    for attempt in range(retries):
        try:
            response = requests.post(ENDPOINT, headers={'Authorization': f'Bearer {api_key}'},
                                     json=payload, timeout=timeout)
            if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                time.sleep(2 ** attempt)
                continue
            if response.status_code != 200:
                base.update({'status': 'api_error', 'http_status': response.status_code,
                             'error': response.text[:300]})
                return base
            data = response.json()
            content_text = data['choices'][0]['message']['content']
            parsed = parse_answer(content_text)
            base.update(parsed)
            base.update({'status': 'ok', 'usage': data.get('usage', {})})
            return base
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
                continue
            base.update({'status': 'request_error', 'error': f'{type(exc).__name__}: {str(exc)[:200]}'})
            return base
    return base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', default='outputs/road_head/road_labels_template.jsonl')
    p.add_argument('--output', default='outputs/road_head/road_labels_qwen_proposals.jsonl')
    p.add_argument('--nuscenes-root', default='/data/nuscenes')
    p.add_argument('--version', default='auto')
    p.add_argument('--num-frames', type=int, default=4)
    p.add_argument('--frame-stride', type=int, default=1)
    p.add_argument('--pad-history', action='store_true',
                   help='Repeat the first scene frame for targets with insufficient history')
    p.add_argument('--camera', default='CAM_FRONT')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--max-width', type=int, default=800)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--timeout', type=int, default=90)
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--limit', type=int, default=0, help='process only this many new rows; 0 means all')
    args = p.parse_args()
    if args.num_frames != 4:
        p.error('This annotation prompt currently requires exactly four frames')
    api_key = os.getenv('QWEN_API_KEY')
    if not api_key:
        p.error('QWEN_API_KEY is absent. Source ~/.bashrc before running.')
    rows = [json.loads(line) for line in Path(args.template).read_text().splitlines() if line.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if output.is_file():
        for line in output.read_text().splitlines():
            if line.strip():
                value = json.loads(line)
                existing[value['sample_token']] = value
    pending = [row for row in rows if row['sample_token'] not in existing or existing[row['sample_token']]['status'] != 'ok']
    if args.limit:
        pending = pending[:args.limit]
    print(f'Template={len(rows)} existing_ok={sum(v["status"] == "ok" for v in existing.values())} pending={len(pending)}', flush=True)
    nusc = load_nuscenes(args.nuscenes_root, args.version)
    sequences = [(row, sequence_for_target(nusc, row, args.num_frames, args.frame_stride,
                                           args.camera, args.pad_history))
                 for row in pending]
    # Only the main thread writes JSONL, so partial runs remain resumable.
    with ThreadPoolExecutor(max_workers=args.workers) as pool, output.open('a') as out:
        futures = {pool.submit(propose, row, paths, api_key, args.model,
                               args.max_width, args.timeout, args.retries): row
                   for row, paths in sequences}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            out.write(json.dumps(result, ensure_ascii=False) + '\n')
            out.flush()
            print(f'{done}/{len(futures)} {result["sample_token"]} status={result.get("status")} '
                  f'label={result.get("proposed_label")} conf={result.get("confidence")}', flush=True)
    print(f'Wrote proposals to {output}; these are NOT ground truth.', flush=True)


if __name__ == '__main__':
    main()
