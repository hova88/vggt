"""Compare Qwen Flash/Max proposals for Flash-abstained NuScenes road samples."""
import argparse
import json
from collections import Counter
from pathlib import Path


def latest_rows(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            value = json.loads(line)
            rows[value['sample_token']] = value
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flash', default='outputs/road_head/road_labels_qwen_v3_proposals.jsonl')
    p.add_argument('--max', dest='max_path', default='outputs/road_head/road_labels_qwen_max_null23_proposals.jsonl')
    p.add_argument('--output', default='outputs/road_head/qwen_flash_vs_max_null23.json')
    args = p.parse_args()
    flash, maximum = latest_rows(args.flash), latest_rows(args.max_path)
    abstained = [x for x in flash.values() if x.get('status') == 'ok' and x.get('proposed_label') is None]
    comparisons = []
    for f in abstained:
        m = maximum.get(f['sample_token'])
        comparisons.append({
            'sample_token': f['sample_token'], 'scene': f['scene'],
            'cam_front_path': f['cam_front_path'],
            'flash': {k: f.get(k) for k in ('model', 'status', 'proposed_label', 'confidence', 'evidence', 'reason')},
            'max': None if m is None else {k: m.get(k) for k in ('model', 'status', 'proposed_label',
                                                                'confidence', 'evidence', 'reason')},
        })
    matched = sum(x['max'] is not None and x['max']['status'] == 'ok' and
                  x['max']['proposed_label'] == x['flash']['proposed_label'] for x in comparisons)
    report = {
        'notice': 'Model proposals only; agreement is not classification accuracy or ground truth.',
        'flash_total': len(flash),
        'flash_distribution': dict(Counter(str(x.get('proposed_label')) for x in flash.values() if x.get('status') == 'ok')),
        'flash_abstained': len(abstained),
        'max_completed': sum(x['max'] is not None and x['max']['status'] == 'ok' for x in comparisons),
        'max_distribution_on_abstained': dict(Counter(str(x['max']['proposed_label']) for x in comparisons
                                                     if x['max'] is not None and x['max']['status'] == 'ok')),
        'agreement_on_flash_abstained': matched,
        'comparisons': comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {output}: Flash abstained={len(abstained)}, Max completed={report["max_completed"]}, agreement={matched}')


if __name__ == '__main__':
    main()
