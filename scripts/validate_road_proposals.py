"""Validate a full Qwen proposal JSONL against its CAM_FRONT sample template."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', default='outputs/road_head/road_labels_v1mini_all_samples_template.jsonl')
    p.add_argument('--proposals', default='outputs/road_head/road_labels_v1mini_all_samples_qwen_flash_proposals.jsonl')
    p.add_argument('--output', default='outputs/road_head/v1mini_flash_labeling_summary.json')
    args = p.parse_args()
    template, proposals = read_rows(args.template), read_rows(args.proposals)
    expected = {row['sample_token']: row for row in template}
    actual = {row['sample_token']: row for row in proposals}
    if len(expected) != len(template) or len(actual) != len(proposals):
        raise ValueError('Duplicate sample_token in template or proposals')
    if set(expected) != set(actual):
        raise ValueError(f'Token mismatch: missing={len(set(expected)-set(actual))}, '
                         f'extra={len(set(actual)-set(expected))}')
    for token, row in actual.items():
        source = expected[token]
        if row.get('status') != 'ok' or row.get('review_status') != 'unverified_model_proposal':
            raise ValueError(f'Unsuccessful or incorrectly reviewed proposal: {token}')
        if 'label' in row or 'soft_label' in row:
            raise ValueError(f'Unverified model proposal contains training target: {token}')
        if row['scene_token'] != source['scene_token'] or row['cam_front_path'] != source['cam_front_path']:
            raise ValueError(f'Template metadata mismatch: {token}')
        paths = row.get('image_paths', [])
        if len(paths) != 4 or paths[-1] != source['cam_front_path']:
            raise ValueError(f'Invalid frame sequence: {token}')
        if any('/CAM_FRONT/' not in path for path in paths):
            raise ValueError(f'Non-CAM_FRONT image in sequence: {token}')
        if row.get('history_padded_count', 0) != len(paths) - len(set(paths)):
            raise ValueError(f'Incorrect history padding count: {token}')
        label = row.get('proposed_label')
        if label is not None and (type(label) is not int or label not in range(5)):
            raise ValueError(f'Invalid proposed label: {token}')
    by_scene = defaultdict(list)
    for row in actual.values():
        by_scene[row['scene']].append(row)
    summary = {
        'dataset': 'NuScenes v1.0-mini',
        'scope': 'all CAM_FRONT sample keyframes; sweeps excluded',
        'is_ground_truth': False,
        'review_status': 'unverified_model_proposal',
        'model': sorted({row['model'] for row in actual.values()}),
        'policy_version': sorted({row['policy_version'] for row in actual.values()}),
        'template_rows': len(template),
        'proposal_rows': len(proposals),
        'unique_sample_tokens': len(actual),
        'scenes': len(by_scene),
        'proposed_class_distribution': dict(Counter(str(row.get('proposed_label')) for row in actual.values())),
        'history_padded_count': dict(Counter(str(row.get('history_padded_count', 0)) for row in actual.values())),
        'per_scene': {scene: {'samples': len(rows),
                              'proposed_class_distribution': dict(Counter(str(row.get('proposed_label')) for row in rows))}
                      for scene, rows in sorted(by_scene.items())},
        'total_api_tokens': sum((row.get('usage') or {}).get('total_tokens', 0) for row in actual.values()),
        'known_review_risk': 'View Image audit found false intersection proposals; do not train as ground truth.',
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(f'Validated {len(proposals)} proposals over {len(by_scene)} scenes; summary={output}')


if __name__ == '__main__':
    main()
