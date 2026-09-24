import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.data.nuscenes_road_dataset import load_nuscenes

p = argparse.ArgumentParser(description='Generate UNLABELED annotation template; fill labels manually.')
p.add_argument('--nuscenes-root', default='/data/nuscenes')
p.add_argument('--version', default='auto')
p.add_argument('--output', default='/data/nuscenes/road_labels_template.jsonl')
p.add_argument('--max-samples', type=int, default=500)
p.add_argument('--num-frames', type=int, default=4)
p.add_argument('--frame-stride', type=int, default=1)
p.add_argument('--include-short-history', action='store_true',
               help='Include the first frames of every scene; labeler can pad their missing history')
a = p.parse_args()
nusc = load_nuscenes(a.nuscenes_root, a.version)
rows = []
for scene in sorted(nusc.scene, key=lambda x: x['name']):
    chain = []
    token = scene['first_sample_token']
    while token:
        s = nusc.get('sample', token)
        chain.append(s)
        token = s['next']
    first_index = 0 if a.include_short_history else (a.num_frames-1)*a.frame_stride
    for i in range(first_index, len(chain)):
        s = chain[i]
        sd = nusc.get('sample_data', s['data']['CAM_FRONT'])
        path = Path(nusc.dataroot) / sd['filename']
        if path.is_file():
            rows.append({'sample_token': s['token'], 'scene_token': scene['token'],
                         'scene': scene['name'], 'timestamp': s['timestamp'], 'cam_front_path': str(path),
                         'history_available_frames': 1 + i // a.frame_stride})
    if len(rows) >= a.max_samples:
        break
rows = rows[:a.max_samples]
output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(''.join(json.dumps(r)+'\n' for r in rows))
print(f'Wrote {len(rows)} UNLABELED rows from {len(set(r["scene_token"] for r in rows))} scenes to {output}')
print('Fill each row with label: 0..4 or soft_label: [five probabilities]. Do not train until real labels are supplied.')
