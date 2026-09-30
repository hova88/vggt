"""Scene-safe NuScenes CAM_FRONT sequences with externally supplied road labels."""
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from vggt.utils.load_fn import load_and_preprocess_images

ROAD_CLASSES = {0: 'elevated_up', 1: 'elevated_down', 2: 'main_road', 3: 'side_road', 4: 'intersection'}


def detect_nuscenes(root, version='auto'):
    root = Path(root)
    candidates = [version] if version != 'auto' else ['v1.0-mini', 'v1.0-trainval']
    for name in candidates:
        # Mini can be symlinked into a trainval root while images live under mini/.
        roots = [root / 'mini', root] if name == 'v1.0-mini' else [root]
        for data_root in roots:
            if (data_root / name / 'scene.json').is_file() and (data_root / 'samples' / 'CAM_FRONT').is_dir():
                return str(data_root), name
    raise FileNotFoundError(f'No complete NuScenes version with CAM_FRONT images under {root}')


def read_annotations(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Road annotation file not found: {path}')
    annotations = {}
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f'{path}:{line_no}: annotation must be an object')
        token = row.get('sample_token')
        if not isinstance(token, str) or not token:
            raise ValueError(f'{path}:{line_no}: invalid sample_token')
        has_hard, has_soft = 'label' in row, 'soft_label' in row
        if has_hard == has_soft:
            raise ValueError(f'{path}:{line_no}: provide exactly one of label or soft_label')
        if has_hard:
            if type(row['label']) is not int or row['label'] not in ROAD_CLASSES:
                raise ValueError(f'{path}:{line_no}: invalid hard label')
        else:
            vals = row['soft_label']
            if (not isinstance(vals, list) or len(vals) != 5 or
                    any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                        for v in vals) or abs(sum(vals) - 1) > 1e-3):
                raise ValueError(f'{path}:{line_no}: invalid soft label')
        if token in annotations:
            raise ValueError(f'Duplicate annotation: {token}')
        annotations[token] = row
    return annotations


def load_nuscenes(root, version='auto'):
    from nuscenes.nuscenes import NuScenes
    dataroot, version = detect_nuscenes(root, version)
    print(f'NuScenes: {version} at {dataroot}', flush=True)
    return NuScenes(version=version, dataroot=dataroot, verbose=False)


def build_records(nusc, annotations, num_frames=4, frame_stride=1, camera='CAM_FRONT',
                  max_sequences=500, pad_history=False):
    if not annotations:
        raise ValueError('No labeled road annotations were found')
    records, stats = [], Counter()
    scene_names = {x['token']: x['name'] for x in nusc.scene}
    for scene in sorted(nusc.scene, key=lambda x: x['name']):
        chain = []
        sample_token = scene['first_sample_token']
        while sample_token:
            sample = nusc.get('sample', sample_token)
            sd = nusc.get('sample_data', sample['data'][camera])
            chain.append((sample, sd))
            sample_token = sample['next']
        for i, (sample, sd) in enumerate(chain):
            stats['all_targets'] += 1
            row = annotations.get(sample['token'])
            if row is None:
                stats['missing_annotation'] += 1
                continue
            indices = [i - j * frame_stride for j in range(num_frames-1, -1, -1)]
            if indices[0] < 0 and not pad_history:
                stats['short_history'] += 1
                continue
            indices = [max(0, j) for j in indices]
            frames = [chain[j][1] for j in indices]
            paths = [str(Path(nusc.dataroot) / f['filename']) for f in frames]
            if any(not Path(p).is_file() for p in paths):
                stats['missing_image'] += 1
                continue
            if 'scene_token' in row and row['scene_token'] != scene['token']:
                raise ValueError(f'Scene token mismatch for {sample["token"]}')
            records.append({'sample_token': sample['token'], 'scene_token': scene['token'],
                            'scene_name': scene_names[scene['token']], 'timestamps': [f['timestamp'] for f in frames],
                            'image_paths': paths, 'annotation': row})
    if max_sequences and len(records) > max_sequences:
        # Spread capped examples across scenes so validation remains possible.
        buckets = defaultdict(list)
        for r in records:
            buckets[r['scene_token']].append(r)
        selected = []
        while len(selected) < max_sequences and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and len(selected) < max_sequences:
                    selected.append(buckets[key].pop(0))
        records = selected
    stats['sequences'] = len(records)
    stats['scenes'] = len({r['scene_token'] for r in records})
    return records, dict(stats)


def split_records(records, val_fraction=0.2, seed=42):
    scenes = sorted({r['scene_token'] for r in records})
    if len(scenes) < 2:
        raise ValueError('At least two annotated scenes are needed for scene-level train/val split')
    random.Random(seed).shuffle(scenes)
    val_scenes = set(scenes[:max(1, round(len(scenes)*val_fraction))])
    return ([r for r in records if r['scene_token'] not in val_scenes],
            [r for r in records if r['scene_token'] in val_scenes])


def class_counts(records):
    counts = torch.zeros(5)
    for r in records:
        row = r['annotation']
        counts += torch.tensor(row['soft_label']) if 'soft_label' in row else torch.nn.functional.one_hot(torch.tensor(row['label']), 5)
    return counts


class NuScenesRoadSequenceDataset(Dataset):
    def __init__(self, records, preprocess_mode='crop', color_jitter=0.0, image_width=518):
        self.records = records
        self.preprocess_mode = preprocess_mode
        self.color_jitter = color_jitter
        self.image_width = image_width

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        images = load_and_preprocess_images(r['image_paths'], mode=self.preprocess_mode)
        if self.image_width != 518:
            if self.image_width % 14:
                raise ValueError('image_width must be divisible by VGGT patch size 14')
            import torch.nn.functional as F
            height = round(images.shape[-2] * self.image_width / images.shape[-1] / 14) * 14
            images = F.interpolate(images, size=(height, self.image_width), mode='bicubic', align_corners=False).clamp(0, 1)
        if self.color_jitter:
            from torchvision.transforms import ColorJitter
            images = ColorJitter(brightness=self.color_jitter, contrast=self.color_jitter,
                                 saturation=self.color_jitter)(images)
        row = r['annotation']
        target = torch.tensor(row['soft_label'], dtype=torch.float32) if 'soft_label' in row else torch.tensor(row['label'], dtype=torch.long)
        return {'images': images, 'target': target, 'sample_token': r['sample_token'],
                'scene_token': r['scene_token'], 'timestamps': r['timestamps'],
                'image_paths': r['image_paths']}
