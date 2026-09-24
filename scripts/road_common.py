"""Shared data, VGGT loading, metrics and visualization for road scripts."""
import json
import gc
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from vggt.models.vggt import VGGT
from vggt.heads.road_probability_head import VGGTRoadClassifier, ROAD_CLASSES
from training.data.nuscenes_road_dataset import (load_nuscenes, read_annotations, build_records,
                                                   split_records, class_counts, NuScenesRoadSequenceDataset)


def config(path):
    return yaml.safe_load(Path(path).read_text())


def data_splits(cfg):
    d = cfg['data']
    if not d.get('scene_split', True):
        raise ValueError('Scene-level split is required to avoid adjacent-frame leakage')
    nusc = load_nuscenes(d['root'], d.get('version', 'auto'))
    ann = read_annotations(d['annotation'])
    records, stats = build_records(nusc, ann, d['num_frames'], d['frame_stride'], d['camera'],
                                   d['max_sequences'], d.get('pad_history', False))
    train, val = split_records(records, d['val_fraction'], cfg['training']['seed'])
    counts = class_counts(train)
    prior = (counts + 1e-6) / (counts.sum() + 5e-6)
    print(f'Dataset: {stats}; train={len(train)} / {len(set(r["scene_token"] for r in train))} scenes; '
          f'val={len(val)} / {len(set(r["scene_token"] for r in val))} scenes; train classes={counts.tolist()}', flush=True)
    if len(val) < 50:
        print('WARNING: validation set too small for reliable calibration', flush=True)
    return train, val, prior, stats


def loader(records, cfg, train=False, sampler=None):
    return DataLoader(NuScenesRoadSequenceDataset(records, cfg['data']['preprocess_mode'],
                      cfg['data'].get('color_jitter', 0) if train else 0,
                      cfg['data'].get('image_width', 518)),
                      batch_size=cfg['training']['batch_size'], shuffle=train and sampler is None,
                      sampler=sampler,
                      num_workers=cfg['training']['num_workers'], pin_memory=True,
                      persistent_workers=cfg['training']['num_workers'] > 0,
                      collate_fn=road_collate)


def road_collate(items):
    targets = [item['target'] for item in items]
    batch = default_collate([{k: v for k, v in item.items() if k != 'target'} for item in items])
    if all(t.ndim == 0 for t in targets):
        batch['target'] = torch.stack(targets)
    else:
        # For a mixed hard/soft batch, one-hot hard CE is mathematically identical
        # to ordinary hard CE, so a single soft-CE expression is valid.
        batch['target'] = torch.stack([F.one_hot(t.long(), 5).float() if t.ndim == 0 else t
                                       for t in targets])
    return batch


def _amp_dtype(amp):
    if amp == 'bf16':
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            raise ValueError('bf16 requested but not supported by the current GPU')
        return torch.bfloat16
    if amp == 'fp16':
        return torch.float16
    if amp == 'none':
        return torch.float32
    raise ValueError(f'Unknown AMP mode: {amp}')


def _new_vggt():
    return VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)


def _local_aggregator(base, path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'VGGT checkpoint not found: {path}')
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    for key in ('model', 'state_dict', 'model_state_dict'):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
    if not isinstance(state, dict):
        raise TypeError('VGGT checkpoint must contain a tensor state_dict')

    expected = base.aggregator.state_dict()
    loaded = {}
    wrappers = ('module.', '_orig_mod.', 'model.', 'vggt.', 'backbone.')
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        while key.startswith(wrappers):
            key = next(key[len(prefix):] for prefix in wrappers if key.startswith(prefix))
        key = key.removeprefix('aggregator.')
        if key in expected:
            if key in loaded:
                raise ValueError(f'Duplicate VGGT aggregator key: {key}')
            loaded[key] = value
    missing = sorted(expected.keys() - loaded.keys())
    wrong_shape = sorted(key for key in expected.keys() & loaded.keys()
                         if expected[key].shape != loaded[key].shape)
    if missing or wrong_shape:
        raise RuntimeError(f'VGGT aggregator checkpoint mismatch: '
                           f'missing={missing[:8]}, wrong_shape={wrong_shape[:8]}')
    base.aggregator.load_state_dict(loaded, strict=True)
    print(f'Loaded {len(loaded)} VGGT aggregator tensors from {path}', flush=True)
    return base


def _hub_aggregator(cfg):
    if not torch.cuda.is_available():
        return VGGT.from_pretrained('facebook/VGGT-1B', enable_camera=False,
                                    enable_point=False, enable_depth=False, enable_track=False)

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download('facebook/VGGT-1B', 'model.safetensors')
    device = torch.device('cuda', torch.cuda.current_device())
    base = _new_vggt().to(device=device, dtype=_amp_dtype(cfg['training']['amp']))
    if cfg['training']['finetune_mode'] == 'last_blocks':
        n = cfg['training']['last_blocks']
        if not 1 <= n <= base.aggregator.depth:
            raise ValueError('last_blocks must be between 1 and aggregator depth')
        for blocks in (base.aggregator.frame_blocks, base.aggregator.global_blocks):
            for block in blocks[-n:]:
                block.float()
    with safe_open(path, framework='pt', device='cpu') as weights, torch.no_grad():
        available = set(weights.keys())
        tensors = base.aggregator.state_dict()
        missing = [key for key in tensors if f'aggregator.{key}' not in available]
        if missing:
            raise RuntimeError(f'Pretrained VGGT is missing aggregator tensors: {missing[:8]}')
        for key, destination in tensors.items():
            destination.copy_(weights.get_tensor(f'aggregator.{key}').to(
                device=destination.device, dtype=destination.dtype))
    print(f'Loaded pretrained VGGT aggregator from {path}', flush=True)
    return base


def make_model(cfg, prior=None, checkpoint=None):
    finetuned = (torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
                 if checkpoint else None)
    if finetuned is not None and not isinstance(finetuned, dict):
        raise TypeError('Road checkpoint must be a dictionary')
    pretrained_path = cfg['model'].get('checkpoint') or (
        finetuned.get('pretrained_path') if finetuned else None)
    base = (_local_aggregator(_new_vggt(), pretrained_path) if pretrained_path
            else _hub_aggregator(cfg))
    model = VGGTRoadClassifier(base.aggregator, cfg['model']['road_head'],
                               cfg['training']['finetune_mode'], cfg['training']['last_blocks'],
                               class_prior=prior, use_prior_correction=cfg['hmm']['use_prior_correction'])
    if finetuned is not None:
        model.road_head.load_state_dict(finetuned['road_head'], strict=True)
        updates = finetuned.get('aggregator_trainable', {})
        trained_cfg = (finetuned.get('config') or {}).get('training', {})
        if trained_cfg.get('finetune_mode') == 'last_blocks' and not updates:
            raise RuntimeError('Fine-tuned checkpoint has no updated VGGT block weights')
        if updates:
            expected = model.aggregator.state_dict()
            invalid = sorted(key for key in updates if key not in expected)
            if invalid:
                raise RuntimeError(f'Unexpected aggregator update keys: {invalid[:8]}')
            if trained_cfg.get('finetune_mode') == 'last_blocks':
                n = trained_cfg['last_blocks']
                first = model.aggregator.depth - n
                prefixes = tuple(f'{block}.{index}.'
                                 for block in ('frame_blocks', 'global_blocks')
                                 for index in range(first, model.aggregator.depth))
                trained_names = {name for name, _ in model.aggregator.named_parameters()
                                 if name.startswith(prefixes)}
                missing = sorted(trained_names - updates.keys())
                if missing:
                    raise RuntimeError(f'Fine-tuned checkpoint is missing block weights: {missing[:8]}')
            model.aggregator.load_state_dict(updates, strict=False)
        model.temperature = float(finetuned.get('temperature', 1.0))
        if finetuned.get('class_prior') is not None:
            model.class_prior.copy_(torch.as_tensor(finetuned['class_prior']))
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.road_head.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: total={total:,}, road_head={head:,}, trainable={trainable:,}, '
          f'frozen={total-trainable:,}', flush=True)
    return model


def place_model(model, device, cfg):
    device = torch.device(device)
    if device.type == 'cuda' and next(model.aggregator.parameters()).device.type == 'cpu':
        model.aggregator.to(dtype=_amp_dtype(cfg['training']['amp']))
        if model.finetune_mode == 'last_blocks':
            n = cfg['training']['last_blocks']
            for blocks in (model.aggregator.frame_blocks, model.aggregator.global_blocks):
                for block in blocks[-n:]:
                    block.float()  # AdamW updates trainable weights in fp32.
        gc.collect()
    return model.to(device)


def parameter_groups(model, head_lr, backbone_lr):
    head = [parameter for parameter in model.road_head.parameters() if parameter.requires_grad]
    blocks = [parameter for parameter in model.aggregator.parameters() if parameter.requires_grad]
    if not head or (model.finetune_mode == 'last_blocks' and not blocks):
        raise RuntimeError('The requested road-head or VGGT blocks are frozen')
    selected = {id(parameter) for parameter in head + blocks}
    omitted = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and id(parameter) not in selected]
    if omitted:
        raise RuntimeError(f'Trainable parameters missing from optimizer: {omitted[:8]}')
    groups = [{'params': head, 'lr': head_lr}]
    if blocks:
        groups.append({'params': blocks, 'lr': backbone_lr})
    return groups, head, blocks


def target_prob(target):
    return F.one_hot(target.long(), 5).float() if target.ndim == 1 else target.float()


def loss_fn(logits, target):
    if target.ndim == 1:
        return F.cross_entropy(logits.float(), target.long())
    return -(target * F.log_softmax(logits.float(), -1)).sum(-1).mean()


def autocast_context(device, amp):
    if device.type != 'cuda':
        return nullcontext()
    dtype = _amp_dtype(amp)
    return nullcontext() if dtype == torch.float32 else torch.autocast('cuda', dtype=dtype)


def collect(model, batches, device, amp, debug_first=False):
    model.eval()
    logits, targets, samples = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(batches):
            images = batch['images'].to(device)
            with autocast_context(device, amp):
                out = model(images, debug=debug_first and i == 0)
            if debug_first and i == 0:
                for key in ('images_shape', 'aggregated_tokens_shape', 'patch_start_idx',
                            'patch_tokens_shape', 'frame_road_features_shape',
                            'temporal_features_shape'):
                    print(key, out[key])
                for key in ('road_embedding', 'road_logits', 'road_prob'):
                    print(key, tuple(out[key].shape))
                assert out['road_logits'].shape == (images.shape[0], 5)
                assert torch.allclose(out['road_prob'].sum(-1), torch.ones(images.shape[0], device=device), atol=1e-4)
            logits.append(out['road_logits'].float().cpu())
            targets.append(target_prob(batch['target']).cpu())
            samples.extend(batch['sample_token'])
    return torch.cat(logits), torch.cat(targets), samples


def metrics(logits, targets, tau=1.0, bins=10):
    probs = (logits.float() / tau).softmax(-1)
    truth, pred = targets.argmax(-1), probs.argmax(-1)
    cm = torch.bincount(truth * 5 + pred, minlength=25).reshape(5, 5)
    precision = cm.diag() / cm.sum(0).clamp_min(1)
    recall = cm.diag() / cm.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    nll = -(targets * probs.clamp_min(1e-12).log()).sum(-1).mean()
    brier = ((probs - targets)**2).sum(-1).mean()
    conf = probs.max(-1).values
    correct = pred.eq(truth).float()
    ece = 0.0
    for j in range(bins):
        mask = (conf >= j/bins) & (conf < (j+1)/bins if j < bins-1 else conf <= 1)
        if mask.any():
            ece += (mask.float().mean() * (correct[mask].mean()-conf[mask].mean()).abs()).item()
    return {'accuracy': correct.mean().item(), 'macro_f1': f1.mean().item(),
            'per_class_precision': precision.tolist(), 'per_class_recall': recall.tolist(),
            'confusion_matrix': cm.tolist(), 'nll': nll.item(), 'brier': brier.item(), 'ece': ece}


def save_plots(logits, targets, tau, outdir, bins=10):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    m = metrics(logits, targets, tau, bins)
    (outdir/'metrics.json').write_text(json.dumps(m, indent=2))
    fig, ax = plt.subplots()
    ax.imshow(m['confusion_matrix'], cmap='Blues')
    ax.set(xlabel='predicted', ylabel='target', xticks=range(5), yticks=range(5))
    fig.tight_layout(); fig.savefig(outdir/'confusion_matrix.png'); plt.close(fig)
    probs = (logits/tau).softmax(-1)
    conf, pred = probs.max(-1)
    correct = pred.eq(targets.argmax(-1)).float()
    xs, ys, counts = [], [], []
    for j in range(bins):
        mask = (conf >= j/bins) & (conf < (j+1)/bins if j < bins-1 else conf <= 1)
        if mask.any():
            xs.append(conf[mask].mean().item()); ys.append(correct[mask].mean().item()); counts.append(mask.sum().item())
    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1], '--', color='gray')
    ax.scatter(xs, ys, s=np.asarray(counts)*5)
    ax.set(xlim=(0,1), ylim=(0,1), xlabel='confidence', ylabel='accuracy', title=f'ECE={m["ece"]:.3f}')
    fig.tight_layout(); fig.savefig(outdir/'reliability_diagram.png'); plt.close(fig)
    return m


def save_attention(model, batch, device, amp, outdir, stem='val0'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad(), autocast_context(device, amp):
        out = model(batch['images'][:1].to(device), debug=True)
    spatial = out['spatial_attention'][0].float().cpu().numpy()
    temporal = out['temporal_attention'][0].float().cpu().numpy()
    h, w = batch['images'].shape[-2:]
    grid = (h//model.aggregator.patch_size, w//model.aggregator.patch_size)
    fig, axes = plt.subplots(1, len(spatial), figsize=(4*len(spatial), 3))
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(batch['images'][0, i].permute(1, 2, 0).cpu().numpy())
        heat = spatial[i, -grid[0]*grid[1]:].reshape(grid)
        ax.imshow(heat, cmap='jet', alpha=0.45, extent=(0,w,h,0), interpolation='bilinear')
        ax.axis('off'); ax.set_title(f'frame {i}')
    fig.tight_layout(); fig.savefig(outdir/f'{stem}_spatial.png'); plt.close(fig)
    fig, ax = plt.subplots(); ax.bar(range(len(temporal)), temporal)
    ax.set(xlabel='frame', ylabel='attention weight', ylim=(0,1))
    fig.tight_layout(); fig.savefig(outdir/f'{stem}_temporal.png'); plt.close(fig)
