"""Shared configuration, model loading, metrics and visualization for road scripts."""
import json
import math
import gc
import os
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


def data_splits(cfg, dummy_labels=False):
    d = cfg['data']
    if not d.get('scene_split', True):
        raise ValueError('Scene-level split is required to avoid adjacent-frame leakage')
    nusc = load_nuscenes(d['root'], d.get('version', 'auto'))
    ann = read_annotations(d['annotation'])
    if dummy_labels:
        print('*** DUMMY LABELS: CODE SMOKE TEST ONLY; NO TRAINING OR METRIC MEANING ***', flush=True)
    records, stats = build_records(nusc, ann, d['num_frames'], d['frame_stride'], d['camera'],
                                   d['max_sequences'], d.get('pad_history', False), dummy_labels)
    train, val = split_records(records, d['val_fraction'], cfg['training']['seed'])
    counts = class_counts(train)
    prior = (counts + 1e-6) / (counts.sum() + 5e-6)
    print(f'Dataset: {stats}; train={len(train)} / {len(set(r["scene_token"] for r in train))} scenes; '
          f'val={len(val)} / {len(set(r["scene_token"] for r in val))} scenes; train classes={counts.tolist()}', flush=True)
    if len(val) < 50:
        print('WARNING: validation set too small for reliable calibration', flush=True)
    return train, val, prior, stats


def loader(records, cfg, train=False):
    return DataLoader(NuScenesRoadSequenceDataset(records, cfg['data']['preprocess_mode'],
                      cfg['data'].get('color_jitter', 0) if train else 0,
                      cfg['data'].get('image_width', 518)),
                      batch_size=cfg['training']['batch_size'], shuffle=train,
                      num_workers=cfg['training']['num_workers'], pin_memory=True,
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


def make_model(cfg, prior=None, checkpoint=None, pretrained=True):
    if pretrained and cfg['model'].get('checkpoint'):
        base = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)
        state = torch.load(cfg['model']['checkpoint'], map_location='cpu')
        state = state.get('model', state.get('state_dict', state))
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        if any(k.startswith('aggregator.') for k in state):
            pass
        elif any(k.startswith('frame_blocks.') for k in state):
            state = {'aggregator.'+k: v for k, v in state.items()}
        missing, unexpected = base.load_state_dict(state, strict=False)
        aggregator_missing = [k for k in missing if k.startswith('aggregator.')]
        if aggregator_missing:
            raise RuntimeError(f'Local checkpoint is missing {len(aggregator_missing)} VGGT aggregator tensors')
        print(f'VGGT checkpoint: missing={len(missing)}, unexpected={len(unexpected)}')
    elif pretrained:
        if torch.cuda.is_available():
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open
            weight_path = hf_hub_download('facebook/VGGT-1B', 'model.safetensors')
            amp = cfg['training']['amp']
            dtype = (torch.bfloat16 if amp == 'bf16' and torch.cuda.is_bf16_supported()
                     else torch.float16 if amp in ('bf16', 'fp16') else torch.float32)
            # Allocate once on GPU, then copy only aggregator tensors one at a time.
            # Loading the 5 GB full checkpoint on CPU first can exhaust shared RAM/VRAM.
            base = VGGT(enable_camera=False, enable_point=False, enable_depth=False,
                        enable_track=False).to(device='cuda', dtype=dtype)
            if cfg['training']['finetune_mode'] == 'last_blocks':
                n = cfg['training']['last_blocks']
                for blocks in (base.aggregator.frame_blocks, base.aggregator.global_blocks):
                    for block in blocks[-n:]:
                        block.float()  # keep trainable weights and AdamW updates in fp32
            state = base.state_dict()
            with safe_open(weight_path, framework='pt', device='cpu') as weights, torch.no_grad():
                available = set(weights.keys())
                missing = [k for k in state if k.startswith('aggregator.') and k not in available]
                if missing:
                    raise RuntimeError(f'Pretrained checkpoint missing aggregator tensors: {missing[:5]}')
                for key, dst in state.items():
                    if key.startswith('aggregator.'):
                        dst.copy_(weights.get_tensor(key).to(device=dst.device, dtype=dst.dtype))
            gc.collect()
            try:
                with open(weight_path, 'rb') as weight_file:
                    os.posix_fadvise(weight_file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
            print(f'Loaded pretrained aggregator from {weight_path} by streamed tensor copy', flush=True)
        else:
            base = VGGT.from_pretrained('facebook/VGGT-1B', enable_camera=False,
                                        enable_point=False, enable_depth=False, enable_track=False)
    else:
        print('*** RANDOM VGGT WEIGHTS: CODE SMOKE TEST ONLY ***', flush=True)
        base = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)
    model = VGGTRoadClassifier(base.aggregator, cfg['model']['road_head'],
                               cfg['training']['finetune_mode'], cfg['training']['last_blocks'],
                               class_prior=prior, use_prior_correction=cfg['hmm']['use_prior_correction'])
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu')
        model.road_head.load_state_dict(state['road_head'])
        if state.get('aggregator_trainable'):
            model.aggregator.load_state_dict(state['aggregator_trainable'], strict=False)
        model.temperature = float(state.get('temperature', 1.0))
        if state.get('class_prior') is not None:
            model.class_prior.copy_(torch.tensor(state['class_prior']))
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.road_head.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: total={total:,}, road_head={head:,}, trainable={trainable:,}, frozen={total-trainable:,}', flush=True)
    return model


def place_model(model, device, cfg):
    device = torch.device(device)
    if device.type == 'cuda' and model.finetune_mode == 'head_only':
        amp = cfg['training']['amp']
        dtype = torch.bfloat16 if amp == 'bf16' and torch.cuda.is_bf16_supported() else torch.float16 if amp in ('bf16', 'fp16') else torch.float32
        if next(model.aggregator.parameters()).device.type != 'cuda':
            model.aggregator.to(dtype=dtype)
            gc.collect()
            model.aggregator.to(device=device)
        model.road_head.to(device=device)
        model.class_prior = model.class_prior.to(device)
        print(f'Frozen VGGT aggregator resident as {dtype}', flush=True)
        return model
    return model.to(device)


def target_prob(target):
    return F.one_hot(target.long(), 5).float() if target.ndim == 1 else target.float()


def loss_fn(logits, target):
    if target.ndim == 1:
        return F.cross_entropy(logits.float(), target.long())
    return -(target * F.log_softmax(logits.float(), -1)).sum(-1).mean()


def autocast_context(device, amp):
    if device.type != 'cuda':
        return nullcontext()
    dtype = torch.bfloat16 if amp == 'bf16' and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast('cuda', dtype=dtype)


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
