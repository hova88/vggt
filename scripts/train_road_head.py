"""Train the road head with scene-disjoint NuScenes train/validation splits."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.road_common import (  # noqa: E402
    autocast_context, collect, config, data_splits, loader, loss_fn,
    make_model, parameter_groups, place_model, save_plots,
)
from vggt.heads.road_probability_head import ROAD_CLASSES  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/road_head_nuscenes_small.yaml')
    parser.add_argument('--pretrained', help='Local VGGT model.pt; overrides model.checkpoint')
    parser.add_argument('--max-steps', type=int, default=0)
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error('--max-steps must be nonnegative')

    cfg = config(args.config)
    if args.pretrained:
        cfg['model']['checkpoint'] = str(Path(args.pretrained).expanduser().resolve(strict=True))
    accumulation = cfg['training']['accum_steps']
    if accumulation < 1:
        raise ValueError('training.accum_steps must be at least 1')
    seed = cfg['training']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train, val, prior, _ = data_splits(cfg)
    if not train or not val:
        raise RuntimeError('Empty train or validation split')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = place_model(make_model(cfg, prior), device, cfg)
    groups, head_params, block_params = parameter_groups(
        model, cfg['training']['head_lr'], cfg['training']['backbone_lr'])
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['training']['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda' and
                                      cfg['training']['amp'] == 'fp16')
    trainable = head_params + block_params
    train_loader, val_loader = loader(train, cfg, True), loader(val, cfg)
    outdir = Path(cfg['training']['output_dir'])
    outdir.mkdir(parents=True, exist_ok=True)
    best_nll, global_step = float('inf'), 0

    for epoch in range(1, cfg['training']['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = correct = count = 0
        for i, batch in enumerate(train_loader):
            images = batch['images'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            with autocast_context(device, cfg['training']['amp']):
                logits = model(images)['road_logits']
                loss = loss_fn(logits, target)
            window_size = min(accumulation, len(train_loader) -
                              (i // accumulation) * accumulation)
            scaler.scale(loss / window_size).backward()
            if (i + 1) % accumulation == 0 or i + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, cfg['training']['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            labels = target.argmax(-1) if target.ndim == 2 else target
            total_loss += loss.item() * len(images)
            correct += (logits.argmax(-1) == labels).sum().item()
            count += len(images)
            if args.max_steps and global_step >= args.max_steps:
                break

        val_logits, val_targets, _ = collect(model, val_loader, device,
                                              cfg['training']['amp'])
        metrics = save_plots(val_logits, val_targets, 1.0, outdir,
                             cfg['calibration']['ece_bins'])
        print(f'epoch={epoch} train_loss={total_loss/count:.4f} '
              f'train_acc={correct/count:.3f} val={metrics}', flush=True)
        if metrics['nll'] < best_nll:
            best_nll = metrics['nll']
            trainable_names = {name for name, p in model.aggregator.named_parameters()
                               if p.requires_grad}
            torch.save({
                'road_head': {name: tensor.detach().cpu() for name, tensor in
                              model.road_head.state_dict().items()},
                'aggregator_trainable': {
                    name: tensor.detach().cpu() for name, tensor in
                    model.aggregator.state_dict().items() if name in trainable_names
                },
                'pretrained_path': cfg['model'].get('checkpoint'),
                'epoch': epoch,
                'global_step': global_step,
                'config': cfg,
                'class_mapping': {str(k): v for k, v in ROAD_CLASSES.items()},
                'class_prior': prior.tolist(),
                'temperature': 1.0,
            }, outdir / 'best.pt')
        if args.max_steps and global_step >= args.max_steps:
            break


if __name__ == '__main__':
    main()
