import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.road_common import (config, data_splits, loader, make_model, loss_fn,
                                 place_model, autocast_context, collect, save_plots, save_attention)
from training.data.nuscenes_road_dataset import NuScenesRoadSequenceDataset
from torch.utils.data import DataLoader

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/road_head_nuscenes_small.yaml')
p.add_argument('--dummy-labels', action='store_true', help='NO training meaning; smoke test only')
p.add_argument('--random-vggt', action='store_true', help='NO training meaning; smoke test only')
p.add_argument('--smoke', action='store_true', help='one batch, one optimization step')
p.add_argument('--visualize-smoke', action='store_true', help='save attention maps in smoke mode')
p.add_argument('--tiny-overfit', action='store_true')
p.add_argument('--max-steps', type=int, default=0)
a = p.parse_args()
cfg = config(a.config)
if a.dummy_labels and not a.smoke:
    p.error('--dummy-labels is restricted to --smoke; it cannot produce a training checkpoint')
if a.random_vggt and not a.smoke:
    p.error('--random-vggt is restricted to --smoke')
random.seed(cfg['training']['seed']); np.random.seed(cfg['training']['seed']); torch.manual_seed(cfg['training']['seed'])
train, val, prior, stats = data_splits(cfg, a.dummy_labels)
if a.tiny_overfit:
    train = train[:min(32, len(train))]; val = train
if not train or not val: raise RuntimeError('Empty train or val split')
sample = NuScenesRoadSequenceDataset(train[:1], cfg['data']['preprocess_mode'],
                                     image_width=cfg['data'].get('image_width', 518))[0]
print('Dataset sample:', {k: sample[k] for k in ('scene_token','sample_token','image_paths','timestamps','target')},
      'image tensor shape:', tuple(sample['images'].shape), flush=True)
from PIL import Image
with Image.open(sample['image_paths'][0]) as im: print('Preprocess:', im.size, '->', tuple(sample['images'].shape[-2:]),
    'mode=', cfg['data']['preprocess_mode'],
    'VGGT resize to width 518 + center crop/pad; optional bicubic downscale to',
    cfg['data'].get('image_width', 518), flush=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = place_model(make_model(cfg, prior, pretrained=not a.random_vggt), device, cfg)
if a.smoke:
    first = next(iter(loader(train[:1], cfg)))
    model.eval()
    with torch.no_grad(), autocast_context(device, cfg['training']['amp']):
        out = model(first['images'].to(device), debug=True)
    for key in ('images_shape','aggregated_tokens_shape','patch_start_idx','patch_tokens_shape',
                'frame_road_features_shape','temporal_features_shape'):
        print(key, out[key], flush=True)
    for key in ('road_embedding','road_logits','road_prob'):
        print(key, tuple(out[key].shape), flush=True)
    assert out['road_logits'].shape == (1,5)
    assert torch.allclose(out['road_prob'].sum(-1), torch.ones(1,device=device), atol=1e-4)
    model.train(); optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                                 lr=cfg['training']['head_lr'])
    optimizer.zero_grad()
    with autocast_context(device, cfg['training']['amp']):
        loss = loss_fn(model(first['images'].to(device))['road_logits'], first['target'].to(device))
    loss.backward(); optimizer.step()
    print(f'SMOKE ONLY: one gradient step succeeded, loss={loss.item():.4f}', flush=True)
    if a.visualize_smoke:
        save_attention(model, first, device, cfg['training']['amp'], cfg['training']['output_dir'], 'dummy_smoke')
        print('SMOKE ONLY: saved attention visualizations', flush=True)
    if device.type == 'cuda':
        print('peak_gpu_memory_MiB=', torch.cuda.max_memory_allocated()/2**20, flush=True)
    sys.exit(0)
head_params = list(model.road_head.parameters())
block_params = [p for p in model.aggregator.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW([{'params': head_params, 'lr': cfg['training']['head_lr']},
                               {'params': block_params, 'lr': cfg['training']['backbone_lr']}],
                              weight_decay=cfg['training']['weight_decay'])
scaler = torch.cuda.amp.GradScaler(enabled=device.type=='cuda' and cfg['training']['amp']=='fp16')
train_loader, val_loader = loader(train, cfg, True), loader(val, cfg)
outdir = Path(cfg['training']['output_dir']); outdir.mkdir(parents=True, exist_ok=True)
best_nll, global_step = float('inf'), 0
if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
for epoch in range(cfg['training']['epochs']):
    model.train(); optimizer.zero_grad(); total_loss=0; correct=0; count=0
    for i, batch in enumerate(train_loader):
        images, target = batch['images'].to(device), batch['target'].to(device)
        with autocast_context(device, cfg['training']['amp']):
            logits = model(images)['road_logits']; loss = loss_fn(logits, target)
        scaler.scale(loss/cfg['training']['accum_steps']).backward()
        if (i+1)%cfg['training']['accum_steps']==0 or i+1==len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['training']['grad_clip'])
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); global_step += 1
        total_loss += loss.item()*len(images); correct += (logits.argmax(-1)==(target.argmax(-1) if target.ndim==2 else target)).sum().item(); count += len(images)
        if a.max_steps and global_step>=a.max_steps: break
    val_logits, val_targets, _ = collect(model, val_loader, device, cfg['training']['amp'], debug_first=epoch==0)
    metrics = save_plots(val_logits, val_targets, 1.0, outdir, cfg['calibration']['ece_bins'])
    print(f'epoch={epoch+1} train_loss={total_loss/count:.4f} train_acc={correct/count:.3f} val={metrics}', flush=True)
    if metrics['nll'] < best_nll:
        best_nll = metrics['nll']
        prefixes = tuple(f'{b}.{j}.' for b in ('frame_blocks', 'global_blocks')
                         for j in range(model.aggregator.depth-cfg['training']['last_blocks'], model.aggregator.depth))
        trainable_state = {k: v.cpu() for k, v in model.aggregator.state_dict().items()
                           if k.startswith(prefixes)} if block_params else {}
        torch.save({'road_head': model.road_head.state_dict(), 'aggregator_trainable': trainable_state,
                    'optimizer': optimizer.state_dict(), 'epoch': epoch+1, 'global_step': global_step,
                    'config': cfg, 'class_mapping': {str(k):v for k,v in __import__('vggt.heads.road_probability_head',fromlist=['ROAD_CLASSES']).ROAD_CLASSES.items()},
                    'class_prior': prior.tolist(), 'temperature': 1.0,
                    'dummy': False}, outdir/'best.pt')
    if a.max_steps and global_step>=a.max_steps: break
if device.type=='cuda': print('peak_gpu_memory_MiB=', torch.cuda.max_memory_allocated()/2**20)
if not a.tiny_overfit:
    for j, batch in enumerate(val_loader):
        if j >= 3: break
        save_attention(model, batch, device, cfg['training']['amp'], outdir, f'val{j}')
