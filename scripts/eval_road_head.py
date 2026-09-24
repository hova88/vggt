import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.road_common import config, data_splits, loader, make_model, place_model, collect, save_plots, save_attention
p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/road_head_nuscenes_small.yaml'); p.add_argument('--checkpoint', required=True)
a=p.parse_args(); cfg=config(a.config)
_, val, prior, _=data_splits(cfg)
model=place_model(make_model(cfg, prior, a.checkpoint),'cuda' if torch.cuda.is_available() else 'cpu',cfg)
tfile=Path(cfg['training']['output_dir'])/'temperature.json'
if tfile.is_file(): model.temperature=float(json.loads(tfile.read_text())['temperature'])
ld=loader(val,cfg); device=next(model.parameters()).device
logits,targets,_=collect(model,ld,device,cfg['training']['amp'],debug_first=True)
m=save_plots(logits,targets,model.temperature,cfg['training']['output_dir'],cfg['calibration']['ece_bins'])
for j,batch in enumerate(ld):
    if j>=3: break
    save_attention(model,batch,device,cfg['training']['amp'],cfg['training']['output_dir'],f'val{j}')
print(json.dumps(m,indent=2))
