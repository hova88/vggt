import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from scripts.road_common import config, data_splits, loader, make_model, place_model, collect, metrics
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/road_head_nuscenes_small.yaml'); p.add_argument('--checkpoint',required=True)
a=p.parse_args(); cfg=config(a.config)
_,val,prior,_=data_splits(cfg)
if len(val)<50: print('WARNING: validation set too small for reliable calibration')
model=place_model(make_model(cfg,prior,a.checkpoint),'cuda' if torch.cuda.is_available() else 'cpu',cfg)
logits,targets,_=collect(model,loader(val,cfg),next(model.parameters()).device,cfg['training']['amp'])
log_tau=torch.nn.Parameter(torch.zeros(())); opt=torch.optim.LBFGS([log_tau],lr=0.1,max_iter=100,line_search_fn='strong_wolfe')
def closure():
    opt.zero_grad(); loss=-(targets*F.log_softmax(logits/log_tau.exp(),-1)).sum(-1).mean(); loss.backward(); return loss
opt.step(closure); tau=log_tau.exp().item()
output=Path(cfg['training']['output_dir'])/'temperature.json'; output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps({'temperature':tau,'validation_sequences':len(val),'before':metrics(logits,targets),'after':metrics(logits,targets,tau)},indent=2))
print(output.read_text())
