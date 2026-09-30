import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from vggt.utils.load_fn import load_and_preprocess_images
from scripts.road_common import config, make_model, place_model, autocast_context
from training.data.nuscenes_road_dataset import load_nuscenes
from vggt.heads.road_probability_head import ROAD_CLASSES
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/road_head_nuscenes_small.yaml'); p.add_argument('--checkpoint',required=True)
p.add_argument('--images',nargs='*',help='ordered oldest to newest; exactly T paths'); p.add_argument('--sample-token',help='use CAM_FRONT history ending at this NuScenes sample')
a=p.parse_args(); cfg=config(a.config)
if bool(a.images)==bool(a.sample_token): p.error('provide exactly one of --images or --sample-token')
if a.images: paths=a.images
else:
    nusc=load_nuscenes(cfg['data']['root'],cfg['data'].get('version','auto')); sample=nusc.get('sample',a.sample_token); chain=[]
    while sample and len(chain)<1+(cfg['data']['num_frames']-1)*cfg['data']['frame_stride']:
        sd=nusc.get('sample_data',sample['data'][cfg['data']['camera']]); chain.append(str(Path(nusc.dataroot)/sd['filename']))
        sample=nusc.get('sample',sample['prev']) if sample['prev'] else None
    if len(chain)<1+(cfg['data']['num_frames']-1)*cfg['data']['frame_stride']: raise ValueError('Insufficient scene history')
    paths=[chain[j] for j in range((cfg['data']['num_frames']-1)*cfg['data']['frame_stride'],-1,-cfg['data']['frame_stride'])]
if len(paths)!=cfg['data']['num_frames']: p.error('image count must equal configured num_frames')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=place_model(make_model(cfg,checkpoint=a.checkpoint),device,cfg)
tfile=Path(cfg['training']['output_dir'])/'temperature.json'
if tfile.is_file(): model.temperature=float(json.loads(tfile.read_text())['temperature'])
model.eval(); images=load_and_preprocess_images(paths,mode=cfg['data']['preprocess_mode']).unsqueeze(0).to(device)
if cfg['data'].get('image_width',518)!=518:
    import torch.nn.functional as F
    width=cfg['data']['image_width']; height=round(images.shape[-2]*width/images.shape[-1]/14)*14
    images=F.interpolate(images[0],size=(height,width),mode='bicubic',align_corners=False).clamp(0,1).unsqueeze(0)
with torch.no_grad(),autocast_context(device,cfg['training']['amp']): out=model(images)
result={'road_prob':{ROAD_CLASSES[k]:v for k,v in enumerate(out['road_prob'][0].tolist())},
        'road_logits':out['road_logits'][0].float().tolist(),'temperature':model.temperature,
        'class_prior':model.class_prior.tolist(),'entropy':out['entropy'][0].item(),
        'normalized_entropy':out['normalized_entropy'][0].item(),
        'entropy_confidence':out['entropy_confidence'][0].item()}
if 'hmm_observation' in out: result['hmm_observation']=out['hmm_observation'][0].tolist()
print(json.dumps(result,indent=2))
