# NuScenes road-state training

The classifier predicts five classes from four ordered `CAM_FRONT` frames:
`elevated_up`, `elevated_down`, `main_road`, `side_road`, and `intersection`.
NuScenes does not provide these labels. The annotation file configured at
`data.annotation` must contain one JSON object per target `sample_token`, with
exactly one of `label` (integer 0–4) or `soft_label` (five probabilities).
The label describes the final frame.

The archived Qwen files under `outputs/road_head` are **unverified proposals**.
They contain known false intersection proposals and are not suitable as verified
training or calibration labels without review.

## Scene-split training

Run from the repository root. `--pretrained` points to a local VGGT `model.pt`;
it overrides `model.checkpoint` in the YAML. If neither is provided, the loader
uses the official Hugging Face checkpoint.

```bash
python scripts/train_road_head.py \
  --config configs/road_head_nuscenes_small.yaml \
  --pretrained /model/vggt-1b.pt

python scripts/eval_road_head.py \
  --config configs/road_head_nuscenes_small.yaml \
  --checkpoint outputs/road_head/best.pt

python scripts/calibrate_road_head.py \
  --config configs/road_head_nuscenes_small.yaml \
  --checkpoint outputs/road_head/best.pt
```

Training uses disjoint scenes for train and validation; `best.pt` is selected
by validation NLL. The checkpoint stores the road head, updated aggregator
parameters, and local pretrained path. Evaluation and calibration reload that
path when `model.checkpoint` is unset. Keep the base VGGT file available.

## Two-GPU v1.0-mini memorization experiment

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/train_road_overfit_mini_ddp.py \
  --config configs/road_head_nuscenes_small.yaml \
  --pretrained /model/vggt-1b.pt
```

This experiment forces `v1.0-mini`, pads short scene histories, and uses the
same labeled samples for training and evaluation. Its metrics measure
memorization, not generalization. Defaults are width 518, batch size 4 per GPU,
four trainable final frame/global blocks, bf16, and class weights
`1,1,1,8,2`. It stops early at 100% same-sample accuracy and NLL at most
0.05. The exact experiment settings are saved as
`runs/road_overfit_mini_ddp/config.yaml`; use that config when loading its
`best.pt` later. Increase `--batch-size` only after checking peak GPU memory.

`outputs/road_head` also contains sample-keyframe templates, Qwen proposals,
visual audit notes, and their validation summary. Run
`python scripts/validate_road_proposals.py` to check proposal coverage.
