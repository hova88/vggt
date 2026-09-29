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

### Per-epoch visualization component

`scripts/road_training_visualizer.py` contains `RoadTrainingVisualizer`.
The DDP trainer calls `on_epoch_end(...)` on rank 0 after its full evaluation;
the component does not add visualization work to the training forward pass.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/train_road_overfit_mini_ddp.py \
  --config configs/road_head_nuscenes_small.yaml \
  --pretrained /model/vggt-1b.pt \
  --batch-size 8 --workers 2 --eval-workers 0 --prefetch-factor 1 \
  --vis-every 1 --vis-attention-samples 3 --vis-error-samples 3
```

Visualization is enabled by default. Each run gets a separate timestamped
directory under `runs/road_overfit_mini_ddp/visualizations/`; the console prints
its `index.html` path. Open that file to browse metric trends and epoch reports.
Each `epoch_XXXX/` contains:

- `curves.png`: training accuracy/loss, same-sample accuracy/NLL, all-class and
  present-class F1, balanced accuracy, recall, ECE, and Brier score.
- `confusion_matrix.png` and `reliability.png`: count/row-normalized confusion,
  reliability bins, and confidence distributions. Classes without samples are
  marked `n/a` in confusion matrices; present-class metrics exclude them.
- `metrics.json`, `predictions.jsonl`, and `errors.csv`: per-class/per-scene
  metrics, every prediction with probabilities and original frame paths, and
  mistakes sorted by predicted confidence. Error counts and consecutive error
  streaks track repeatedly difficult examples.
- `cases/*.png`: original model-input frames, spatial attention overlays,
  temporal query weights, and class probabilities. Fixed probes are selected
  across represented classes and stay identical across epochs. Up to three
  most confident mistakes are also rendered; overlapping selections reuse one
  forward pass. Each case also saves raw attention in `.npz` and metadata in
  `.json`.
- `index.html`: links to files, per-class/per-scene tables, mistake pairs,
  the top 20 confident mistakes, and attention images.

Attention is the road head's spatial/temporal query attention averaged across
heads; it is not class-conditioned attribution or VGGT attention rollout.
Heatmaps exclude the optional camera token and align to the exact preprocessed
image grid. All frames in one case share a raw-weight color scale, shown on the
colorbar. Full evaluation predictions supply the case labels/probabilities;
the separate `debug=True` attention pass can have minor numerical differences,
so its probabilities are saved separately in case metadata. Soft-label error
counts and confusion matrices use the target argmax, matching existing metrics.

Metrics and all-error tables are saved every epoch. `--vis-every 5` reduces only
attention rendering to every fifth epoch. Attention cases run one at a time
with gradients disabled; rank 1 waits for rank 0 to complete evaluation/reports.
Reduce `--vis-attention-samples` / `--vis-error-samples` to bound the additional
time and disk usage, or use `--no-vis` to disable the component. Matplotlib is
already included in the road-head dependencies.

`outputs/road_head` also contains sample-keyframe templates, Qwen proposals,
visual audit notes, and their validation summary. Run
`python scripts/validate_road_proposals.py` to check proposal coverage.
