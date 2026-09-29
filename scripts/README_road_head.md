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
  mistakes are also rendered, covering different target classes and scenes
  after selecting the most confident error; overlapping selections reuse one
  forward pass. Each case also saves raw attention in `.npz` and metadata in
  `.json`.
- `index.html`: links to files, per-class/per-scene tables, mistake pairs,
  the top 20 confident mistakes, and attention images.

Attention is the road head's spatial/temporal query attention averaged across
heads; it is not class-conditioned attribution or VGGT attention rollout.
Heatmaps exclude the optional camera token and align to the exact preprocessed
image grid. Colors show patch weight divided by that frame's mean patch weight;
1 means uniform. All frames in a case share a color scale. Overlay opacity is
proportional to deviation from uniform, so nearly constant weights do not tint
the entire image or create false hotspots. Case JSON and epoch metrics include
normalized attention entropy and maximum relative deviation; the near-uniform
threshold is 5%. Temporal weights show six decimals and a uniform baseline.
Raw, unmodified weights remain available in NPZ files.
Full evaluation predictions supply the case labels/probabilities;
the separate `debug=True` attention pass can have minor numerical differences,
so its probabilities are saved separately in case metadata. Soft-label error
counts and confusion matrices use the target argmax, matching existing metrics.

Metrics and all-error tables are saved every epoch. `--vis-every 5` reduces only
attention rendering to every fifth epoch. Attention cases run one at a time
with gradients disabled; rank 1 waits for rank 0 to complete evaluation/reports.
Reduce `--vis-attention-samples` / `--vis-error-samples` to bound the additional
time and disk usage, or use `--no-vis` to disable the component. Matplotlib is
already included in the road-head dependencies.

Each new run also saves its own configuration snapshot and timestamps in
`run.json`. The overview shows the latest and best epoch, diagnostics, and a
compact epoch table before the figures. Missing target classes show `n/a` in
the metric table. The trainer records optimizer update counts, and curves use
updates as the x-axis when those counts are available. With 378 samples, two
GPUs, and batch size 190 per GPU, an epoch contains just one update; 30 epochs
then provide only 30 updates. A smaller batch or more epochs is needed to
compare against experiments that performed hundreds of updates.

A report directory belongs to its original run. Starting another training run
with `--no-vis` does not update that report. The shared parent `config.yaml`
may be overwritten by another run, so it is not a reliable historical snapshot.

### Interactive model internals (EL-VIT style)

`scripts/road_model_inspector.py` observes real tensors during one eval forward;
`scripts/road_model_inspector.html` is its self-contained browser viewer. It
shows patch input/kernel/output, actual MLP GELU input/output, DINO positional
embedding contributions, selected feature channels, effective Q/K, individual
attention heads, and patch feature cosine similarity. Select a frame, module,
head, or captured query patch; clicking a marked input patch updates the views.
Global attention can target another frame. Road spatial attention correctly
uses an ego query rather than a patch-to-patch attention matrix.

Enable it in the existing per-epoch component:

```bash
# Add to the normal training command:
--vis-every 5 --vis-model-internals
```

On attention epochs, the first fixed probe gets a `cases/*_internals.html`
link in the epoch report. It reuses that probe's existing debug forward; no
extra model is loaded and training forwards have no inspection hooks. By
default this captures the final DINO block, final VGGT frame/global blocks,
and road spatial attention, heads 0/1, 32 display channels, and at most 64
query patches. Cosine similarity is computed from the full feature dimension.
Attention entries are reconstructed in FP32 from effective Q/K after actual
normalization and RoPE. The sampled matrix is not renormalized; global
softmax includes every frame and special token. These views show associations
and information allocation, not class-conditioned causal attribution.

Generate a viewer from an existing trained checkpoint without restarting
training:

```bash
python scripts/road_model_inspector.py \
  --checkpoint runs/road_overfit_mini_ddp/best.pt \
  --pretrained /model/vggt-1b.pt \
  --predictions runs/road_overfit_mini_ddp/visualizations/RUN/epoch_0005/predictions.jsonl \
  --output-dir runs/model_inspection
```

Replace `RUN` with the report directory. `--predictions` preserves saved
targets and image paths when live annotations have since changed; those
historical prediction scores are not reused as new inference results.
Without it, targets come from the live annotation file recorded in the
checkpoint config. The viewer identifies checkpoint epoch, optimizer steps,
and target source. The checkpoint's config snapshot controls preprocessing
and model architecture; `--config` is only a fallback for older checkpoints
without that snapshot. The default sample has distinct historical frames
when available. Repeat `--sample-token TOKEN` for specific samples; use
`--blocks 11,23 --heads 0,1,4,7` to capture more layers/heads (higher cost).

Open the printed `index.html`, or serve its output directory through the
existing SSH tunnel. Use a recent Chrome/Edge browser supporting native gzip
decompression; no CDN or additional frontend dependencies are needed. Arrays
use lossless BF16 packing when exactly representable, otherwise FP32, and
gzip compression. A four-frame 518-wide default snapshot is approximately
16 MB for the verified sample; allow time and disk space per saved probe.
Existing reports cannot recover internal tensors that were never captured;
checkpoint inference produces a separately identified snapshot.

### Refresh an existing report without retraining

```bash
python scripts/road_training_visualizer.py \
  --refresh runs/road_overfit_mini_ddp/visualizations/run_20260929_154417_466111 \
  --config runs/road_overfit_mini_ddp/config.yaml
```

This writes a separate `reviewed/index.html` using saved predictions and raw
attention. It runs on CPU without loading VGGT or changing the original report,
weights, or training process. For old reports without a snapshot, `--config`
supplies image preprocessing settings only; the report explicitly marks the
original training configuration as unavailable. Reconstructed image shapes
must match the saved shapes. Historical attention is not recomputed from a
different model checkpoint. Only saved attention cases are available; diverse
case selection applies to future training runs.

`outputs/road_head` also contains sample-keyframe templates, Qwen proposals,
visual audit notes, and their validation summary. Run
`python scripts/validate_road_proposals.py` to check proposal coverage.
