# Static review of the road-head trainer

Scope: `scripts/train_road_head.py`, its configuration, and the shared helpers
used for loading, precision, loss, validation metrics, and checkpoint reload.
The implementation borrows the explicit function structure and progress-log
conventions of [timm's training script](https://github.com/huggingface/pytorch-image-models/blob/main/train.py),
while retaining the repository's road-classification model and data pipeline.

This review is based on source inspection only. At the user's request, no
training, tests, import checks, compilation, dependency installation, or runtime
benchmarks were performed. The log sample below is illustrative, not captured
from execution. Runtime correctness and performance remain unverified.

## Correctness findings and changes

### 1. Unequal microbatches were weighted equally during accumulation

Priority: medium; fixed in `train_one_epoch`.

Previously, the trainer divided each mean microbatch loss by the number of
microbatches in the accumulation window. That handles an incomplete final
window when microbatch sizes are equal, but a short final batch receives the
same weight as a full batch. For batch sizes 4 and 1, the original gradient is
`(mean_gradient_4 + mean_gradient_1) / 2`; the intended sample mean is
`(4 * mean_gradient_4 + mean_gradient_1) / 5`.

The revised loop weights loss by `batch_size / window_capacity`, then multiplies
unscaled accumulated gradients by `window_capacity / window_samples`. The
product gives each sample weight `1 / window_samples`. The common denominator
keeps the accumulated loss near the scale of a mean loss for FP16 scaling.
The final window uses its actual number of samples without retaining several
batches in memory. Stochastic dropout and numerical rounding still mean that
accumulated microbatches need not exactly reproduce one concatenated forward.

### 2. FP16 overflow counted as an optimizer update

Priority: medium; fixed in `train_one_epoch`.

`GradScaler.step()` can skip AdamW when gradients contain nonfinite values. The
previous loop incremented `global_step` regardless, so `--max-steps` could stop
before the requested number of real updates. The revised loop compares the
scale before and after `update()`: a decrease under the standard scaler means
the update was skipped. Skips are logged and counted separately. Step limits
are checked after complete accumulation windows, followed by full validation.
This detection assumes the standard GradScaler policy created by this script.

### 3. Probability clamping capped validation NLL

Priority: medium; fixed in `scripts/road_common.py::metrics`.

The previous formula took `log(max(probability, 1e-12))`. It capped a hard target's
penalty at approximately 27.63 even if a confidently wrong prediction deserved
much higher NLL. This matters because NLL selects `best.pt` and is also used for
calibration reporting. For target class 1 and logits `[100, 0, 0, 0, 0]`, the
correct cross entropy is approximately 100, rather than 27.63.

The shared helper now computes NLL with `log_softmax` directly. Probability
metrics use its exponentiated values. Training loss, validation progress, plot
metrics, and checkpoint selection therefore use the same stable NLL definition.
Other consumers of the shared helper also receive this numerical correction.
Historical NLL values can differ for extremely confident wrong predictions.

### 4. FP32/BF16 gradients had no nonfinite-update protection

Priority: medium; fixed in `train_one_epoch`.

Loss scaling is enabled only for CUDA FP16. Previously, an invalid gradient in
FP32/BF16 could reach AdamW even with clipping enabled. The revised gradient
norm call uses `error_if_nonfinite=True` when the scaler is disabled. A zero
clip threshold now means no clipping, with an infinite maximum norm retaining
the finite-norm check. For FP16, the enabled scaler handles skipped updates.
Training loss totals and validation logits/metrics are also checked for finite
values, preventing invalid metrics from silently passing checkpoint selection.

### 5. Root-level plots could describe a different epoch from best.pt

Priority: low; fixed in the epoch orchestration.

The original plots were overwritten every epoch while the checkpoint was only
replaced on an improvement. The revised trainer preserves latest-epoch plots
at the original paths and saves selected-epoch plots under `best/`. The selected
metrics are also embedded in `best.pt`. A checkpoint is written to `best.pt.tmp`
and renamed only after serialization finishes, protecting the old checkpoint
from a partially written replacement. The plots and checkpoint are separate
writes; interruption can still leave plots older than the checkpoint. The
metrics embedded in `best.pt` identify the selected result.

## Structure, comments, and logs

All added code comments, docstrings, CLI help, and trainer log messages are in
English. The module explains input/target shapes, final-frame supervision, the
selection metric, and the checkpoint's dependency on the pretrained backbone.
Functions separate argument parsing, early setting validation, logging setup,
training, validation, checkpoint serialization, and run orchestration.

Progress output uses `Train:` / `Test:` messages with batch position, loss/NLL,
`Acc@1`, elapsed time, and sequence throughput. Training also reports separate
head/backbone learning rates, peak allocated GPU memory, batch-wait time, and
successful updates. Loss and accuracy are sample-weighted averages since epoch
start. Training timing/rate are interval averages; validation timing/rate are
averages since validation start. GPU memory is peak allocated tensor memory
since epoch reset, not total device usage or reserved allocator memory.

An illustrative message format is:

```text
2026-09-30 12:00:00 | INFO | Train: 1 [  20/100 ( 20%)]  Loss: 1.2345  Acc@1: 55.00  Time: 0.400s/batch  Rate: 2.5 seq/s  Data: 0.005s/batch  LR: 1.000e-04  Mem: 8000 MiB  Updates: 20
```

`training.log_interval` defaults to 20 and can be overridden with
`--log-interval`. The first batch, final batch, and batch reaching the step
limit are logged. CSV rows store accuracy as a fraction; console `Acc@1` is a
percentage. `train.log` appends, while each run writes a new `summary.csv`,
`config.yaml`, and `args.json`. Shared setup helper prints still go to stdout
rather than through the trainer logger.

Training totals remain on the device until a progress boundary, removing the
original per-microbatch loss/accuracy scalar reads. CUDA synchronization at
logging boundaries makes timing include queued GPU work. FP16 scaler operations
can still synchronize at optimizer boundaries, and validation copies each
batch's small logits to CPU. No measured speedup is asserted.

## Compatibility checked by source inspection

- The existing scene split and class prior come from `data_splits`; no split
  or label semantics were changed.
- Hard labels and soft distributions still use `loss_fn`. Validation converts
  hard labels to one-hot distributions using the existing `target_prob` helper.
- `model.train()` preserves the classifier's frozen-aggregator eval policy in
  `head_only` mode. Validation uses `model.eval()` and disabled autograd.
- Existing head/backbone parameter groups and their learning rates are retained.
  Trainable parameters are checked for omissions by the existing group helper.
- The unified GradScaler API is preferred with a legacy CUDA API fallback.
  CPU uses FP32 through the existing autocast helper; BF16 does not enable scaling.
- Required checkpoint fields remain compatible with `road_common.make_model`.
  The additional `metrics` field does not change its loading contract.
- Only trainable aggregator parameters are serialized. Source inspection found
  fixed normalization buffers, rather than mutable BatchNorm running statistics,
  in the current aggregator. If future architecture changes introduce mutable
  buffers, the checkpoint filter must be revisited.
- Existing YAML configurations without `log_interval` retain a default value.
  Positive loop counts, valid AMP mode, nonnegative finite optimizer settings,
  and positive ECE bin counts are checked before model loading.

## Deliberate limits and future runtime checks

This is a single-device trainer. The separate overfit DDP experiment retains
its own behavior. AdamW uses fixed learning rates, and weight decay still
applies to every parameter in each group, including biases and normalization
parameters. Introducing a scheduler or decay exclusions would change the
optimization recipe and should be evaluated with an actual experiment.

Checkpoints support evaluation/calibration, not exact training resume. They do
not include optimizer moments, scaler state, RNG state, or loader position.
The saved seed controls initialization and shuffling without promising full
CUDA determinism. Use a separate output directory for each run: existing
configuration, CSV, plots, and selected checkpoint can be replaced. The trainer
warns when an existing `best.pt` is present.

Small validation sets remain unsuitable for firm calibration conclusions; the
shared split helper retains its existing warning. Accuracy on soft labels uses
the highest-probability class; NLL and Brier use the full target distribution.
Macro F1 retains the shared five-class definition, including absent classes.

When runtime execution is permitted in the training environment, useful checks
are: compare accumulated and full-batch gradients with dropout disabled for
microbatch sizes 4 and 1; exercise a short final accumulation window; verify
that a skipped FP16 update does not advance the step count; reload a selected
checkpoint in both finetune modes; and inspect logged throughput and artifact
consistency. None of these checks was executed during this edit.
