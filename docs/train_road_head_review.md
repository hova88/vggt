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
accumulated gradients by `window_capacity / window_samples` before unscaling. The
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

The trainer supports direct single-device execution and torchrun CUDA DDP.
The separate overfit DDP experiment retains its own behavior. AdamW uses fixed
learning rates, and weight decay still
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

## DDP extension reviewed against the committed baseline

The baseline for this extension is commit `06a9c1e` (training comments, log
interval, and static review). Commit `16735a3` already changed the shared model
builder to allocate VGGT on the current rank's CUDA device and avoid a full
FP32 host allocation per worker. The existing overfit DDP script demonstrates
the repository's model wrapper and rank-0 evaluation approach; its same-sample
memorization objective and stopping rule are not imported into this trainer.

The baseline `train_road_head.py` was single-device code. Running that version
through torchrun would start independent trainers, leave them on the default
GPU, and allow simultaneous writes to the same artifacts. A launch command
alone therefore could not make the committed implementation correct for DDP.
The extension adds actual distributed initialization, training synchronization,
data sharding, collective metrics, and exclusive artifact ownership.

### Launch and device ownership

Run from the repository root on a host with two visible CUDA GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/train_road_head.py \
  --config configs/road_head_nuscenes_small.yaml \
  --pretrained /model/vggt-1b.pt \
  --log-interval 20
```

`distributed_setup` reads world size, global rank, and local rank from the
launcher environment. It sets the CUDA device before process-group operations
or `make_model`. The local device index is relative to `CUDA_VISIBLE_DEVICES`;
global rank is used for data sharding and artifact ownership. Both historical
local-rank CLI spellings are accepted and checked against the environment.
Multi-process runs initialize NCCL through `env://`; direct Python execution
retains single-GPU/CPU support. These launcher conventions match
[PyTorch's torchrun documentation](https://docs.pytorch.org/docs/2.14/elastic/run.html).

Every GPU stores a full model replica. DDP does not allow a model that is too
large for one GPU to fit by spreading its parameters across devices. DDP's
constructor also performs initial parameter synchronization; for this model,
that includes a substantial one-time backbone communication cost. The shared
builder's per-rank allocation improvement remains in use.

### Data partition and collective alignment

All ranks load the same scene-disjoint records. A SHA-256 fingerprint covering
configuration, ordered train/validation records, and `--max-steps` is gathered
before constructing DDP. Disagreement raises on every rank. This catches
inconsistent split order, labels, paths, configuration, and stopping settings;
it cannot verify image bytes on different hosts.

Training uses `DistributedSampler` with the common split seed and
`drop_last=False`. `set_epoch(epoch)` changes the shuffle each epoch. For N
records and W ranks, each rank receives `ceil(N / W)` indices. Total sampling
is W times that count, so up to W - 1 positions repeat records when padding is
necessary. The shared DataLoader disables its own shuffle when a sampler is
provided. These semantics follow the
[DistributedSampler contract](https://docs.pytorch.org/docs/2.14/data.html#torch.utils.data.distributed.DistributedSampler).

Because every rank has the same sampled count and batch size, ranks have equal
batch counts and equal-sized final microbatches. Their accumulation boundaries,
logging collectives, and stopping checks therefore occur in the same order.
A sampler producing uneven batch counts must not replace this sampler without
also redesigning collective alignment and gradient normalization.

With N = 5, W = 2, batch size 2, and accumulation 2, padding yields 6 sampled
positions, 3 per rank. Each rank sees batches of size 2 and 1. There is one
optimizer update using 6 sampled sequences, even though the nominal maximum
batch capacity is 8. Startup logs expose the extra sampled position. Training
loss/accuracy and CSV sample counts describe the sampled stream, including
repeats; validation describes the original unpadded validation records.

### Accumulation and gradient normalization proof

For a window, let C be the nominal capacity on each rank, n its actual sample
count, and g(r, j) the unscaled per-sample gradient. Both C and n are equal across
ranks because of the sampler and identical loader settings. Let S be the shared
FP16 scale, or 1 when scaling is disabled. Before correction, the synchronized
DDP gradient is:

```text
(S / (W * C)) * sum over ranks r and window samples j of g(r, j)
```

Multiplying by `C / n` and unscaling by S gives:

```text
(1 / (W * n)) * sum over ranks r and window samples j of g(r, j)
```

This is the mean gradient over the sampled global window. No extra division by
world size is required: standard DDP already averages gradients. The correction
is now applied before `scaler.unscale_`, so the scaler's finite check also sees
any overflow introduced by that correction. Clipping still follows unscaling.

For all nonfinal microbatches in a window, `no_sync()` encloses both forward and
backward. The final microbatch runs normally, synchronizing the accumulated
gradients. The condition explicitly includes the last loader batch, so an
incomplete final window is synchronized and updated. These choices follow
[PyTorch's DDP no_sync contract and gradient reduction behavior](https://github.com/pytorch/pytorch/blob/v2.3.1/torch/nn/parallel/distributed.py).

The proof concerns sample weighting and gradient reduction. Separate dropout
masks, different samples from sampler padding, and floating-point reduction
order mean it is not a claim of identical results to the original single-GPU
training run. Learning rates remain as configured; increasing world size does
not implicitly multiply learning rates.

### Trainable parameters and model state

The complete `VGGTRoadClassifier` is wrapped in DDP. The optimizer retains raw
parameter references, which remain valid after wrapping. In `head_only`, only
the head contributes trainable parameters, and the classifier's `train()`
override keeps the aggregator in eval mode. In `last_blocks`, both the head and
selected frame/global blocks contribute gradients and are reduced by DDP.

Source inspection traced the selected blocks through the final cached token
features to `road_logits`, and traced the head's trainable projections, queries,
position tensor, attention modules, and classifier to the loss. This supports
`find_unused_parameters=False` for the current architecture. The aggregator
uses `use_reentrant=False` for activation checkpointing, avoiding the additional
DDP restrictions associated with reentrant checkpointing. Future changes that
add conditional or unused trainable branches must revisit these assumptions.

`broadcast_buffers=False` avoids per-forward buffer synchronization. The
current buffers inspected here are fixed image normalization constants and a
class prior derived identically on all ranks; they are not BatchNorm running
statistics. Initial weights use the same seed and DDP synchronizes parameters.
After wrapping, DDP runs use `seed + rank` for stochastic training and loader
worker seeds, while retaining the common sampler/split seed. Single-device
execution does not introduce this additional reseeding step.

### AMP and common stopping decisions

Every rank creates the same standard scaler from the verified AMP settings.
After a synchronized accumulation boundary, ranks have the same reduced
trainable gradients and identical normalization factors. They therefore make
the same finite-gradient/overflow decision under the standard scaler policy.
An FP16 skip increments `skipped_updates` but not `global_step`; a successful
replica update advances one logical global step on every rank.

`--max-steps` is checked only at complete accumulation boundaries. It counts
global updates, not W times the number of replica optimizer calls. All ranks
leave training at the same batch and perform the same final statistics
collectives. Rank 0 then evaluates the entire validation split before the run
exits. FP32/BF16 finite-norm errors occur after the DDP reduction as well.
This reasoning assumes the standard DDP reducer/scaler, identical optimizer
settings, and no custom communication hooks or independently restored states.

### Training metrics and rank-0 validation

Logging boundaries sum local loss totals, correct counts, and sample counts
across ranks. Loss and accuracy use these sums, rather than averaging batch or
rank means. Reduction operates on a new tensor, preserving each rank's local
running totals for subsequent logs. Interval throughput uses total sampled
sequences divided by the largest rank interval duration. Time, batch-wait time,
and allocated memory use rank maxima; memory is not summed across devices.
Epoch training duration similarly uses the largest rank duration.

Validation is deliberately centralized. Rank 0 iterates the complete validation
loader without a distributed sampler, and calls the raw model rather than the
DDP wrapper. There are no validation-time collectives inside that forward while
peers are waiting. Each validation record contributes once. NLL, confusion
matrix, macro F1, Brier, and ECE come from the full prediction set; nonlinear
metrics are never averaged across rank-local subsets.

Only rank 0 selects the best model and owns its selection state. Workers do not
need a copy of best NLL because it does not control subsequent training or early
stopping. The per-epoch phase completion flag releases all ranks after
validation and writes finish. On the next epoch, DDP `train()` restores training
mode on every rank, including rank 0's model that was set to eval for validation.

### Artifact ownership, workers, and failure handling

Only rank 0 creates trainer output artifacts or opens CSV/log files. It saves
the underlying module, preserving original state-dict key names without a
`module.` prefix. The head and updated aggregator parameters retain the prior
inference checkpoint contract. `args.json` and `summary.csv` now record world
size, and the CSV also records global nominal batch size. Checkpoints remain
inference artifacts rather than resumable optimizer snapshots.

The collective order for a normal run is:

1. Process-group initialization and rank-0 setup completion broadcast.
2. Input fingerprint gather and DDP constructor synchronization.
3. Rank-0 CSV-open completion broadcast.
4. Training gradient reductions at update boundaries, statistics reductions
   at progress boundaries, and final training-statistics reductions.
5. Rank-0 full validation/artifact phase completion broadcast.
6. Either the next epoch or coordinated exit after the step/epoch limit.

The rank-0 phase context catches ordinary exceptions and broadcasts an error
flag before raising, so healthy workers leave their wait with an explicit
failure. Hard crashes, worker/data loading failures, CUDA failures, and broken
collectives depend on torchrun's worker supervision and the configured process
group timeout. The script does not promise recovery from these failures.
`finally` destroys the process group without adding an unconditional cleanup
barrier that could wait for a failed rank.

Workers can wait throughout validation and plotting/checkpoint I/O, so the NCCL
collective timeout is configurable through `training.ddp_timeout_seconds`
(default: 3600). It must exceed the full expected rank-0 phase duration.
Centralized validation uses one GPU while peers wait; this is a correctness
and simplicity choice with a clear throughput cost for large validation sets.

When DDP loader workers are enabled, the trainer defaults their context to
`spawn` and permits `forkserver`, rejecting `fork`. The shared loader passes
that context only when workers are nonzero. This follows
[PyTorch's NCCL/DataLoader multiprocessing warning](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).
Dataset classes/collation functions are defined at module scope and the trainer
has a main guard, as required for spawned worker imports. Shared setup helpers
still print from each rank; the trainer logger writes only from rank 0.

### Static-review conclusion and execution limits

The source implements one worker per GPU, aligned sampler/accumulation windows,
standard averaged-gradient DDP updates, full-split validation, and exclusive
rank-0 writes. The reviewed ordering and normalization support the intended
DDP method for the current model and equal-length sampler. This conclusion is
conditional on the inspected contracts, not an execution-based certification.

No code execution was performed for this DDP extension: no torchrun launches,
training, tests, imports, compilation, dependency installation, or performance
measurements. Review consisted of source/diff inspection and a whitespace diff
check. Remaining runtime checks include actual NCCL startup, both finetune
modes with multiple accumulation windows, a padded dataset and short tail,
FP16 skipped updates, checkpoint reload, spawned loader workers, rank-0 error
propagation, and multi-node filesystem/device behavior if deployed that way.
