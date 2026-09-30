"""Train a VGGT road classifier on scene-disjoint NuScenes splits.

The flow follows timm's explicit setup / train_one_epoch / validate structure,
using standard PyTorch and Python logging without a timm dependency.
Each sample contains ordered images [B, T, 3, H, W] and one final-frame target:
class indices [B] or probabilities [B, 5]. Shared data helpers handle mixed
hard/soft labels and prevent scene leakage between training and validation.

Select best.pt by validation NLL at temperature 1.0, before calibration.
Checkpoints store the road head and trainable aggregator parameters, rather
than another copy of the frozen VGGT backbone. These are inference artifacts:
optimizer, scaler, and RNG states are not saved for training recovery.
Keep the original pretrained VGGT checkpoint available for evaluation.

Single GPU: python scripts/train_road_head.py --config CONFIG --pretrained MODEL
Two GPUs: torchrun --standalone --nproc-per-node=2 scripts/train_road_head.py
          --config CONFIG --pretrained MODEL
Batch size and worker count are per rank. Global nominal batch size is
batch_size * accum_steps * WORLD_SIZE. Only rank 0 validates and writes files.
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

# Support direct execution without an editable repository installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.road_common import (  # noqa: E402
    autocast_context, config, data_splits, loader, loss_fn, make_model,
    metrics as classification_metrics, parameter_groups, place_model,
    save_plots, target_prob,
)
from vggt.heads.road_probability_head import ROAD_CLASSES  # noqa: E402


_logger = logging.getLogger("train_road_head")


def parse_args():
    """Parse run overrides; dataset and optimizer settings live in YAML."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/road_head_nuscenes_small.yaml")
    parser.add_argument("--pretrained", help="Local VGGT model.pt; overrides model.checkpoint")
    # torchrun normally supplies LOCAL_RANK through the environment. Accept both
    # historical CLI spellings as well for launchers that pass an explicit rank.
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="Stop after N successful optimizer updates; 0 runs all epochs",
    )
    parser.add_argument(
        "--log-interval", type=int, default=None,
        help="Log every N batches; overrides training.log_interval (default: 20)",
    )
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps must be nonnegative")
    if args.log_interval is not None and args.log_interval < 1:
        parser.error("--log-interval must be at least 1")
    return args


def validate_config(cfg):
    """Reject invalid loop settings before loading the large VGGT backbone."""
    training = cfg["training"]
    for key in ("epochs", "batch_size", "accum_steps", "log_interval", "ddp_timeout_seconds"):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    workers = training["num_workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("training.num_workers must be a nonnegative integer")
    for key in ("head_lr", "backbone_lr", "weight_decay", "grad_clip"):
        value = training[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"training.{key} must be a finite nonnegative number")
    if training["amp"] not in ("none", "fp16", "bf16"):
        raise ValueError("training.amp must be one of: none, fp16, bf16")
    bins = cfg["calibration"]["ece_bins"]
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("calibration.ece_bins must be a positive integer")


def distributed_setup(args, timeout_seconds):
    """Bind each torchrun worker to its GPU before VGGT allocation or NCCL use.

    LOCAL_RANK indexes the GPUs visible to this process, whereas RANK is global
    across nodes. Never use the global rank as a local CUDA device index.
    Direct Python execution retains the original single-GPU/CPU behavior.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank or 0)))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid WORLD_SIZE, RANK, or LOCAL_RANK")
    if args.local_rank is not None and args.local_rank != local_rank:
        raise ValueError("--local-rank disagrees with LOCAL_RANK")
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise ValueError("LOCAL_RANK exceeds the number of visible CUDA devices")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if world_size > 1:
            raise RuntimeError("Multi-process training requires CUDA GPUs and the NCCL backend")
        device = torch.device("cpu")
    if world_size > 1:
        # Nonzero ranks wait while rank 0 validates and saves artifacts. Permit
        # long validation passes via a configurable collective timeout.
        dist.init_process_group(
            backend="nccl", init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
        )
    return rank, world_size, device


@contextmanager
def rank_zero_phase(device):
    """Run an artifact/validation phase on rank 0 and release peers together.

    Every rank must enter this context in the same order. Its caller guards the
    work with the yielded boolean. Broadcast a completion/error flag so workers
    do not begin the next DDP epoch while rank 0 is validating or writing files.
    Ordinary rank-0 exceptions reach peers; process death and collective failures
    remain the responsibility of torchrun and the process-group timeout.
    """
    primary = not dist.is_initialized() or dist.get_rank() == 0
    error = None
    try:
        yield primary
    except Exception as exc:
        error = exc
    if dist.is_initialized():
        status = torch.tensor(int(error is not None), device=device)
        dist.broadcast(status, src=0)
        if status.item() and error is None:
            raise RuntimeError("Rank 0 failed during validation or artifact writing; see its traceback")
    if error is not None:
        raise error


def verify_distributed_inputs(cfg, train, val, max_steps):
    """Check that all ranks agree on settings and ordered labeled records.

    DistributedSampler assumes an identical dataset/order on every rank. A
    compact fingerprint catches divergent manifests/configs before DDP begins;
    it does not compare the actual image bytes stored on different hosts.
    """
    if not dist.is_initialized():
        return
    payload = json.dumps({"config": cfg, "train": train, "val": val, "max_steps": max_steps}, sort_keys=True)
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    fingerprints = [None] * dist.get_world_size()
    dist.all_gather_object(fingerprints, fingerprint)
    if len(set(fingerprints)) != 1:
        raise RuntimeError("Ranks disagree on configuration or ordered train/validation records")


def setup_logging(outdir, rank=0):
    """Send timestamped trainer messages to stderr and an append-only train.log.

    Shared helpers still print their setup diagnostics to stdout. train.log
    captures this trainer's messages; it is not a redirect of all process output.
    """
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    for handler in list(_logger.handlers):
        handler.close()
        _logger.removeHandler(handler)
    if rank != 0:
        _logger.addHandler(logging.NullHandler())
        return
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(outdir / "train.log")):
        handler.setFormatter(formatter)
        _logger.addHandler(handler)


def synchronize(device):
    """Finish queued CUDA work when measuring a logging interval or epoch."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def training_statistics(totals, samples, device):
    """Reduce loss sum, correct predictions, and sampled sequence count globally.

    Reduce a copy so the local running totals are never added twice. Counts
    include any padded duplicates introduced by the training sampler.
    """
    stats = torch.cat((totals, torch.tensor([samples], dtype=totals.dtype, device=device)))
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return stats.tolist()


def train_one_epoch(model, batches, optimizer, scaler, device, training, epoch,
                    global_step=0, max_steps=0):
    """Train one epoch with sample-weighted gradient accumulation.

    Each microbatch loss is a mean. Weight it by batch size using a common
    denominator for the accumulation window, then correct that denominator
    to the actual sample count before the update. This handles a short final
    microbatch and an incomplete window without buffering image sequences.

    global_step counts successful optimizer updates. An FP16 overflow skipped
    by GradScaler does not count. Check the step limit only at update boundaries
    so stopping never discards gradients from an unfinished window.

    DistributedSampler gives every rank the same number of samples/batches,
    including equal-size final microbatches. DDP averages gradients across ranks;
    normalizing by each rank's identical window count therefore gives the mean
    over the global window. This relies on the sampler used by run_training.
    """
    # The classifier's train() override also keeps a frozen aggregator in eval
    # mode during head-only training; use it rather than toggling the head alone.
    model.train()
    optimizer.zero_grad(set_to_none=True)
    trainable = [p for group in optimizer.param_groups for p in group["params"]]
    accumulation, total_batches = training["accum_steps"], len(batches)
    window_samples = samples = interval_samples = skipped_updates = 0
    last_logged = 0
    # Device-side totals avoid loss.item()/accuracy.item() synchronization on
    # every microbatch. Read them only when printing progress and at epoch end.
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    synchronize(device)
    started = interval_started = batch_finished = time.perf_counter()
    data_seconds = 0.0

    for batch_idx, batch in enumerate(batches):
        data_seconds += time.perf_counter() - batch_finished
        images = batch["images"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        batch_size = images.shape[0]
        if window_samples == 0:
            # A fixed denominator keeps accumulated gradients near the scale of
            # a mean loss, avoiding unnecessary FP16 overflow from a summed loss.
            window_batches = min(accumulation, total_batches - batch_idx)
            window_capacity = window_batches * training["batch_size"]
        update_boundary = (batch_idx + 1) % accumulation == 0 or batch_idx + 1 == total_batches
        # no_sync must enclose BOTH the forward and backward pass. Synchronize
        # on the final microbatch of every window, including an incomplete tail.
        sync_context = (
            model.no_sync() if isinstance(model, DistributedDataParallel) and not update_boundary
            else nullcontext()
        )
        with sync_context:
            with autocast_context(device, training["amp"]):
                logits = model(images)["road_logits"]
                loss = loss_fn(logits, target)
            # Weight every sample equally even if microbatch sizes differ.
            scaler.scale(loss * (batch_size / window_capacity)).backward()
        window_samples += batch_size
        samples += batch_size
        interval_samples += batch_size
        with torch.no_grad():
            labels = target.argmax(-1) if target.ndim == 2 else target
            totals[0] += loss.detach().double() * batch_size
            totals[1] += logits.argmax(-1).eq(labels).sum()

        if update_boundary:
            # Correct the sample denominator while gradients are still scaled,
            # then unscale before clipping. This lets GradScaler also detect any
            # overflow introduced by the denominator correction itself.
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.mul_(window_capacity / window_samples)
            scaler.unscale_(optimizer)
            # A zero threshold disables clipping. Still check the gradient norm
            # in FP32/BF16, where no enabled scaler protects the optimizer from
            # nonfinite gradients. In FP16, let GradScaler handle skipped updates.
            torch.nn.utils.clip_grad_norm_(
                trainable, training["grad_clip"] or float("inf"),
                error_if_nonfinite=not scaler.is_enabled(),
            )
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # With the default scaler, overflow lowers the scale. With scaling
            # disabled, both scales are 1 and every optimizer update is counted.
            # DDP has reduced all trainable gradients at this point. Identical
            # initial scaler states and window sizes give the same overflow and
            # step decision on every rank, so --max-steps remains collective.
            if scaler.get_scale() < previous_scale:
                skipped_updates += 1
                _logger.warning("Train: %d | skipped nonfinite FP16 update at batch %d", epoch, batch_idx + 1)
            else:
                global_step += 1
            optimizer.zero_grad(set_to_none=True)
            window_samples = 0

        stop = bool(max_steps and global_step >= max_steps)
        if (batch_idx == 0 or (batch_idx + 1) % training["log_interval"] == 0
                or batch_idx + 1 == total_batches or stop):
            synchronize(device)
            now = time.perf_counter()
            # All ranks enter these collectives at identical batch boundaries,
            # even though only rank 0 writes the resulting progress message.
            loss_sum, correct, global_samples = training_statistics(totals, samples, device)
            if not math.isfinite(loss_sum):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, batch {batch_idx + 1}")
            elapsed = max(now - interval_started, 1e-12)
            interval_batches = batch_idx + 1 - last_logged
            learning_rates = "/".join(f"{group['lr']:.3e}" for group in optimizer.param_groups)
            memory = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
            timing = torch.tensor([elapsed, data_seconds, memory], dtype=torch.float64, device=device)
            interval_count = torch.tensor(interval_samples, dtype=torch.float64, device=device)
            if dist.is_initialized():
                dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                dist.all_reduce(interval_count, op=dist.ReduceOp.SUM)
            elapsed, data_wait, memory = timing.tolist()
            _logger.info(
                "Train: %d [%4d/%d (%3.0f%%)]  Loss: %.4f  Acc@1: %.2f  "
                "Time: %.3fs/batch  Rate: %.1f seq/s  Data: %.3fs/batch  "
                "LR: %s  Mem: %.0f MiB  Updates: %d",
                epoch, batch_idx + 1, total_batches, 100 * (batch_idx + 1) / total_batches,
                loss_sum / global_samples, 100 * correct / global_samples, elapsed / interval_batches,
                interval_count.item() / elapsed, data_wait / interval_batches,
                learning_rates, memory, global_step,
            )
            last_logged = batch_idx + 1
            interval_started, interval_samples, data_seconds = now, 0, 0.0
        batch_finished = time.perf_counter()
        if stop:
            break

    if not samples:
        raise RuntimeError("Training loader produced no samples")
    synchronize(device)
    loss_sum, correct, global_samples = training_statistics(totals, samples, device)
    seconds = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    return {
        "loss": loss_sum / global_samples, "accuracy": correct / global_samples,
        "samples": int(global_samples), "seconds": seconds.item(),
        "skipped_updates": skipped_updates,
    }, global_step


@torch.no_grad()
def validate(model, batches, device, amp, epoch, log_interval, bins):
    """Evaluate all sequences using raw logits at temperature 1.0.

    Soft targets contribute their full distributions to NLL/Brier; accuracy
    uses their argmax class. Cache only small [N, 5] logits/targets on CPU for
    metrics and plotting, never image sequences or VGGT feature tensors.
    """
    model.eval()
    logits_list, targets_list = [], []
    total_loss = correct = samples = 0
    synchronize(device)
    started = time.perf_counter()
    for batch_idx, batch in enumerate(batches):
        images = batch["images"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(images)["road_logits"]
        logits = logits.float().cpu()
        targets = target_prob(batch["target"]).cpu()
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Nonfinite validation logits at batch {batch_idx + 1}")
        logits_list.append(logits)
        targets_list.append(targets)
        total_loss += loss_fn(logits, targets).item() * len(images)
        correct += logits.argmax(-1).eq(targets.argmax(-1)).sum().item()
        samples += len(images)
        if batch_idx == 0 or (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == len(batches):
            elapsed = time.perf_counter() - started
            _logger.info(
                "Test:  %d [%4d/%d]  NLL: %.4f  Acc@1: %.2f  Time: %.3fs/batch  Rate: %.1f seq/s",
                epoch, batch_idx + 1, len(batches), total_loss / samples,
                100 * correct / samples, elapsed / (batch_idx + 1), samples / max(elapsed, 1e-12),
            )
    if not samples:
        raise RuntimeError("Validation loader produced no samples")
    logits, targets = torch.cat(logits_list), torch.cat(targets_list)
    result = classification_metrics(logits, targets, tau=1.0, bins=bins)
    if not all(math.isfinite(result[key]) for key in ("nll", "accuracy", "macro_f1", "brier", "ece")):
        raise FloatingPointError("Nonfinite validation metrics")
    return result, logits, targets


def save_checkpoint(model, cfg, prior, path, epoch, global_step, val_metrics):
    """Atomically replace an inference checkpoint compatible with road_common.

    Filter the aggregator state by trainable parameter names to omit the frozen
    backbone. The current aggregator has no mutable training buffers requiring
    inclusion. Write beside the destination before replacing it, so an interrupted
    write leaves the previously saved checkpoint intact.
    """
    trainable_names = {name for name, p in model.aggregator.named_parameters() if p.requires_grad}
    checkpoint = {
        "road_head": {name: tensor.detach().cpu() for name, tensor in model.road_head.state_dict().items()},
        "aggregator_trainable": {
            name: tensor.detach().cpu() for name, tensor in model.aggregator.state_dict().items()
            if name in trainable_names
        },
        "pretrained_path": cfg["model"].get("checkpoint"),
        "epoch": epoch,
        "global_step": global_step,
        "config": cfg,
        "class_mapping": {str(k): v for k, v in ROAD_CLASSES.items()},
        "class_prior": prior.tolist(),
        "temperature": 1.0,
        "metrics": val_metrics,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def run_training(args, cfg, rank, world_size, device):
    """Train on each rank; validate the full split and write artifacts on rank 0."""
    training = cfg["training"]
    if world_size > 1:
        # NCCL plus forked DataLoader workers is unsafe. Spawn also ensures that
        # workers do not inherit their parent's already-allocated CUDA model.
        training.setdefault("multiprocessing_context", "spawn")
        if training["num_workers"] > 0 and training["multiprocessing_context"] not in ("spawn", "forkserver"):
            raise ValueError("DDP DataLoader workers require multiprocessing_context: spawn or forkserver")
    outdir = Path(training["output_dir"]).expanduser()
    with rank_zero_phase(device) as primary:
        if primary:
            outdir.mkdir(parents=True, exist_ok=True)
        setup_logging(outdir, rank)
        if primary:
            _logger.info("Starting road-head training | config=%s | output=%s | world_size=%d", args.config, outdir, world_size)
            if (outdir / "best.pt").exists():
                _logger.warning("Existing run artifacts will be replaced in %s; use a fresh output_dir to retain them", outdir)
            # Persist settings after applying CLI and worker-context overrides.
            (outdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            run_args = {**vars(args), "world_size": world_size, "backend": "nccl" if world_size > 1 else None}
            (outdir / "args.json").write_text(json.dumps(run_args, indent=2) + "\n")

    seed = training["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Use the same seed for initial weights and scene splitting. Dataset splitting
    # also uses its own Random(seed), so later rank-specific seeds cannot alter it.
    train, val, prior, _ = data_splits(cfg)
    if not train or not val:
        raise RuntimeError("Empty train or validation split")
    verify_distributed_inputs(cfg, train, val, args.max_steps)
    if device.type == "cpu" and training["amp"] != "none":
        _logger.warning("CUDA unavailable; CPU training uses FP32 without autocast")
    train_loader = loader(train, cfg, True)
    train_sampler = None
    if world_size > 1:
        # Pad to ceil(N / world_size) samples per rank rather than discarding
        # records. This equalizes batch/window boundaries, but can repeat up to
        # world_size - 1 training examples. Validation is never padded.
        train_sampler = DistributedSampler(
            train_loader.dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=seed, drop_last=False,
        )
        train_loader = loader(train, cfg, True, sampler=train_sampler)
    val_loader = loader(val, cfg) if rank == 0 else None

    model = place_model(make_model(cfg, prior), device, cfg)
    groups, head_params, block_params = parameter_groups(model, training["head_lr"], training["backbone_lr"])
    optimizer = torch.optim.AdamW(groups, weight_decay=training["weight_decay"])
    # Wrap the complete classifier so both head_only and last_blocks synchronize
    # all trainable parameters. Immutable image normalization/prior buffers need
    # no per-forward broadcast. Current trainable parameters all reach the loss;
    # the aggregator uses non-reentrant gradient checkpointing in last_blocks.
    train_model = (
        DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index,
            broadcast_buffers=False, find_unused_parameters=False,
        ) if world_size > 1 else model
    )
    # DDP has synchronized initial model parameters. Decorrelate dropout and
    # worker augmentations across ranks without changing sampler/split seeds.
    if world_size > 1:
        rank_seed = seed + rank
        random.seed(rank_seed)
        np.random.seed(rank_seed % 2**32)
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed_all(rank_seed)
    # Prefer the unified AMP API; retain compatibility with older PyTorch.
    # BF16 has sufficient exponent range and does not require loss scaling.
    use_scaler = device.type == "cuda" and training["amp"] == "fp16"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    _logger.info(
        "Device: %s | AMP: %s | Train: %d | Val: %d | Batch/rank: %d | Accumulation: %d | Global nominal batch: %d",
        device, training["amp"] if device.type == "cuda" else "none", len(train), len(val),
        training["batch_size"], training["accum_steps"], training["batch_size"] * training["accum_steps"] * world_size,
    )
    if train_sampler is not None:
        padded = train_sampler.total_size - len(train)
        _logger.info(
            "DDP: %d ranks | samples/rank=%d | padded train samples=%d | validation=full split on rank 0",
            world_size, train_sampler.num_samples, padded,
        )
    _logger.info(
        "AdamW: head=%s params, lr=%.3e | backbone=%s params, lr=%.3e | weight_decay=%g | grad_clip=%g",
        f"{sum(p.numel() for p in head_params):,}", training["head_lr"],
        f"{sum(p.numel() for p in block_params):,}", training["backbone_lr"],
        training["weight_decay"], training["grad_clip"],
    )

    best_nll, best_epoch, global_step = float("inf"), 0, 0
    # Start a new CSV alongside the new best-model selection; train.log appends.
    # Enter the CSV context only on rank 0; ExitStack closes it on errors too.
    with ExitStack() as stack:
        summary = None
        with rank_zero_phase(device) as primary:
            if primary:
                summary = stack.enter_context((outdir / "summary.csv").open("w", newline=""))
        writer = None
        for epoch in range(1, training["epochs"] + 1):
            epoch_started = time.perf_counter()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_metrics, global_step = train_one_epoch(
                train_model, train_loader, optimizer, scaler, device, training, epoch, global_step, args.max_steps,
            )
            with rank_zero_phase(device) as primary:
                if primary:
                    # Use the underlying module, never the DDP wrapper: peers
                    # are waiting, so a DDP forward collective could deadlock.
                    # Full-split validation avoids sampler padding, duplicate
                    # predictions, and averaging nonlinear metrics across ranks.
                    val_metrics, val_logits, val_targets = validate(
                        model, val_loader, device, training["amp"], epoch,
                        training["log_interval"], cfg["calibration"]["ece_bins"],
                    )
                    # Root plots describe the latest epoch; best/ describes best.pt.
                    save_plots(val_logits, val_targets, 1.0, outdir, cfg["calibration"]["ece_bins"])
                    if val_metrics["nll"] < best_nll:
                        save_checkpoint(model, cfg, prior, outdir / "best.pt", epoch, global_step, val_metrics)
                        best_nll, best_epoch = val_metrics["nll"], epoch
                        save_plots(val_logits, val_targets, 1.0, outdir / "best", cfg["calibration"]["ece_bins"])
                        _logger.info("Best NLL: %.4f at epoch %d | checkpoint=%s", best_nll, best_epoch, outdir / "best.pt")

                    row = {
                        "epoch": epoch, "global_step": global_step, "world_size": world_size,
                        "global_nominal_batch_size": training["batch_size"] * training["accum_steps"] * world_size,
                        "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
                        "train_samples": train_metrics["samples"], "train_seconds": train_metrics["seconds"],
                        "skipped_updates": train_metrics["skipped_updates"],
                        **{f"val_{key}": val_metrics[key] for key in ("nll", "accuracy", "macro_f1", "brier", "ece")},
                        "head_lr": optimizer.param_groups[0]["lr"],
                        "backbone_lr": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
                        "epoch_seconds": time.perf_counter() - epoch_started,
                    }
                    if writer is None:
                        writer = csv.DictWriter(summary, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    summary.flush()
                    _logger.info(
                        "Epoch: %d | Train loss: %.4f | Train Acc@1: %.2f | Val NLL: %.4f | "
                        "Val Acc@1: %.2f | F1: %.4f | ECE: %.4f | Brier: %.4f | Time: %.1fs",
                        epoch, train_metrics["loss"], 100 * train_metrics["accuracy"], val_metrics["nll"],
                        100 * val_metrics["accuracy"], val_metrics["macro_f1"], val_metrics["ece"],
                        val_metrics["brier"], row["epoch_seconds"],
                    )
            if args.max_steps and global_step >= args.max_steps:
                _logger.info("Reached --max-steps=%d after a complete accumulation window", args.max_steps)
                break
    _logger.info("Training complete | best epoch=%d | best NLL=%.4f | updates=%d", best_epoch, best_nll, global_step)


def main():
    """Initialize the launcher environment and always release the process group."""
    args = parse_args()
    cfg = config(args.config)
    cfg["training"].setdefault("log_interval", 20)
    cfg["training"].setdefault("ddp_timeout_seconds", 3600)
    if args.log_interval is not None:
        cfg["training"]["log_interval"] = args.log_interval
    if args.pretrained:
        cfg["model"]["checkpoint"] = str(Path(args.pretrained).expanduser().resolve(strict=True))
    validate_config(cfg)
    try:
        rank, world_size, device = distributed_setup(args, cfg["training"]["ddp_timeout_seconds"])
        run_training(args, cfg, rank, world_size, device)
    finally:
        # No final barrier: a failed rank must not trap healthy peers in cleanup.
        # torchrun handles failed workers; ordinary rank-0 phase errors are also
        # signaled explicitly before the other workers leave their wait.
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
