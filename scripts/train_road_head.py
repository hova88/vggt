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
"""

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

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
    for key in ("epochs", "batch_size", "accum_steps", "log_interval"):
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


def setup_logging(outdir):
    """Send timestamped trainer messages to stderr and an append-only train.log.

    Shared helpers still print their setup diagnostics to stdout. train.log
    captures this trainer's messages; it is not a redirect of all process output.
    """
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    for handler in list(_logger.handlers):
        handler.close()
        _logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(outdir / "train.log")):
        handler.setFormatter(formatter)
        _logger.addHandler(handler)


def synchronize(device):
    """Finish queued CUDA work when measuring a logging interval or epoch."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
        with autocast_context(device, training["amp"]):
            logits = model(images)["road_logits"]
            loss = loss_fn(logits, target)
        # Weight every sample equally even if microbatch sizes differ. Gradients
        # match a concatenated batch up to rounding and stochastic layers.
        scaler.scale(loss * (batch_size / window_capacity)).backward()
        window_samples += batch_size
        samples += batch_size
        interval_samples += batch_size
        with torch.no_grad():
            labels = target.argmax(-1) if target.ndim == 2 else target
            totals[0] += loss.detach().double() * batch_size
            totals[1] += logits.argmax(-1).eq(labels).sum()

        update_boundary = (batch_idx + 1) % accumulation == 0 or batch_idx + 1 == total_batches
        if update_boundary:
            # Unscale before normalization/clipping. GradScaler records any
            # nonfinite gradients here and skips the corresponding FP16 update.
            scaler.unscale_(optimizer)
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.mul_(window_capacity / window_samples)
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
            loss_sum, correct = totals.tolist()
            if not math.isfinite(loss_sum):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, batch {batch_idx + 1}")
            elapsed = max(now - interval_started, 1e-12)
            interval_batches = batch_idx + 1 - last_logged
            learning_rates = "/".join(f"{group['lr']:.3e}" for group in optimizer.param_groups)
            memory = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
            _logger.info(
                "Train: %d [%4d/%d (%3.0f%%)]  Loss: %.4f  Acc@1: %.2f  "
                "Time: %.3fs/batch  Rate: %.1f seq/s  Data: %.3fs/batch  "
                "LR: %s  Mem: %.0f MiB  Updates: %d",
                epoch, batch_idx + 1, total_batches, 100 * (batch_idx + 1) / total_batches,
                loss_sum / samples, 100 * correct / samples, elapsed / interval_batches,
                interval_samples / elapsed, data_seconds / interval_batches,
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
    loss_sum, correct = totals.tolist()
    return {
        "loss": loss_sum / samples, "accuracy": correct / samples,
        "samples": samples, "seconds": time.perf_counter() - started,
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


def main():
    """Set up a single-device run, then train/validate and select by NLL."""
    args = parse_args()
    cfg = config(args.config)
    training = cfg["training"]
    training.setdefault("log_interval", 20)
    if args.log_interval is not None:
        training["log_interval"] = args.log_interval
    if args.pretrained:
        cfg["model"]["checkpoint"] = str(Path(args.pretrained).expanduser().resolve(strict=True))
    validate_config(cfg)

    outdir = Path(training["output_dir"]).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir)
    _logger.info("Starting road-head training | config=%s | output=%s", args.config, outdir)
    if (outdir / "best.pt").exists():
        _logger.warning("Existing run artifacts will be replaced in %s; use a fresh output_dir to retain them", outdir)
    # Persist effective configuration, including overrides, for later inference.
    # The step limit belongs to args.json rather than shared model/data settings.
    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (outdir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    seed = training["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # DataLoader seeds worker RNGs from torch's generator. These seeds control
    # initialization/shuffling but do not guarantee deterministic CUDA kernels.
    train, val, prior, _ = data_splits(cfg)
    if not train or not val:
        raise RuntimeError("Empty train or validation split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and training["amp"] != "none":
        _logger.warning("CUDA unavailable; CPU training uses FP32 without autocast")
    train_loader, val_loader = loader(train, cfg, True), loader(val, cfg)

    model = place_model(make_model(cfg, prior), device, cfg)
    groups, head_params, block_params = parameter_groups(model, training["head_lr"], training["backbone_lr"])
    optimizer = torch.optim.AdamW(groups, weight_decay=training["weight_decay"])
    # Prefer the unified AMP API; retain compatibility with older PyTorch.
    # BF16 has sufficient exponent range and does not require loss scaling.
    use_scaler = device.type == "cuda" and training["amp"] == "fp16"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    _logger.info(
        "Device: %s | AMP: %s | Train: %d | Val: %d | Batch: %d | Accumulation: %d | Effective batch: %d",
        device, training["amp"] if device.type == "cuda" else "none", len(train), len(val),
        training["batch_size"], training["accum_steps"], training["batch_size"] * training["accum_steps"],
    )
    _logger.info(
        "AdamW: head=%s params, lr=%.3e | backbone=%s params, lr=%.3e | weight_decay=%g | grad_clip=%g",
        f"{sum(p.numel() for p in head_params):,}", training["head_lr"],
        f"{sum(p.numel() for p in block_params):,}", training["backbone_lr"],
        training["weight_decay"], training["grad_clip"],
    )

    best_nll, best_epoch, global_step = float("inf"), 0, 0
    # Start a new CSV alongside the new best-model selection; train.log appends.
    with (outdir / "summary.csv").open("w", newline="") as summary:
        writer = None
        for epoch in range(1, training["epochs"] + 1):
            epoch_started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_metrics, global_step = train_one_epoch(
                model, train_loader, optimizer, scaler, device, training, epoch, global_step, args.max_steps,
            )
            val_metrics, val_logits, val_targets = validate(
                model, val_loader, device, training["amp"], epoch,
                training["log_interval"], cfg["calibration"]["ece_bins"],
            )
            # Root-level plots describe the latest epoch; best/ plots describe
            # the epoch selected for best.pt. This avoids comparing mismatched runs.
            save_plots(val_logits, val_targets, 1.0, outdir, cfg["calibration"]["ece_bins"])
            if val_metrics["nll"] < best_nll:
                save_checkpoint(model, cfg, prior, outdir / "best.pt", epoch, global_step, val_metrics)
                best_nll, best_epoch = val_metrics["nll"], epoch
                save_plots(val_logits, val_targets, 1.0, outdir / "best", cfg["calibration"]["ece_bins"])
                _logger.info("Best NLL: %.4f at epoch %d | checkpoint=%s", best_nll, best_epoch, outdir / "best.pt")

            row = {
                "epoch": epoch, "global_step": global_step,
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


if __name__ == "__main__":
    main()
