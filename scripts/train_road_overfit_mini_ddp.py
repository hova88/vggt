"""Memorization experiment on all labeled nuScenes v1.0-mini samples.

Run from the project root:
  torchrun --standalone --nproc-per-node=2 scripts/train_road_overfit_mini_ddp.py

Train and evaluation use the same union of the project's train/val samples.
The resulting accuracy is a memorization diagnostic, not generalization.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.road_common import (  # noqa: E402
    autocast_context,
    collect,
    config,
    data_splits,
    loader,
    loss_fn,
    make_model,
    metrics as classification_metrics,
    parameter_groups,
    place_model,
)
from scripts.road_training_visualizer import RoadTrainingVisualizer  # noqa: E402
from training.data.nuscenes_road_dataset import class_counts  # noqa: E402
from vggt.heads.road_probability_head import ROAD_CLASSES  # noqa: E402


def distributed_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA GPUs")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        torch.cuda.set_device(0)
    return world_size, local_rank, torch.device("cuda", local_rank)


def supervised_loss(logits, target, class_weights):
    if class_weights is None:
        return loss_fn(logits, target)
    if target.ndim == 1:
        probs = F.one_hot(target.long(), num_classes=logits.shape[-1]).float()
    else:
        probs = target.float()
    return -(probs * F.log_softmax(logits.float(), dim=-1) * class_weights).sum(-1).mean()


def save_best(model, cfg, prior, path, epoch, step, nll, pretrained_path):
    trainable_names = {name for name, parameter in model.aggregator.named_parameters()
                       if parameter.requires_grad}
    checkpoint = {
        "road_head": {name: tensor.detach().cpu() for name, tensor in
                      model.road_head.state_dict().items()},
        "aggregator_trainable": {name: tensor.detach().cpu() for name, tensor in
                                 model.aggregator.state_dict().items()
                                 if name in trainable_names},
        "pretrained_path": str(pretrained_path),
        "epoch": epoch,
        "global_step": step,
        "nll": nll,
        "temperature": 1.0,
        "config": cfg,
        "class_prior": prior.tolist(),
        "class_mapping": {str(key): value for key, value in ROAD_CLASSES.items()},
        "experiment": "overfit_v1.0-mini_train_equals_eval",
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/road_head_nuscenes_small.yaml")
    parser.add_argument("--pretrained", default="/model/vggt-1b.pt")
    parser.add_argument("--output-dir", default="runs/road_overfit_mini_ddp")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size per GPU; global batch = batch-size * GPU count")
    parser.add_argument("--workers", type=int, default=2, help="Training DataLoader workers per GPU")
    parser.add_argument("--eval-workers", type=int, default=0, help="Evaluation workers on rank 0")
    parser.add_argument("--prefetch-factor", type=int, default=1, help="Prefetched batches per worker")
    parser.add_argument("--image-width", type=int, default=518)
    parser.add_argument("--last-blocks", type=int, default=4,
                        help="Unfreeze this many final frame/global aggregator blocks")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--amp", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--class-weights", default="1,1,1,8,2",
                        help="Five positive CE weights; use 1,1,1,1,1 for original loss")
    parser.add_argument("--target-nll", type=float, default=0.05)
    parser.add_argument("--no-vis", action="store_true", help="Disable the per-epoch visualization component")
    parser.add_argument("--vis-every", type=int, default=1,
                        help="Render attention every N epochs; metrics and error analysis remain per-epoch")
    parser.add_argument("--vis-model-internals", action="store_true",
                        help="Capture interactive VGGT internals for one fixed probe on attention epochs")
    parser.add_argument("--vis-attention-samples", type=int, default=3,
                        help="Fixed attention probes, selected across classes")
    parser.add_argument("--vis-error-samples", type=int, default=3,
                        help="Most confident mistakes to visualize per attention epoch")
    args = parser.parse_args()
    if (args.batch_size < 1 or args.workers < 0 or args.eval_workers < 0
            or args.prefetch_factor < 1 or args.epochs < 1 or args.last_blocks < 0):
        parser.error("Invalid batch size, worker count, prefetch factor, epoch count, or last-block count")
    if args.vis_every < 1 or min(args.vis_attention_samples, args.vis_error_samples) < 0:
        parser.error("Invalid visualization interval or sample count")
    weights = [float(value) for value in args.class_weights.split(",")]
    if len(weights) != 5 or any(not np.isfinite(value) or value <= 0 for value in weights):
        parser.error("--class-weights needs five positive finite numbers")
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("This GPU does not support bf16; use --amp fp16")
    pretrained_path = Path(args.pretrained).expanduser().resolve(strict=True)

    world_size, local_rank, device = distributed_setup()
    rank = dist.get_rank() if world_size > 1 else 0
    try:
        cfg = config(args.config)
        cfg["data"]["version"] = "v1.0-mini"
        cfg["data"]["pad_history"] = True
        cfg["data"]["image_width"] = args.image_width
        cfg["model"]["checkpoint"] = str(pretrained_path)
        cfg["model"]["road_head"]["dropout"] = 0.0
        cfg["training"].update({
            "finetune_mode": "last_blocks" if args.last_blocks else "head_only",
            "epochs": args.epochs,
            "last_blocks": args.last_blocks,
            "batch_size": args.batch_size,
            "num_workers": args.workers,
            "prefetch_factor": args.prefetch_factor,
            "head_lr": args.head_lr,
            "backbone_lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
            "amp": args.amp,
            "accum_steps": 1,
            "output_dir": args.output_dir,
        })
        cfg["visualization"] = {"enabled": not args.no_vis, "attention_every": args.vis_every,
                                "model_internals": args.vis_model_internals,
                                "attention_samples": args.vis_attention_samples,
                                "error_samples": args.vis_error_samples}
        seed = cfg["training"]["seed"]
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        torch.backends.cudnn.benchmark = True

        train, val, prior, _stats = data_splits(cfg)
        records = list(train) + list(val)
        if not records:
            raise RuntimeError("No labeled v1.0-mini samples")
        tokens = [record["sample_token"] for record in records]
        if len(tokens) != len(set(tokens)):
            raise RuntimeError("Duplicate sample_token after combining train and validation")
        counts = class_counts(records)
        prior = (counts + 1e-6) / (counts.sum() + 5e-6)
        base = loader(records, cfg, True)
        sampler = DistributedSampler(base.dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=False) if world_size > 1 else None
        train_loader = loader(records, cfg, True, sampler=sampler)
        eval_loader = loader(records, cfg, num_workers=args.eval_workers) if rank == 0 else None

        model = make_model(cfg, prior)
        model = place_model(model, device, cfg)
        groups, head_params, block_params = parameter_groups(
            model, args.head_lr, args.backbone_lr)
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp == "fp16")
        parameters = head_params + block_params
        weighted = None if all(value == 1 for value in weights) else torch.tensor(weights, device=device)

        train_model = (DistributedDataParallel(model, device_ids=[local_rank],
                                               output_device=local_rank,
                                               broadcast_buffers=False,
                                               find_unused_parameters=False)
                       if world_size > 1 else model)
        output_dir = Path(args.output_dir)
        visualizer = None
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "config.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            print(f"samples={len(base.dataset)} global_batch={args.batch_size * world_size} "
                  f"image_width={args.image_width} last_blocks={args.last_blocks} "
                  f"class_counts={counts.tolist()}", flush=True)
            print(f"trainable head={sum(p.numel() for p in head_params):,}, "
                  f"aggregator={sum(p.numel() for p in block_params):,}", flush=True)
            print("Train and evaluation use the SAME samples; metrics measure memorization.",
                  flush=True)
            print(f"Optimizer updates per epoch={len(train_loader)}; "
                  f"planned updates={len(train_loader) * args.epochs}", flush=True)
            if not args.no_vis:
                visualizer = RoadTrainingVisualizer(
                    output_dir, records, ROAD_CLASSES, every=args.vis_every,
                    attention_samples=args.vis_attention_samples, error_samples=args.vis_error_samples,
                    bins=cfg["calibration"]["ece_bins"], target_nll=args.target_nll, run_config=cfg,
                    model_internals=args.vis_model_internals)
                print(f"Visualizations: {visualizer.output_dir / 'index.html'}", flush=True)
        if world_size > 1:
            dist.barrier()

        best_nll = float("inf")
        global_step = 0
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(1, args.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            train_model.train()
            stats = torch.zeros(3, device=device, dtype=torch.float64)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for batch in train_loader:
                images = batch["images"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, args.amp):
                    logits = train_model(images)["road_logits"]
                    loss = supervised_loss(logits, target, weighted)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, cfg["training"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                labels = target.argmax(-1) if target.ndim == 2 else target
                stats[0] += loss.detach() * len(images)
                stats[1] += (logits.detach().argmax(-1) == labels).sum()
                stats[2] += len(images)
            if world_size > 1:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started

            if rank == 0:
                logits, targets, samples = collect(model, eval_loader, device, args.amp,
                                                   debug_first=epoch == 1)
                metrics = classification_metrics(logits, targets, bins=cfg["calibration"]["ece_bins"])
                (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                memory_gib = torch.cuda.max_memory_allocated(device) / 2**30
                print(f"epoch={epoch} train_loss={stats[0] / stats[2]:.4f} "
                      f"train_acc={stats[1] / stats[2]:.4f} "
                      f"same_sample_acc={metrics['accuracy']:.4f} "
                      f"same_sample_nll={metrics['nll']:.4f} "
                      f"samples_per_sec={stats[2] / seconds:.1f} "
                      f"gpu0_peak_GiB={memory_gib:.2f} "
                      f"confusion={metrics['confusion_matrix']}", flush=True)
                if visualizer is not None:
                    vis_started = time.perf_counter()
                    summary = visualizer.on_epoch_end(
                        epoch, logits, targets, samples, metrics,
                        {"train_loss": (stats[0] / stats[2]).item(),
                         "train_acc": (stats[1] / stats[2]).item(),
                         "samples_per_sec": (stats[2] / seconds).item(),
                         "optimizer_steps": global_step,
                         "optimizer_steps_per_epoch": len(train_loader),
                         "gpu0_cumulative_peak_GiB": memory_gib},
                        model, eval_loader.dataset, device, args.amp)
                    print(f"vis epoch={epoch} errors={summary['error_count']} "
                          f"high_confidence_errors={summary['high_confidence_errors']} "
                          f"present_class_f1={summary['macro_f1_present']:.4f} "
                          f"seconds={time.perf_counter() - vis_started:.1f}", flush=True)
                if metrics["nll"] < best_nll:
                    best_nll = metrics["nll"]
                    save_best(model, cfg, prior, output_dir / "best.pt", epoch,
                              global_step, best_nll, pretrained_path)
                stop = metrics["accuracy"] >= 1.0 - 1e-6 and metrics["nll"] <= args.target_nll
            else:
                stop = False
            if world_size > 1:
                signal = torch.tensor(int(stop), device=device)
                dist.broadcast(signal, src=0)
                stop = bool(signal.item())
            if stop:
                if rank == 0:
                    print("Memorization target reached; stopping early.", flush=True)
                break
        if visualizer is not None:
            visualizer.finish('memorization_target' if stop else 'epoch_limit')
        print(f"rank={rank} peak_gpu_memory_GiB="
              f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f}", flush=True)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
