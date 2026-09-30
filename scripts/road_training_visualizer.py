"""Per-epoch reports, road-head attention overlays, and error analysis.

The trainer calls RoadTrainingVisualizer.on_epoch_end on rank 0 only. Full
evaluation predictions stay on CPU; debug attention is computed one sample at
a time on a bounded set of fixed probes and confident mistakes.
"""

import csv
import argparse
import copy
import hashlib
import html
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext

import numpy as np


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding='utf-8')


def prediction_rows(logits, targets, samples, records):
    """Join by sample_token, never by DataLoader position or scene ordering."""
    logits, targets = _numpy(logits), _numpy(targets)
    if logits.ndim != 2 or targets.shape != logits.shape or len(samples) != len(logits):
        raise ValueError('Prediction, target, and sample shapes do not match')
    if not len(samples) or len(samples) != len(set(samples)):
        raise ValueError('Evaluation needs nonempty, unique sample tokens')
    if not np.isfinite(logits).all() or not np.isfinite(targets).all():
        raise ValueError('Non-finite evaluation logits or targets')
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probs = np.exp(log_probs)
    truth, pred = targets.argmax(axis=1), probs.argmax(axis=1)
    losses = -(targets * log_probs).sum(axis=1)
    rows = []
    for i, token in enumerate(samples):
        record = records[token]
        rows.append({'sample_token': token, 'scene_token': record['scene_token'],
                     'scene_name': record.get('scene_name', record['scene_token']),
                     'target': int(truth[i]), 'prediction': int(pred[i]),
                     'target_probabilities': targets[i].tolist(),
                     'probabilities': probs[i].tolist(), 'confidence': float(probs[i, pred[i]]),
                     'nll': float(losses[i]), 'correct': bool(truth[i] == pred[i]),
                     'timestamps': record['timestamps'], 'image_paths': record['image_paths']})
    return rows, probs


def attention_grid(spatial, temporal, image_shape, patch_size, use_camera_token=False):
    """Remove the optional camera token; preserve patch order and raw weights."""
    frames, channels, height, width = image_shape
    spatial, temporal = _numpy(spatial), _numpy(temporal)
    if channels != 3 or height % patch_size or width % patch_size:
        raise ValueError('Attention overlays require RGB images aligned to the patch grid')
    gh, gw = height // patch_size, width // patch_size
    expected = gh * gw + int(use_camera_token)
    if spatial.shape != (frames, expected) or temporal.shape != (frames,):
        raise ValueError(f'Attention shape mismatch: {spatial.shape}, {temporal.shape}; '
                         f'expected {(frames, expected)}, {(frames,)}')
    if not np.isfinite(spatial).all() or not np.isfinite(temporal).all():
        raise ValueError('Non-finite attention weights')
    if (spatial < 0).any() or (temporal < 0).any():
        raise ValueError('Negative attention weights')
    camera_mass = spatial[:, 0] if use_camera_token else np.zeros(frames)
    patches = spatial[:, int(use_camera_token):].reshape(frames, gh, gw)
    return patches, camera_mass


def attention_diagnostics(patches, temporal):
    """Measure concentration without amplifying tiny differences into saliency."""
    flat = _numpy(patches).astype(np.float64).reshape(len(patches), -1)
    temporal = _numpy(temporal).astype(np.float64)
    if (not np.isfinite(flat).all() or not np.isfinite(temporal).all()
            or (flat < 0).any() or (temporal < 0).any()
            or (flat.sum(-1) <= 0).any() or temporal.sum() <= 0):
        raise ValueError('Attention distributions must have positive finite mass')
    spatial = flat / flat.sum(-1, keepdims=True)
    temporal = temporal / temporal.sum()
    def concentration(probabilities):
        count = probabilities.shape[-1]
        entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum(-1)
        normalized = entropy / np.log(count) if count > 1 else np.ones_like(entropy)
        deviation = np.abs(probabilities * count - 1).max(-1)
        return normalized, deviation
    se, sd = concentration(spatial)
    te, td = concentration(temporal)
    return {'spatial_normalized_entropy': se.tolist(),
            'spatial_max_relative_deviation': sd.tolist(),
            'temporal_normalized_entropy': float(te),
            'temporal_max_relative_deviation': float(td),
            'spatial_near_uniform': bool(np.all(sd <= 0.05)),
            'temporal_near_uniform': bool(td <= 0.05),
            'near_uniform_threshold': 0.05}


def report_findings(summary):
    findings = []
    absent = [i for i, n in enumerate(summary['class_support']) if n == 0]
    if absent:
        findings.append(f'Classes {absent} have no targets; their metrics are unavailable.')
    missed = [i for i, n in enumerate(summary['class_support'])
              if n and summary['per_class_recall'][i] == 0]
    if missed:
        findings.append(f'Classes {missed} have zero recall. Inspect their probes and target labels.')
    if summary['accuracy'] <= summary['majority_baseline'] + 1e-6:
        findings.append('Accuracy does not exceed the majority-class baseline.')
    if summary.get('optimizer_steps_per_epoch') == 1:
        findings.append('Only one optimizer update per epoch: compare progress by update count.')
    cases = summary.get('attention_diagnostics', {})
    if cases:
        spatial = sum(d['spatial_near_uniform'] for d in cases.values())
        temporal = sum(d['temporal_near_uniform'] for d in cases.values())
        findings.append(f'Saved probes: spatial attention near uniform in {spatial}/{len(cases)}, '
                        f'temporal attention near uniform in {temporal}/{len(cases)} '
                        '(maximum deviation from uniform <= 5%). This measures selected probes only.')
    return findings


def plot_attention(path, images, patches, temporal, row, names, camera_mass):
    """Overlay on the exact cropped/resized model inputs, not the raw JPGs."""
    plt = _pyplot()
    images = _numpy(images)
    frames, _, height, width = images.shape
    columns = max(2, frames)
    fig = plt.figure(figsize=(max(10, frames * 3.5), 8), constrained_layout=True)
    gs = fig.add_gridspec(3, columns, height_ratios=(1, 1, 0.9))
    overlay_axes = []
    diagnostics = attention_diagnostics(patches, temporal)
    mean = patches.mean(axis=(1, 2), keepdims=True)
    relative = np.divide(patches, mean, out=np.zeros_like(patches, dtype=float), where=mean > 0)
    vmax = max(2.0, float(relative.max()))
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax)
    for i in range(frames):
        rgb = images[i].transpose(1, 2, 0).clip(0, 1)
        offset = (row['timestamps'][i] - row['timestamps'][-1]) / 1e6
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(rgb, extent=(0, width, height, 0), origin='upper')
        ax.set_title(f'Frame {i}: {offset:+.1f}s' + (' (target)' if i == frames - 1 else ''))
        ax.axis('off')
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(rgb, extent=(0, width, height, 0), origin='upper')
        # Uniform attention remains transparent, rather than tinting the whole image.
        alpha = 0.65 * np.minimum(np.abs(relative[i] - 1), 1)
        heat = ax.imshow(relative[i], cmap='coolwarm', norm=norm, alpha=alpha,
                         extent=(0, width, height, 0), origin='upper', interpolation='bilinear')
        deviation = diagnostics['spatial_max_relative_deviation'][i]
        ax.set_title(f'Temporal weight {temporal[i]:.6f}\nMax spatial deviation {deviation:.2%}', fontsize=9)
        ax.axis('off')
        overlay_axes.append(ax)
    # An opaque scalar mappable keeps the legend readable when overlays are transparent.
    from matplotlib.cm import ScalarMappable
    fig.colorbar(ScalarMappable(norm=norm, cmap='coolwarm'), ax=overlay_axes,
                 shrink=0.85, label="Weight / frame's mean patch weight (1 = uniform)")
    split = columns // 2
    ax = fig.add_subplot(gs[2, :split])
    ax.bar(range(frames), temporal, color='tab:blue')
    ax.axhline(1 / frames, color='gray', linestyle='--', label='Uniform baseline')
    ax.legend(fontsize=8)
    ax.set(xticks=range(frames), xlabel='Frame (oldest → target)', ylabel='Attention weight',
           ylim=(0, 1), title='Temporal road query')
    ax = fig.add_subplot(gs[2, split:])
    labels = [f'{i}: {name}' + (' [TARGET]' if i == row['target'] else '') +
              (' [PRED]' if i == row['prediction'] else '') for i, name in enumerate(names)]
    ax.barh(range(len(names)), row['probabilities'], color='tab:blue')
    ax.set(yticks=range(len(names)), yticklabels=labels, xlim=(0, 1), xlabel='Probability',
           title='Full-evaluation prediction')
    ax.invert_yaxis()
    fig.suptitle(f"{row['scene_name']} / {row['sample_token']}\n"
                 f"Target: {names[row['target']]} | prediction: {names[row['prediction']]} "
                 f"({row['confidence']:.1%}) | NLL: {row['nll']:.3f}", fontsize=11)
    camera_note = f' | camera-token mass: {np.round(camera_mass, 3).tolist()}' if camera_mass.any() else ''
    uniform_note = '\nNear-uniform spatial attention: no concentrated region' if diagnostics['spatial_near_uniform'] else ''
    fig.supxlabel('Road-head query weights; shared relative scale; weak deviations stay transparent' +
                  uniform_note + camera_note,
                  fontsize=9)
    try:
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)


class RoadTrainingVisualizer:
    def __init__(self, output_dir, records, class_names, every=1, attention_samples=3,
                 error_samples=3, bins=10, target_nll=0.05, run_config=None, model_internals=False):
        if every < 1 or min(attention_samples, error_samples) < 0 or bins < 1:
            raise ValueError('Invalid visualization interval or case count')
        self.names = [class_names[i] for i in range(len(class_names))]
        self.records = {r['sample_token']: r for r in records}
        if len(self.records) != len(records):
            raise ValueError('Duplicate visualization sample tokens')
        self.indices = {r['sample_token']: i for i, r in enumerate(records)}
        self.every, self.error_samples, self.bins = every, error_samples, bins
        self.target_nll = target_nll
        self.model_internals = model_internals
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.output_dir = Path(output_dir) / 'visualizations' / f'run_{stamp}'
        self.output_dir.mkdir(parents=True)
        self.history, self.epochs = [], []
        self.error_counts, self.error_streaks = Counter(), Counter()
        self.fixed_tokens = self._fixed_probes(attention_samples)
        self.run_metadata = {'created_at': datetime.now().astimezone().isoformat(), 'status': 'running',
                             'run_config': copy.deepcopy(run_config), 'last_completed_epoch': None}
        _json(self.output_dir / 'run.json', self.run_metadata)
        _json(self.output_dir / 'settings.json', {
            'every': every, 'attention_samples': attention_samples, 'error_samples': error_samples, 'bins': bins,
            'fixed_tokens': self.fixed_tokens, 'class_names': self.names, 'target_nll': target_nll,
            'model_internals': model_internals,
            'evaluation': 'same samples as training; memorization, not generalization',
            'attention': 'road-head spatial/temporal query weights, averaged over heads; '
                         'not class-conditioned saliency or VGGT attention rollout'})

    def _error_probes(self, errors):
        """Keep the highest-confidence error, then cover target classes and scenes."""
        chosen, classes, scenes = [], set(), set()
        for diverse in ('class', 'scene', 'remaining'):
            for row in errors:
                if len(chosen) >= self.error_samples:
                    return chosen
                if row['sample_token'] in chosen:
                    continue
                if diverse == 'class' and row['target'] in classes:
                    continue
                if diverse == 'scene' and row['scene_name'] in scenes:
                    continue
                chosen.append(row['sample_token'])
                classes.add(row['target'])
                scenes.add(row['scene_name'])
        return chosen

    def _fixed_probes(self, count):
        buckets = defaultdict(list)
        for r in sorted(self.records.values(), key=lambda x: (x.get('scene_name', ''), x['timestamps'][-1])):
            label = (int(np.argmax(r['annotation']['soft_label'])) if 'soft_label' in r['annotation']
                     else r['annotation']['label'])
            buckets[label].append(r['sample_token'])
        selected, step = [], 0
        while len(selected) < min(count, len(self.records)):
            for label in sorted(buckets):
                values = buckets[label]
                if step < len(values) and len(selected) < count:
                    selected.append(values[(len(values) // 2 + step) % len(values)])
            step += 1
        return selected

    def on_epoch_end(self, epoch, logits, targets, samples, metrics, training,
                     model, dataset, device, amp):
        """Called after the existing full evaluation; returns the epoch summary."""
        if self.history and epoch <= self.history[-1]['epoch']:
            raise ValueError('Visualization epochs must be strictly increasing')
        rows, probs = prediction_rows(logits, targets, samples, self.records)
        if set(samples) != set(self.records):
            raise ValueError('Visualization needs exactly the evaluated dataset samples')
        cm = np.asarray(metrics['confusion_matrix'])
        support = cm.sum(axis=1)
        if cm.shape != (len(self.names), len(self.names)) or cm.sum() != len(rows):
            raise ValueError('Confusion matrix does not match predictions')
        errors = []
        for row in rows:
            token = row['sample_token']
            self.error_counts[token] += int(not row['correct'])
            self.error_streaks[token] = self.error_streaks[token] + 1 if not row['correct'] else 0
            row.update(error_epochs=self.error_counts[token], error_streak=self.error_streaks[token])
            if not row['correct']:
                errors.append(row)
        errors.sort(key=lambda r: (-r['confidence'], -r['nll'], r['sample_token']))
        precision, recall = np.asarray(metrics['per_class_precision']), np.asarray(metrics['per_class_recall'])
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        present = support > 0
        scene_stats = []
        for name in sorted({row['scene_name'] for row in rows}):
            group = [row for row in rows if row['scene_name'] == name]
            scene_stats.append({'scene': name, 'samples': len(group),
                                'errors': sum(not r['correct'] for r in group),
                                'accuracy': sum(r['correct'] for r in group) / len(group),
                                'nll': float(np.mean([r['nll'] for r in group]))})
        summary = {**metrics, **training, 'epoch': epoch, 'samples': len(rows),
                   'class_support': support.tolist(), 'per_class_f1': f1.tolist(),
                   'majority_baseline': float(support.max() / support.sum()),
                   'macro_f1_present': float(f1[present].mean()),
                   'balanced_accuracy_present': float(recall[present].mean()),
                   'error_count': len(errors),
                   'high_confidence_errors': sum(r['confidence'] >= 0.9 for r in errors),
                   'per_scene': scene_stats}
        self.history.append(summary)
        epoch_dir = self.output_dir / f'epoch_{epoch:04d}'
        epoch_dir.mkdir()
        with (epoch_dir / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self._error_csv(epoch_dir / 'errors.csv', errors)
        self._plot_history(epoch_dir / 'curves.png')
        shutil.copyfile(epoch_dir / 'curves.png', self.output_dir / 'curves.png')
        self._plot_confusion(cm, epoch_dir / 'confusion_matrix.png')
        self._plot_reliability(probs, rows, epoch_dir / 'reliability.png')
        cases = {}
        summary['error_attention_tokens'] = self._error_probes(errors)
        if epoch % self.every == 0:
            tokens = list(dict.fromkeys(self.fixed_tokens + summary['error_attention_tokens']))
            if tokens:
                cases = self._render_cases(tokens, {r['sample_token']: r for r in rows}, epoch_dir,
                                           model, dataset, device, amp,
                                           optimizer_steps=training.get('optimizer_steps'))
        summary['attention_diagnostics'] = {
            token: json.loads((epoch_dir / relative).with_suffix('.json').read_text(encoding='utf-8'))['attention_diagnostics']
            for token, relative in cases.items()}
        summary['findings'] = report_findings(summary)
        summary['reported_at'] = datetime.now().astimezone().isoformat()
        _json(epoch_dir / 'metrics.json', summary)
        self._epoch_report(epoch_dir, summary, errors, cases)
        self.epochs.append(epoch)
        self.run_metadata.update(last_completed_epoch=epoch, updated_at=summary['reported_at'])
        _json(self.output_dir / 'run.json', self.run_metadata)
        # Write complete summaries once, after attention diagnostics are available.
        with (self.output_dir / 'history.jsonl').open('w', encoding='utf-8') as stream:
            for item in self.history:
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        self._index()
        return summary

    def finish(self, reason):
        self.run_metadata.update(status='completed', stop_reason=reason,
                                 updated_at=datetime.now().astimezone().isoformat())
        _json(self.output_dir / 'run.json', self.run_metadata)
        self._index()

    def _error_csv(self, path, errors):
        fields = ['sample_token', 'scene_name', 'target', 'prediction', 'confidence', 'nll',
                  'error_epochs', 'error_streak', *[f'p_{name}' for name in self.names], 'image_paths']
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in errors:
                writer.writerow({**{key: row[key] for key in fields if key in row and key != 'image_paths'},
                                 **{f'p_{name}': row['probabilities'][i] for i, name in enumerate(self.names)},
                                 'image_paths': json.dumps(row['image_paths'], ensure_ascii=False)})

    def _plot_history(self, path):
        plt = _pyplot()
        from matplotlib.ticker import MaxNLocator
        fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
        has_updates = all(isinstance(r.get('optimizer_steps'), int) for r in self.history)
        epochs = [r['optimizer_steps'] if has_updates else r['epoch'] for r in self.history]
        def line(ax, key, label, **kwargs):
            ax.plot(epochs, [r[key] for r in self.history], label=label, marker='.', **kwargs)
        ax = axes[0, 0]
        line(ax, 'train_acc', 'During training')
        line(ax, 'accuracy', 'Same-sample evaluation')
        ax.axhline(self.history[-1]['majority_baseline'], color='gray', linestyle='--', label='Majority baseline')
        ax.set(title='Accuracy', ylim=(0, 1.03))
        line(axes[0, 1], 'train_loss', 'Weighted training loss')
        axes[0, 1].set(title='Training objective (weighted CE)')
        axes[0, 1].set_ylim(bottom=0)
        line(axes[1, 0], 'nll', 'Unweighted same-sample NLL')
        axes[1, 0].axhline(self.target_nll, color='gray', linestyle='--', label=f'Target {self.target_nll:g}')
        axes[1, 0].set(title='NLL')
        axes[1, 0].set_ylim(bottom=0)
        line(axes[1, 1], 'macro_f1', 'Macro F1 (all classes)')
        line(axes[1, 1], 'macro_f1_present', 'Macro F1 (present classes)')
        line(axes[1, 1], 'balanced_accuracy_present', 'Balanced accuracy (present)')
        axes[1, 1].set(title='Class-balanced metrics', ylim=(0, 1.03))
        for i, name in enumerate(self.names):
            n = self.history[-1]['class_support'][i]
            if n:
                axes[2, 0].plot(epochs, [r['per_class_recall'][i] for r in self.history],
                                marker='.', label=f'{i}: {name} (n={n})')
        axes[2, 0].set(title='Recall by class', ylim=(0, 1.03))
        line(axes[2, 1], 'ece', 'ECE')
        line(axes[2, 1], 'brier', 'Brier score')
        axes[2, 1].set(title='Calibration scores (lower is better)')
        axes[2, 1].set_ylim(bottom=0)
        for ax in axes.flat:
            ax.set_xlabel('Optimizer updates' if has_updates else 'Epoch')
            ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True, min_n_ticks=1))
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle('Training and evaluation use the SAME samples')
        try:
            fig.savefig(path, dpi=120)
        finally:
            plt.close(fig)

    def _plot_confusion(self, cm, path):
        plt = _pyplot()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        support = cm.sum(axis=1)
        normalized = np.divide(cm, support[:, None], out=np.zeros_like(cm, dtype=float),
                               where=support[:, None] > 0)
        for ax, values, title in zip(axes, (cm, normalized), ('Counts', 'Row-normalized recall')):
            cmap = plt.get_cmap('Blues').with_extremes(bad='#eeeeee')
            masked = np.ma.masked_where(np.broadcast_to(support[:, None] == 0, cm.shape), values)
            heat = ax.imshow(masked, cmap=cmap, vmin=0, vmax=max(float(values.max()), 1e-12))
            labels = [f'{i}: {name.replace("_", " ")}' for i, name in enumerate(self.names)]
            ax.set(xticks=range(len(self.names)), yticks=range(len(self.names)),
                   xticklabels=labels, yticklabels=labels, xlabel='Predicted', ylabel='True', title=title)
            plt.setp(ax.get_xticklabels(), rotation=35, ha='right', fontsize=8)
            plt.setp(ax.get_yticklabels(), fontsize=8)
            for i in range(len(self.names)):
                for j in range(len(self.names)):
                    value = 'n/a' if support[i] == 0 else (str(cm[i, j]) if title == 'Counts' else f'{values[i, j]:.0%}')
                    ax.text(j, i, value, ha='center', va='center', fontsize=8,
                            color='white' if support[i] and values[i, j] > values.max() / 2 else 'black')
            fig.colorbar(heat, ax=ax, shrink=0.75)
        try:
            fig.savefig(path, dpi=120)
        finally:
            plt.close(fig)

    def _plot_reliability(self, probs, rows, path):
        plt = _pyplot()
        confidence = probs.max(axis=1)
        correct = np.array([r['correct'] for r in rows])
        bin_ids = np.minimum((confidence * self.bins).astype(int), self.bins - 1)
        centers, accuracy, mean_confidence, counts = [], [], [], []
        for i in range(self.bins):
            mask = bin_ids == i
            if mask.any():
                centers.append((i + 0.5) / self.bins)
                accuracy.append(correct[mask].mean())
                mean_confidence.append(confidence[mask].mean())
                counts.append(int(mask.sum()))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        ax = axes[0]
        ax.bar(centers, accuracy, width=0.8 / self.bins, alpha=0.6, label='Accuracy')
        ax.scatter(centers, mean_confidence, color='tab:orange', label='Mean confidence')
        ax.plot([0, 1], [0, 1], '--', color='gray')
        for x, count in zip(centers, counts):
            ax.text(x, 1.04, f'n={count}', ha='center', fontsize=7)
        ax.set(xlim=(0, 1), ylim=(0, 1.12), xlabel='Confidence bin', ylabel='Accuracy / confidence',
               title='Reliability (argmax targets)')
        ax.legend(fontsize=8)
        edges = np.linspace(0, 1, self.bins + 1)
        axes[1].hist([confidence[correct], confidence[~correct]], bins=edges,
                     label=['Correct', 'Wrong'], color=['tab:blue', 'tab:red'], stacked=True)
        axes[1].set(xlabel='Prediction confidence', ylabel='Samples', title='Confidence distribution')
        axes[1].legend()
        try:
            fig.savefig(path, dpi=120)
        finally:
            plt.close(fig)

    def _render_cases(self, tokens, rows, epoch_dir, model, dataset, device, amp, optimizer_steps=None):
        import torch
        from scripts.road_common import autocast_context
        case_dir = epoch_dir / 'cases'
        case_dir.mkdir()
        rendered = {}
        was_training = model.training
        model.eval()
        try:
            for token in tokens:
                item = dataset[self.indices[token]]
                if item['sample_token'] != token:
                    raise ValueError('Visualization dataset order differs from its records')
                images = item['images']
                inspect = self.model_internals and bool(self.fixed_tokens) and token == self.fixed_tokens[0]
                if inspect:
                    from scripts.road_model_inspector import ModelInspector
                    inspector = ModelInspector(model)
                else:
                    inspector = nullcontext()
                with inspector, torch.no_grad(), autocast_context(device, amp):
                    out = model(images.unsqueeze(0).to(device, non_blocking=True), debug=True)
                spatial = _numpy(out['spatial_attention'][0])
                temporal = _numpy(out['temporal_attention'][0])
                # Keep the uncalibrated probability convention used by full evaluation.
                debug_probs = _numpy(out['road_logits'][0].float().softmax(-1))
                del out
                patches, camera_mass = attention_grid(spatial, temporal, tuple(images.shape),
                                                       model.aggregator.patch_size,
                                                       model.road_head.use_camera_token)
                stem = hashlib.sha256(token.encode('utf-8')).hexdigest()[:16]
                if inspect:
                    inspector.write(case_dir / f'{stem}_internals.html', images, rows[token],
                                    self.names, debug_probs,
                                    {'epoch': int(epoch_dir.name.split('_')[-1]),
                                     'optimizer_steps': optimizer_steps,
                                     'source': 'current model, same debug forward as attention overlay'})
                plot_attention(case_dir / f'{stem}.png', images, patches, temporal, rows[token],
                               self.names, camera_mass)
                np.savez_compressed(case_dir / f'{stem}.npz', spatial_attention=spatial,
                                    patch_attention=patches, temporal_attention=temporal,
                                    camera_mass=camera_mass, image_shape=np.array(images.shape),
                                    evaluation_probabilities=rows[token]['probabilities'],
                                    attention_pass_probabilities=debug_probs)
                _json(case_dir / f'{stem}.json', {**rows[token],
                      'attention_pass_probabilities': debug_probs.tolist(),
                      'attention_pass_prediction': int(debug_probs.argmax()),
                      'overlay_coordinates': 'preprocessed model input',
                      'heat_scale': 'weight / mean patch weight; uniform=1; shared across frames',
                      'attention_diagnostics': attention_diagnostics(patches, temporal),
                      'camera_mass': camera_mass.tolist()})
                rendered[token] = f'cases/{stem}.png'
        finally:
            model.train(was_training)
        return rendered

    def _epoch_report(self, epoch_dir, summary, errors, cases):
        escape = html.escape
        def metric(value, support):
            return f'{value:.3f}' if support else 'n/a'
        class_rows = ''.join(f'<tr><td>{i}: {escape(name)}</td><td>{summary["class_support"][i]}</td>'
                             f'<td>{metric(summary["per_class_precision"][i], summary["class_support"][i])}</td>'
                             f'<td>{metric(summary["per_class_recall"][i], summary["class_support"][i])}</td>'
                             f'<td>{metric(summary["per_class_f1"][i], summary["class_support"][i])}</td></tr>'
                             for i, name in enumerate(self.names))
        scene_rows = ''.join(f'<tr><td>{escape(r["scene"])}</td><td>{r["samples"]}</td>'
                             f'<td>{r["errors"]}</td><td>{r["accuracy"]:.1%}</td><td>{r["nll"]:.3f}</td></tr>'
                             for r in summary['per_scene'])
        pairs = Counter((r['target'], r['prediction']) for r in errors)
        pair_rows = ''.join(f'<tr><td>{escape(self.names[a])} → {escape(self.names[b])}</td><td>{n}</td></tr>'
                            for (a, b), n in pairs.most_common())
        def token_link(token):
            return f'<a href="#case-{escape(token)}">{escape(token)}</a>' if token in cases else escape(token)
        error_rows = ''.join(f'<tr><td>{token_link(r["sample_token"])}</td><td>{escape(r["scene_name"])}</td>'
                             f'<td>{escape(self.names[r["target"]])}</td>'
                             f'<td>{escape(self.names[r["prediction"]])}</td><td>{r["confidence"]:.1%}</td>'
                             f'<td>{r["nll"]:.3f}</td><td>{r["error_epochs"]}/{r["error_streak"]}</td></tr>'
                             for r in errors[:20])
        def galleries(tokens):
            def internals_link(token):
                path = Path(cases[token]).with_name(Path(cases[token]).stem + '_internals.html')
                return (f' · <a href="{path.as_posix()}">Interactive model internals</a>'
                        if (epoch_dir / path).is_file() else '')
            return ''.join(f'<figure id="case-{escape(token)}"><figcaption>{escape(token)}{internals_link(token)}</figcaption>'
                           f'<a href="{cases[token]}"><img loading="lazy" src="{cases[token]}" '
                           f'alt="Attention for {escape(token)}"></a></figure>' for token in tokens if token in cases)
        findings = ''.join(f'<li>{escape(message)}</li>' for message in summary.get('findings', []))
        steps = summary.get('optimizer_steps', 'unavailable for historical report')
        body = f'''<h1>Epoch {summary['epoch']}</h1><p><a href="../index.html">Run overview</a></p>
<nav><a href="#diagnostics">Diagnostics</a> · <a href="#metrics">Metrics</a> ·
<a href="#errors">Errors</a> · <a href="#attention">Attention</a></nav>
<p>Same-sample memorization: accuracy {summary['accuracy']:.2%}, NLL {summary['nll']:.4f},
present-class F1 {summary['macro_f1_present']:.4f}, errors {summary['error_count']}/{summary['samples']},
confidence ≥ 90% errors: {summary['high_confidence_errors']}.</p>
<p>Optimizer updates: {steps}. Majority baseline: {summary['majority_baseline']:.2%}.</p>
<h2 id="diagnostics">Diagnostics</h2><ul>{findings}</ul>
<p>Labels describe the final frame. Soft-target error counts use argmax targets.
Attention shows road-head query weights averaged over heads. Near-uniform attention stays transparent;
it does not identify a concentrated region. Target labels are not verified by this report.</p>
<p><a href="metrics.json">Metrics</a> · <a href="predictions.jsonl">All predictions</a> ·
<a href="errors.csv">All errors CSV</a></p>
<h2 id="metrics">Metric figures</h2><img src="curves.png" alt="Metric trends"><img src="confusion_matrix.png" alt="Confusion matrices">
<img src="reliability.png" alt="Reliability and confidence distribution">
<h2>Per-class metrics</h2><table><tr><th>Class</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{class_rows}</table>
<h2>Per-scene metrics</h2><table><tr><th>Scene</th><th>Samples</th><th>Errors</th><th>Accuracy</th><th>NLL</th></tr>{scene_rows}</table>
<h2>Error pairs</h2><table><tr><th>True → prediction</th><th>Count</th></tr>{pair_rows}</table>
<h2 id="errors">Most confident errors (top 20)</h2><table><tr><th>Token</th><th>Scene</th><th>Target</th><th>Predicted</th>
<th>Confidence</th><th>NLL</th><th>Error epochs / streak</th></tr>{error_rows}</table>
<h2 id="attention">Fixed attention probes</h2>{galleries(self.fixed_tokens)}
<h2>Error attention probes</h2><p>New training reports cover different target classes and scenes;
historical reports can display only the attention cases that were saved.</p>
{galleries([t for t in summary.get('error_attention_tokens', []) if t not in self.fixed_tokens])}'''
        self._html(epoch_dir / 'index.html', f'Epoch {summary["epoch"]}', body)

    def _index(self):
        latest = self.history[-1]
        best = min(self.history, key=lambda r: r['nll'])
        metadata = getattr(self, 'run_metadata', {})
        updated = metadata.get('updated_at', 'unavailable for historical report')
        status = metadata.get('status', 'historical')
        links = ''.join(f'<tr><td><a href="epoch_{r["epoch"]:04d}/index.html">Epoch {r["epoch"]}</a></td>'
                        f'<td>{r.get("optimizer_steps", "n/a")}</td><td>{r["accuracy"]:.2%}</td>'
                        f'<td>{r["nll"]:.4f}</td><td>{r["macro_f1_present"]:.3f}</td>'
                        f'<td>{r["error_count"]}</td></tr>' for r in reversed(self.history))
        findings = ''.join(f'<li>{html.escape(message)}</li>' for message in latest.get('findings', []))
        configuration = ('Configuration snapshot is saved for this run.' if metadata.get('run_config')
                         else 'Original training configuration is unavailable for this historical run.')
        self._html(self.output_dir / 'index.html', 'Road training visualization',
                   f'<h1>Road training visualization</h1><p>Train = evaluation; memorization metrics.</p>'
                   f'<p>Report status: {html.escape(status)}. Updated: {html.escape(updated)}.</p>'
                   '<p>This page belongs to one run. A different run started with --no-vis does not update it.</p>'
                   f'<p><a href="history.jsonl">Metric history</a> · <a href="settings.json">Settings</a> · '
                   f'<a href="run.json">Run metadata / configuration</a></p><p>{configuration}</p>'
                   f'<div class="cards"><div>Latest epoch <strong>{latest["epoch"]}</strong></div>'
                   f'<div>Accuracy <strong>{latest["accuracy"]:.2%}</strong></div>'
                   f'<div>NLL <strong>{latest["nll"]:.4f}</strong></div>'
                   f'<div>Present-class F1 <strong>{latest["macro_f1_present"]:.3f}</strong></div>'
                   f'<div>Errors <strong>{latest["error_count"]}/{latest["samples"]}</strong></div></div>'
                   f'<p><a href="epoch_{latest["epoch"]:04d}/index.html">Open latest report</a> · '
                   f'Best saved report NLL: {best["nll"]:.4f} at epoch {best["epoch"]}.</p>'
                   f'<h2>Latest diagnostics</h2><ul>{findings}</ul>'
                   '<h2>Epoch reports</h2><table><tr><th>Epoch</th><th>Updates</th><th>Accuracy</th>'
                   f'<th>NLL</th><th>Present F1</th><th>Errors</th></tr>{links}</table>'
                   '<h2>Metric trends</h2><img src="curves.png" alt="Metric trends">')

    @staticmethod
    def _html(path, title, body):
        Path(path).write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font:15px system-ui,sans-serif;margin:24px;color:#182435;max-width:1400px}}
img{{max-width:100%;height:auto}}table{{border-collapse:collapse;display:block;overflow-x:auto}}
th,td{{padding:7px 10px;text-align:left;border-bottom:1px solid #ddd}}figure{{margin:20px 0}}
figcaption{{overflow-wrap:anywhere}}a{{color:#1769aa}}nav{{margin:12px 0}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}}
.cards>div{{padding:12px 18px;border:1px solid #ddd;border-radius:8px}}
.cards strong{{display:block;font-size:24px}}</style></head><body>{body}</body></html>''',
                              encoding='utf-8')


def refresh_reports(source, output_dir=None, config_path=None):
    """Rebuild completed reports on CPU from their saved predictions/attention.

    Original reports, model weights, and training processes are left intact.
    An old run without a config snapshot requires an explicit rendering config
    when reconstructing case images; it is not claimed as that run's training config.
    """
    source = Path(source).resolve()
    destination = Path(output_dir).resolve() if output_dir else source / 'reviewed'
    if destination == source or destination in source.parents:
        raise ValueError('Refreshed reports need a separate output directory')
    folders = sorted(p for p in source.glob('epoch_*')
                     if p.is_dir() and (p / 'index.html').is_file())
    if not folders:
        raise ValueError('No completed epoch reports found')
    settings = json.loads((source / 'settings.json').read_text(encoding='utf-8'))
    metadata_path = source / 'run.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
    rendering_config = metadata.get('run_config')
    if config_path:
        import yaml
        rendering_config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    has_cases = any(list((folder / 'cases').glob('*.npz')) for folder in folders)
    if has_cases and not rendering_config:
        raise ValueError('This historical run has no config snapshot; pass --config for image reconstruction')
    dataset_type = None
    if has_cases:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from training.data.nuscenes_road_dataset import NuScenesRoadSequenceDataset
        dataset_type = NuScenesRoadSequenceDataset
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / 'settings.json', destination / 'settings.json')
    vis = RoadTrainingVisualizer.__new__(RoadTrainingVisualizer)
    vis.output_dir, vis.names = destination, settings['class_names']
    vis.fixed_tokens, vis.error_samples = settings['fixed_tokens'], settings['error_samples']
    vis.bins = settings.get('bins', (rendering_config or {}).get('calibration', {}).get('ece_bins', 10))
    vis.target_nll = settings.get('target_nll', 0.05)
    vis.history, vis.epochs = [], []
    vis.run_metadata = {**metadata, 'status': 'reviewed_archive', 'source_report': str(source),
                        'run_config': metadata.get('run_config'),
                        'render_config': rendering_config,
                        'updated_at': datetime.now().astimezone().isoformat()}
    for folder in folders:
        target = destination / folder.name
        shutil.copytree(folder, target, dirs_exist_ok=True)
        summary = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()
                if line.strip()]
        if len(rows) != summary['samples']:
            raise ValueError(f'Incomplete predictions: {folder}')
        errors = sorted((r for r in rows if not r['correct']),
                        key=lambda r: (-r['confidence'], -r['nll'], r['sample_token']))
        cases, diagnostics = {}, {}
        for path in sorted((target / 'cases').glob('*.npz')):
            case = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            with np.load(path, allow_pickle=False) as data:
                patches, temporal = data['patch_attention'], data['temporal_attention']
                camera, shape = data['camera_mass'], tuple(data['image_shape'].tolist())
            record = {**case, 'annotation': {'soft_label': case['target_probabilities']}}
            d = rendering_config['data']
            images = dataset_type([record], d['preprocess_mode'], 0, d.get('image_width', 518))[0]['images']
            if tuple(images.shape) != shape:
                raise ValueError(f'Render config does not match saved model-input shape: {path}')
            plot_attention(path.with_suffix('.png'), images, patches, temporal, case, vis.names, camera)
            stats = attention_diagnostics(patches, temporal)
            case['attention_diagnostics'] = stats
            case['heat_scale'] = 'weight / mean patch weight; uniform=1; shared across frames'
            _json(path.with_suffix('.json'), case)
            cases[case['sample_token']] = f'cases/{path.stem}.png'
            diagnostics[case['sample_token']] = stats
        summary['attention_diagnostics'] = diagnostics
        # Historical cases cannot be recomputed from a different epoch's model.
        summary['error_attention_tokens'] = [r['sample_token'] for r in errors
                                             if r['sample_token'] in cases][:vis.error_samples]
        summary['findings'] = report_findings(summary)
        summary['refreshed_at'] = vis.run_metadata['updated_at']
        vis.history.append(summary)
        vis.epochs.append(summary['epoch'])
        vis._plot_history(target / 'curves.png')
        vis._plot_confusion(np.asarray(summary['confusion_matrix']), target / 'confusion_matrix.png')
        vis._plot_reliability(np.asarray([r['probabilities'] for r in rows]), rows, target / 'reliability.png')
        vis._epoch_report(target, summary, errors, cases)
        _json(target / 'metrics.json', summary)
    vis.run_metadata['last_completed_epoch'] = vis.epochs[-1]
    _json(destination / 'run.json', vis.run_metadata)
    with (destination / 'history.jsonl').open('w', encoding='utf-8') as stream:
        for item in vis.history:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
    shutil.copyfile(destination / folders[-1].name / 'curves.png', destination / 'curves.png')
    vis._index()
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refresh completed road reports without loading a model')
    parser.add_argument('--refresh', required=True, help='Existing visualization run directory')
    parser.add_argument('--output-dir', help='Separate report directory; defaults to RUN/reviewed')
    parser.add_argument('--config', help='Rendering config required for old runs without a config snapshot')
    args = parser.parse_args()
    print(refresh_reports(args.refresh, args.output_dir, args.config) / 'index.html')
