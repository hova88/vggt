"""CPU-only checks for token joins, attention geometry, and epoch reports."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.road_training_visualizer import (RoadTrainingVisualizer, attention_grid,
                                              attention_diagnostics, plot_attention, prediction_rows,
                                              refresh_reports, report_findings)


NAMES = {0: 'elevated_up', 1: 'elevated_down', 2: 'main_road', 3: 'side_road', 4: 'intersection'}


def records():
    return [{'sample_token': f'token-{i}', 'scene_token': f'scene-{i}',
             'scene_name': f'scene<{i}>', 'timestamps': [0, 500000, 1000000, 1500000],
             'image_paths': [f'{i}-{j}.jpg' for j in range(4)], 'annotation': {'label': label}}
            for i, label in enumerate((2, 3, 4))]


def metrics(logits, targets):
    _, probs = prediction_rows(logits, targets, [r['sample_token'] for r in records()],
                               {r['sample_token']: r for r in records()})
    truth, pred = targets.argmax(1), probs.argmax(1)
    cm = np.bincount(truth * 5 + pred, minlength=25).reshape(5, 5)
    precision = cm.diagonal() / np.maximum(cm.sum(0), 1)
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {'accuracy': float((truth == pred).mean()), 'macro_f1': float(f1.mean()),
            'per_class_precision': precision.tolist(), 'per_class_recall': recall.tolist(),
            'confusion_matrix': cm.tolist(), 'nll': float(-(targets * np.log(probs)).sum(1).mean()),
            'brier': float(((probs - targets) ** 2).sum(1).mean()), 'ece': 0.1}


class VisualizerTests(unittest.TestCase):
    def test_uniform_attention_is_detected_without_false_concentration(self):
        uniform = np.full((4, 3, 7), 0.6 / 21)
        result = attention_diagnostics(uniform, np.full(4, 0.25))
        self.assertTrue(result['spatial_near_uniform'])
        self.assertTrue(result['temporal_near_uniform'])
        np.testing.assert_allclose(result['spatial_normalized_entropy'], 1)
        np.testing.assert_allclose(result['spatial_max_relative_deviation'], 0, atol=1e-12)
        concentrated = np.zeros((4, 3, 7))
        concentrated[:, 0, 0] = 0.6
        result = attention_diagnostics(concentrated, np.array([0, 0, 0, 1]))
        self.assertFalse(result['spatial_near_uniform'])
        self.assertFalse(result['temporal_near_uniform'])
        np.testing.assert_allclose(result['spatial_normalized_entropy'], 0)
        with self.assertRaises(ValueError):
            attention_diagnostics(np.zeros((4, 3, 7)), np.full(4, 0.25))

    def test_error_probes_cover_classes_and_scenes(self):
        with tempfile.TemporaryDirectory() as temporary:
            vis = RoadTrainingVisualizer(temporary, records(), NAMES, error_samples=3)
            errors = [{'sample_token': str(i), 'target': target, 'scene_name': scene}
                      for i, (target, scene) in enumerate(((4, 'a'), (4, 'a'), (4, 'a'),
                                                         (3, 'b'), (2, 'c')))]
            self.assertEqual(vis._error_probes(errors), ['0', '3', '4'])
            vis.error_samples = 4
            self.assertEqual(vis._error_probes(errors), ['0', '3', '4', '1'])
            vis.error_samples = 0
            self.assertEqual(vis._error_probes(errors), [])

    def test_fixed_probes_cover_present_classes_and_are_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            vis = RoadTrainingVisualizer(temporary, records(), NAMES)
            labels = [vis.records[token]['annotation']['label'] for token in vis.fixed_tokens]
            self.assertEqual(labels, [2, 3, 4])
            self.assertEqual(vis.fixed_tokens, vis._fixed_probes(3))

    def test_join_follows_evaluation_tokens_and_soft_nll(self):
        by_token = {r['sample_token']: r for r in records()}
        samples = ['token-2', 'token-0']
        logits = np.array([[0, 0, 0, 0, 8], [0, 0, 5, 0, 1]], dtype=float)
        targets = np.array([[0, 0, 0, 0, 1], [0, 0, 0.7, 0, 0.3]])
        rows, probs = prediction_rows(logits, targets, samples, by_token)
        self.assertEqual(rows[0]['scene_token'], 'scene-2')
        self.assertEqual(rows[1]['image_paths'], by_token['token-0']['image_paths'])
        self.assertAlmostEqual(rows[1]['nll'], -float(np.dot(targets[1], np.log(probs[1]))))
        with self.assertRaises(ValueError):
            prediction_rows(logits, targets, ['token-0', 'token-0'], by_token)

    def test_camera_token_exclusion_and_patch_order(self):
        spatial = np.array([[0.4, 0.01, 0.02, 0.03, 0.04, 0.2, 0.3],
                            [0.4, 0.3, 0.2, 0.04, 0.03, 0.02, 0.01]])
        grid, camera = attention_grid(spatial, np.array([0.25, 0.75]), (2, 3, 28, 42), 14, True)
        np.testing.assert_array_equal(grid[0], spatial[0, 1:].reshape(2, 3))
        np.testing.assert_array_equal(camera, [0.4, 0.4])
        grid_without_camera, _ = attention_grid(spatial[:, 1:], np.array([0.25, 0.75]),
                                                (2, 3, 28, 42), 14, False)
        np.testing.assert_array_equal(grid, grid_without_camera)
        with self.assertRaises(ValueError):
            attention_grid(spatial, np.array([0.25, 0.75]), (2, 3, 28, 43), 14, True)

    def test_reports_preserve_epochs_and_handle_zero_errors_missing_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            vis = RoadTrainingVisualizer(temporary, records(), NAMES, attention_samples=0, error_samples=0)
            target = np.eye(5)[[2, 3, 4]]
            wrong = np.zeros((3, 5))
            wrong[np.arange(3), [2, 2, 4]] = 10
            training = {'train_loss': 0.5, 'train_acc': 0.5, 'samples_per_sec': 10}
            first = vis.on_epoch_end(1, wrong, target, [r['sample_token'] for r in records()],
                                     metrics(wrong, target), training, None, None, None, None)
            correct = target * 10
            second = vis.on_epoch_end(2, correct, target, [r['sample_token'] for r in records()],
                                      metrics(correct, target), training, None, None, None, None)
            self.assertEqual(first['error_count'], 1)
            self.assertEqual(first['high_confidence_errors'], 1)
            self.assertEqual(second['error_count'], 0)
            self.assertAlmostEqual(second['macro_f1_present'], 1)
            self.assertAlmostEqual(second['macro_f1'], 0.6)
            epoch = vis.output_dir / 'epoch_0002'
            rows = [json.loads(line) for line in (epoch / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[1]['error_epochs'], 1)
            self.assertEqual(rows[1]['error_streak'], 0)
            self.assertEqual(len((epoch / 'errors.csv').read_text(encoding='utf-8').splitlines()), 1)
            self.assertIn('scene&lt;0&gt;', (epoch / 'index.html').read_text(encoding='utf-8'))
            self.assertTrue((vis.output_dir / 'epoch_0001' / 'metrics.json').is_file())
            self.assertEqual(len((vis.output_dir / 'history.jsonl').read_text(encoding='utf-8').splitlines()), 2)
            for filename in ('curves.png', 'confusion_matrix.png', 'reliability.png'):
                self.assertGreater((epoch / filename).stat().st_size, 1000)

    def test_config_snapshot_update_counts_and_completed_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = {'training': {'batch_size': 190}}
            vis = RoadTrainingVisualizer(temporary, records(), NAMES, attention_samples=0,
                                         error_samples=0, run_config=config)
            config['training']['batch_size'] = 10
            targets = np.eye(5)[[2, 3, 4]]
            logits = targets * 10
            summary = vis.on_epoch_end(1, logits, targets, [r['sample_token'] for r in records()],
                                      metrics(logits, targets),
                                      {'train_loss': 0.1, 'train_acc': 1, 'samples_per_sec': 10,
                                       'optimizer_steps': 1, 'optimizer_steps_per_epoch': 1},
                                      None, None, None, None)
            self.assertTrue(any('one optimizer update' in s for s in report_findings(summary)))
            vis.finish('epoch_limit')
            metadata = json.loads((vis.output_dir / 'run.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['run_config']['training']['batch_size'], 190)
            self.assertEqual(metadata['status'], 'completed')
            self.assertEqual(metadata['stop_reason'], 'epoch_limit')
            page = (vis.output_dir / 'index.html').read_text(encoding='utf-8')
            self.assertIn('Latest epoch', page)
            self.assertIn('Open latest report', page)
            self.assertIn('completed', page)
            epoch_page = (vis.output_dir / 'epoch_0001/index.html').read_text(encoding='utf-8')
            self.assertIn('<td>0</td><td>n/a</td><td>n/a</td><td>n/a</td>', epoch_page)

    def test_refresh_preserves_original_report_and_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            vis = RoadTrainingVisualizer(temporary, records(), NAMES, attention_samples=0, error_samples=0)
            targets = np.eye(5)[[2, 3, 4]]
            logits = targets * 10
            vis.on_epoch_end(1, logits, targets, [r['sample_token'] for r in records()],
                             metrics(logits, targets),
                             {'train_loss': 0.1, 'train_acc': 1, 'samples_per_sec': 10},
                             None, None, None, None)
            # The first released report format did not record bins or run metadata.
            settings_path = vis.output_dir / 'settings.json'
            settings = json.loads(settings_path.read_text(encoding='utf-8'))
            settings.pop('bins')
            settings_path.write_text(json.dumps(settings), encoding='utf-8')
            (vis.output_dir / 'run.json').unlink()
            original = (vis.output_dir / 'index.html').read_bytes()
            original_metrics = (vis.output_dir / 'epoch_0001/metrics.json').read_bytes()
            reviewed = refresh_reports(vis.output_dir)
            self.assertEqual((vis.output_dir / 'index.html').read_bytes(), original)
            self.assertEqual((vis.output_dir / 'epoch_0001/metrics.json').read_bytes(), original_metrics)
            self.assertEqual((reviewed / 'epoch_0001/predictions.jsonl').read_bytes(),
                             (vis.output_dir / 'epoch_0001/predictions.jsonl').read_bytes())
            self.assertIn('reviewed_archive', (reviewed / 'index.html').read_text(encoding='utf-8'))
            with self.assertRaises(ValueError):
                refresh_reports(vis.output_dir, vis.output_dir)

    def test_single_frame_attention_renderer(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = {'sample_token': 'case', 'scene_name': 'scene', 'timestamps': [0], 'target': 2,
                   'prediction': 4, 'confidence': 0.8, 'nll': 2.0,
                   'probabilities': [0.05, 0.05, 0.05, 0.05, 0.8]}
            image = np.ones((1, 3, 28, 42)) * 0.5
            patches = np.arange(6).reshape(1, 2, 3) / 15
            path = Path(temporary) / 'attention.png'
            plot_attention(path, image, patches, np.array([1.0]), row,
                           list(NAMES.values()), np.zeros(1))
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
