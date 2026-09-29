"""Numerical checks against real, small VGGT/DINO modules; no weights download."""
import base64
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.road_model_inspector import ModelInspector, pack_arrays
from vggt.models.aggregator import Aggregator
from vggt.layers.vision_transformer import DinoVisionTransformer
from vggt.heads.road_probability_head import VGGTRoadClassifier


def small_model(camera=False):
    aggregator = Aggregator(img_size=42, patch_size=14, embed_dim=16, depth=2,
                            num_heads=2, patch_embed='conv', cached_layer_indices=(1,))
    aggregator.patch_embed = DinoVisionTransformer(img_size=42, patch_size=14, embed_dim=16,
                                                   depth=2, num_heads=2, num_register_tokens=2,
                                                   qkv_bias=True, block_chunks=0)
    return VGGTRoadClassifier(aggregator, {'road_dim': 8, 'spatial_num_heads': 2,
        'temporal_num_heads': 2, 'temporal_layers': 1, 'temporal_ffn_dim': 16,
        'dropout': 0, 'use_camera_token': camera}).eval()


class InspectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = small_model()
        self.images = torch.rand(1, 2, 3, 28, 42)

    def test_observation_preserves_logits_and_captures_real_modules(self):
        with torch.no_grad():
            baseline = self.model(self.images, debug=True)
            with ModelInspector(self.model) as inspector:
                observed = self.model(self.images, debug=True)
        torch.testing.assert_close(observed['road_logits'], baseline['road_logits'], rtol=0, atol=0)
        self.assertFalse(inspector.handles)
        self.assertFalse(inspector.pending)
        self.assertEqual(inspector.data['grid'], [2, 3])
        self.assertEqual([s['id'] for s in inspector.data['stages']],
                         ['dino', 'frame_1', 'global_1', 'road_spatial'])
        for stage in inspector.data['stages']:
            if 'gelu' in stage:
                torch.testing.assert_close(torch.from_numpy(stage['gelu']['output']),
                    F.gelu(torch.from_numpy(stage['gelu']['input'])), atol=1e-6, rtol=1e-6)
        embedding = inspector.data['embedding']
        np.testing.assert_allclose(embedding['projected'] + embedding['position_contribution'],
                                   embedding['prepared'], atol=1e-7)
        conv = inspector.data['conv']
        reference = np.einsum('fqchw,ochw->fqo', conv['input_patches'], conv['kernel']) + conv['bias']
        np.testing.assert_allclose(reference, conv['output'], rtol=1e-5, atol=1e-6)

    def test_effective_qk_include_norm_rope_and_preserve_global_token_order(self):
        captured = {}
        attention = self.model.aggregator.global_blocks[-1].attn
        def original_input(module, args, kwargs):
            captured['input'], captured['pos'] = args[0].clone(), kwargs['pos'].clone()
        handle = attention.register_forward_pre_hook(original_input, with_kwargs=True)
        block_handle = self.model.aggregator.global_blocks[-1].register_forward_hook(
            lambda module, args, output: captured.update(output=output.clone()))
        try:
            with torch.no_grad(), ModelInspector(self.model, channels=3, max_queries=3) as inspector:
                self.model(self.images, debug=True)
            with torch.no_grad():
                raw = attention.qkv(captured['input']).reshape(1, 22, 3, 2, 8).permute(2, 0, 3, 1, 4)
                q = attention.rope(attention.q_norm(raw[0]), captured['pos'])
                k = attention.rope(attention.k_norm(raw[1]), captured['pos'])
            stage = next(s for s in inspector.data['stages'] if s['id'] == 'global_1')
            def flatten(value):
                return torch.from_numpy(value).permute(1, 0, 2, 3).reshape(1, 2, 22, 8)
            torch.testing.assert_close(flatten(stage['q']), q, rtol=0, atol=0)
            torch.testing.assert_close(flatten(stage['k']), k, rtol=0, atol=0)
            weights = ((flatten(stage['q']) @ flatten(stage['k']).transpose(-2, -1)) / np.sqrt(8)).softmax(-1)
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
            patches = captured['output'].reshape(2, 11, 16)[:, 5:]
            normalized = F.normalize(patches, dim=-1)
            expected = (normalized[:, inspector.queries] @ normalized.flatten(0, 1).T).reshape(2, 3, 2, 6)
            torch.testing.assert_close(torch.from_numpy(stage['cosine']), expected)
            self.assertEqual(stage['features'].shape, (2, 6, 3))
        finally:
            handle.remove()
            block_handle.remove()

    def test_road_cross_attention_matches_returned_weights_with_camera_token(self):
        model = small_model(camera=True)
        with torch.no_grad(), ModelInspector(model) as inspector:
            output = model(self.images, debug=True)
        stage = inspector.data['stages'][-1]
        self.assertEqual(stage['query_kind'], 'ego')
        self.assertEqual(stage['prefix'], 1)
        q, k = torch.from_numpy(stage['q']), torch.from_numpy(stage['k'])
        weights = (q @ k.transpose(-2, -1) / np.sqrt(stage['head_dim'])).softmax(-1)
        torch.testing.assert_close(weights.mean(1).reshape(1, 2, 7), output['spatial_attention'])

    def test_invalid_selection_removes_hooks_and_serialization_keeps_float32(self):
        before = sum(len(module._forward_hooks) + len(module._forward_pre_hooks)
                     for module in self.model.modules())
        with self.assertRaises(ValueError):
            with ModelInspector(self.model, blocks=(99,)):
                pass
        after = sum(len(module._forward_hooks) + len(module._forward_pre_hooks)
                    for module in self.model.modules())
        self.assertEqual(before, after)
        array = np.array([1e-8, -1e-8, 1.0000001], dtype=np.float32)
        packed = pack_arrays(array)
        np.testing.assert_array_equal(np.frombuffer(base64.b64decode(packed['data']), dtype='<f4'), array)
        bf16 = np.array([1.0, -0.125, 1.0e-30], dtype=np.float32).view('<u4')
        bf16 = (bf16 & 0xffff0000).view('<f4')
        packed = pack_arrays(bf16)
        self.assertEqual(packed['dtype'], 'bf16')
        restored = np.frombuffer(base64.b64decode(packed['data']), dtype='<u2').astype('<u4') << 16
        np.testing.assert_array_equal(restored.view('<f4'), bf16)
        with self.assertRaises(ValueError):
            pack_arrays(np.array([np.nan]))
        with torch.no_grad(), ModelInspector(self.model) as inspector:
            result = self.model(self.images)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'inspect.html'
            inspector.write(path, self.images[0], {'sample_token': '</script>', 'target': 2},
                            list(map(str, range(5))), result['road_logits'][0].softmax(-1).numpy())
            page = path.read_text(encoding='utf-8')
            self.assertNotIn('__INSPECTION_DATA__', page)
            payload = page.split('<script id="inspection-data" type="application/json">')[1].split('</script>')[0]
            self.assertNotIn('</script>', payload)
            archived = json.loads(payload)
            data = json.loads(gzip.decompress(base64.b64decode(archived['payload'])))
            self.assertEqual(data['row']['sample_token'], '</script>')
            self.assertEqual(data['stages'][2]['q']['shape'], [2, 2, 11, 8])

    def test_epoch_component_reuses_debug_forward_and_links_one_fixed_probe(self):
        from scripts.road_training_visualizer import RoadTrainingVisualizer
        images = self.images[0]
        record = {'sample_token': 'probe', 'scene_token': 'scene', 'scene_name': 'scene',
                  'image_paths': ['a.jpg', 'b.jpg'], 'timestamps': [1, 2], 'annotation': {'label': 2}}
        with tempfile.TemporaryDirectory() as directory:
            vis = RoadTrainingVisualizer(directory, [record], dict(enumerate(map(str, range(5)))),
                                         model_internals=True)
            epoch = vis.output_dir / 'epoch_0005'
            epoch.mkdir()
            dataset = [{'images': images, 'sample_token': 'probe'}]
            row = {**record, 'target': 2, 'prediction': 2, 'probabilities': [0, 0, 1, 0, 0],
                   'confidence': 1.0, 'nll': 0.0}
            calls = []
            handle = self.model.register_forward_hook(lambda *args: calls.append(1))
            self.model.train()
            try:
                cases = vis._render_cases(['probe'], {'probe': row}, epoch, self.model,
                                          dataset, torch.device('cpu'), 'none', optimizer_steps=60)
            finally:
                handle.remove()
            self.assertEqual(len(calls), 1)
            self.assertTrue(self.model.training)
            page = next((epoch / 'cases').glob('*_internals.html'))
            data = json.loads(page.read_text().split('<script id="inspection-data" type="application/json">')[1]
                              .split('</script>')[0])
            data = json.loads(gzip.decompress(base64.b64decode(data['payload'])))
            self.assertEqual(data['provenance']['optimizer_steps'], 60)
            self.assertEqual(data['provenance']['epoch'], 5)
            self.assertEqual(data['row']['sample_token'], 'probe')
            summary = {'epoch': 5, 'class_support': [0, 0, 1, 0, 0],
                       'per_class_precision': [0, 0, 1, 0, 0], 'per_class_recall': [0, 0, 1, 0, 0],
                       'per_class_f1': [0, 0, 1, 0, 0], 'per_scene': [], 'accuracy': 1,
                       'nll': 0, 'macro_f1_present': 1, 'error_count': 0, 'samples': 1,
                       'high_confidence_errors': 0, 'majority_baseline': 1}
            vis._epoch_report(epoch, summary, [], cases)
            self.assertIn(page.name, (epoch / 'index.html').read_text())


if __name__ == '__main__':
    unittest.main()
