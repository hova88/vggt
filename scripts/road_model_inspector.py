"""Bounded, read-only probes of real VGGT internals and an offline HTML viewer.

Hooks observe one eval forward. Attention is reconstructed from effective Q/K
(including QK normalization and RoPE); the model keeps its original SDPA path.
Only selected heads/channels and at most 64 patch-query cosine rows are saved.
"""
import argparse
import base64
import copy
import gzip
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def _array(tensor):
    return tensor.detach().float().cpu().numpy()


def pack_arrays(value):
    """Embed float32 arrays compactly, without rounding tiny attention signals."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value, dtype='<f4')
        if not np.isfinite(array).all():
            raise ValueError('Non-finite model inspection tensor')
        bits = array.view('<u4')
        bf16 = bool(np.all((bits & 0xffff) == 0))
        stored = (bits >> 16).astype('<u2') if bf16 else array
        return {'shape': list(array.shape), 'dtype': 'bf16' if bf16 else 'float32',
                'data': base64.b64encode(stored.tobytes()).decode('ascii')}
    if isinstance(value, dict):
        return {key: pack_arrays(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack_arrays(item) for item in value]
    return value


class ModelInspector:
    def __init__(self, model, blocks=(-1,), heads=(0, 1), channels=32, max_queries=64):
        if not blocks or not heads or min(heads) < 0 or channels < 1 or max_queries < 1:
            raise ValueError('Invalid inspection selection')
        self.model, self.blocks, self.heads = model, tuple(blocks), tuple(dict.fromkeys(heads))
        self.channels, self.max_queries = channels, max_queries
        self.handles, self.data, self.pending = [], {'stages': []}, {}

    def _hook(self, module, callback, pre=False, **kwargs):
        register = module.register_forward_pre_hook if pre else module.register_forward_hook
        self.handles.append(register(callback, **kwargs))

    def __enter__(self):
        if self.model.training:
            raise RuntimeError('Model inspection requires model.eval()')
        try:
            self._hook(self.model.aggregator, self._begin, pre=True)
            backbone = self.model.aggregator.patch_embed
            embed = backbone.patch_embed if hasattr(backbone, 'blocks') else backbone
            self._hook(embed.proj, self._conv_input, pre=True)
            self._hook(embed.proj, self._conv_output)
            self._hook(embed, self._patch_embedding)
            if hasattr(backbone, 'blocks'):
                if getattr(backbone, 'chunked_blocks', False):
                    raise ValueError('Inspection requires unchunked DINO blocks')
                self.dino_prefix = 1 + backbone.num_register_tokens
                self._hook(backbone.blocks[0], self._prepared_embedding, pre=True)
                self._block(backbone.blocks[-1], 'dino', f'DINOv2 block {len(backbone.blocks)-1}',
                            'frame', self.dino_prefix)
            aggregator = self.model.aggregator
            indices = list(dict.fromkeys(i if i >= 0 else aggregator.depth + i for i in self.blocks))
            if any(i < 0 or i >= aggregator.depth for i in indices):
                raise ValueError('Inspection block index outside aggregator depth')
            for index in indices:
                for scope in ('frame', 'global'):
                    self._block(getattr(aggregator, f'{scope}_blocks')[index], f'{scope}_{index}',
                                f'VGGT {scope} block {index}', scope, aggregator.patch_start_idx)
            self._road_attention()
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()

    def _begin(self, module, args):
        import torch
        images = args[0]
        if torch.is_grad_enabled() or images.ndim != 5 or images.shape[0] != 1:
            raise ValueError('Inspection requires no_grad and exactly one sequence')
        if 'grid' in self.data:
            raise ValueError('Create a new inspector for each forward')
        self.frames = images.shape[1]
        self.grid = tuple(size // module.patch_size for size in images.shape[-2:])
        if any(size % module.patch_size for size in images.shape[-2:]):
            raise ValueError('Model input is not patch aligned')
        patches = self.grid[0] * self.grid[1]
        self.queries = np.linspace(0, patches - 1, min(patches, self.max_queries), dtype=int)
        self.data.update({'grid': list(self.grid), 'patch_size': module.patch_size,
                          'query_patches': self.queries.tolist(), 'frames': self.frames})

    def _indices(self, width):
        return np.linspace(0, width - 1, min(width, self.channels), dtype=int).tolist()

    def _conv_input(self, module, args):
        x = args[0].detach()
        ph, pw = module.kernel_size
        if module.stride != (ph, pw) or module.padding != (0, 0) or module.dilation != (1, 1):
            raise ValueError('Inspection requires non-overlapping patch convolution')
        patches = x.unfold(2, ph, ph).unfold(3, pw, pw).permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(self.frames, -1, 3, ph, pw)
        ids = self._indices(module.out_channels)
        self.data['conv'] = {'channels': ids, 'kernel': _array(module.weight[ids]),
                             'bias': (_array(module.bias[ids]) if module.bias is not None
                                      else np.zeros(len(ids), dtype=np.float32)),
                             'input_patches': _array(patches[:, self.queries])}

    def _conv_output(self, module, args, output):
        ids = self.data['conv']['channels']
        self.data['conv']['output'] = _array(output[:, ids].flatten(2).transpose(1, 2))

    def _patch_embedding(self, module, args, output):
        self.data['embedding'] = {'channels': self.data['conv']['channels'],
                                  'projected': _array(output[:, :, self.data['conv']['channels']])}

    def _prepared_embedding(self, module, args):
        embedding = self.data['embedding']
        prepared = _array(args[0][:, self.dino_prefix:, embedding['channels']])
        embedding.update({'prepared': prepared, 'position_contribution': prepared - embedding['projected']})

    def _canonical(self, x, scope):
        if scope == 'global':
            if x.shape[0] != 1 or x.shape[1] % self.frames:
                raise ValueError('Unexpected global token layout')
            return x.reshape(self.frames, -1, x.shape[-1])
        if x.shape[0] != self.frames:
            raise ValueError('Unexpected frame token layout')
        return x

    def _features(self, stage, x):
        import torch.nn.functional as functional
        patches = self._canonical(x.detach(), stage['scope'])[:, stage['prefix']:]
        if patches.shape[1] != self.grid[0] * self.grid[1]:
            raise ValueError('Inspection patch layout differs from image grid')
        ids = self._indices(patches.shape[-1])
        stage.update({'channels': ids, 'feature_dim': patches.shape[-1],
                      'features': _array(patches[:, :, ids])})
        # Similarity uses every feature dimension, never just the displayed channels.
        normalized = functional.normalize(patches.float(), dim=-1)
        queries = normalized[:, self.queries]
        if stage['scope'] == 'global':
            cosine = queries @ normalized.flatten(0, 1).T
            cosine = cosine.reshape(self.frames, len(self.queries), self.frames, -1)
        else:
            cosine = (queries @ normalized.transpose(-2, -1)).unsqueeze(2)
        stage['cosine'] = _array(cosine)

    def _block(self, block, key, label, scope, prefix):
        attention = block.attn
        heads = [head for head in self.heads if head < attention.num_heads]
        if not heads:
            raise ValueError(f'No selected heads exist in {label}')
        stage = {'id': key, 'label': label, 'scope': scope, 'prefix': prefix, 'heads': heads,
                 'query_kind': 'patch', 'feature_source': 'transformer block output',
                 'qk_transform': 'QK normalization + RoPE' if attention.rope is not None
                                 else 'QK normalization' if attention.q_norm.__class__.__name__ != 'Identity'
                                 else 'linear projection'}
        self.data['stages'].append(stage)

        def pre(module, args, kwargs):
            self.pending[key] = kwargs.get('pos', args[1] if len(args) > 1 else None)

        def qkv(module, args, output):
            batch, count, width = output.shape
            dim = width // (3 * attention.num_heads)
            raw = output.detach().reshape(batch, count, 3, attention.num_heads, dim)
            raw = raw.permute(2, 0, 3, 1, 4)
            q, k = attention.q_norm(raw[0][:, heads]), attention.k_norm(raw[1][:, heads])
            if attention.rope is not None:
                q = attention.rope(q, self.pending[key])
                k = attention.rope(k, self.pending[key])
            def canonical(value):
                if scope == 'global':
                    value = value.reshape(1, len(heads), self.frames, -1, dim)[0].permute(1, 0, 2, 3)
                return _array(value)
            stage.update({'q': canonical(q), 'k': canonical(k), 'head_dim': dim})

        self._hook(attention, pre, pre=True, with_kwargs=True)
        self._hook(attention.qkv, qkv)
        self._hook(block, lambda module, args, output: self._features(stage, output))
        if hasattr(block.mlp, 'act') and block.mlp.act.__class__.__name__ == 'GELU':
            def gelu(module, args, output):
                before = self._canonical(args[0].detach(), scope)[:, prefix:]
                after = self._canonical(output.detach(), scope)[:, prefix:]
                ids = self._indices(before.shape[-1])
                stage['gelu'] = {'channels': ids, 'input': _array(before[:, :, ids]),
                                 'output': _array(after[:, :, ids]), 'approximate': module.approximate}
            self._hook(block.mlp.act, gelu)

    def _road_attention(self):
        import torch.nn.functional as functional
        module = self.model.road_head.spatial_attention
        heads = [head for head in self.heads if head < module.num_heads]
        if not heads:
            raise ValueError('No selected heads exist in road spatial attention')
        stage = {'id': 'road_spatial', 'label': 'Road head · spatial cross attention', 'scope': 'frame',
                 'prefix': int(self.model.road_head.use_camera_token), 'heads': heads,
                 'query_kind': 'ego', 'feature_source': 'projected patch input to road spatial attention',
                 'qk_transform': 'linear projection'}
        self.data['stages'].append(stage)

        def capture(module, args):
            query, key = args[:2]
            dim = module.embed_dim // module.num_heads
            weights = module.in_proj_weight.chunk(3)
            biases = module.in_proj_bias.chunk(3) if module.in_proj_bias is not None else (None,) * 3
            q = functional.linear(query, weights[0], biases[0])
            k = functional.linear(key, weights[1], biases[1])
            def split(value):
                return _array(value.reshape(self.frames, -1, module.num_heads, dim).transpose(1, 2)[:, heads])
            stage.update({'q': split(q), 'k': split(k), 'head_dim': dim})
            self._features(stage, key)
        self._hook(module, capture, pre=True)

    def write(self, path, images, row, names, probabilities, provenance=None):
        from PIL import Image
        data = copy.copy(self.data)
        for stage in data['stages']:
            if not all(key in stage for key in ('q', 'k', 'features', 'cosine')):
                raise RuntimeError(f'Inspection hook did not run: {stage["id"]}')
        arrays = _array(images) if hasattr(images, 'detach') else np.asarray(images)
        data['images'] = []
        for frame in arrays:
            buffer = io.BytesIO()
            Image.fromarray(np.uint8(np.clip(frame.transpose(1, 2, 0), 0, 1) * 255)).save(buffer, format='JPEG', quality=90)
            data['images'].append('data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii'))
        data.update({'row': row, 'names': list(names), 'probabilities': np.asarray(probabilities).tolist(),
                     'provenance': provenance or {}, 'created_at': datetime.now().astimezone().isoformat()})
        raw = json.dumps(pack_arrays(data), ensure_ascii=False, allow_nan=False).encode('utf-8')
        payload = json.dumps({'encoding': 'gzip-base64', 'payload': base64.b64encode(
            gzip.compress(raw, compresslevel=5, mtime=0)).decode('ascii')})
        template = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.replace('__INSPECTION_DATA__', payload), encoding='utf-8')
        return {'path': str(path), 'bytes': path.stat().st_size, 'stages': len(data['stages'])}


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from scripts.road_common import config, data_splits, loader, make_model, place_model, autocast_context
    from vggt.heads.road_probability_head import ROAD_CLASSES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Trained road best.pt, not VGGT pretraining weights')
    parser.add_argument('--pretrained', help='Override local VGGT weights path recorded in the training config')
    parser.add_argument('--config', help='Fallback config for checkpoints without a config snapshot')
    parser.add_argument('--predictions', help='Saved predictions.jsonl: reuse original targets and image paths')
    parser.add_argument('--output-dir', default='runs/model_inspection')
    parser.add_argument('--sample-token', action='append', help='Repeat to inspect several samples; default: one sample')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--blocks', default='-1', help='Comma-separated VGGT block indices, zero based; -1 is final block')
    parser.add_argument('--heads', default='0,1', help='Comma-separated heads to capture; viewer selects among these')
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    initial_stat = checkpoint.stat()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    cfg = copy.deepcopy(state.get('config') or (config(args.config) if args.config else None))
    if cfg is None:
        parser.error('Checkpoint has no config snapshot; provide --config')
    if args.pretrained:
        cfg['model']['checkpoint'] = str(Path(args.pretrained).resolve(strict=True))
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    elif torch.cuda.is_available():
        parser.error('For CPU inspection on a GPU host, set CUDA_VISIBLE_DEVICES="" before launching')
    if args.predictions:
        rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding='utf-8').splitlines()
                if line.strip()]
        records = [{**row, 'annotation': {'soft_label': row['target_probabilities']}} for row in rows]
        prior = None
        target_source = str(Path(args.predictions).resolve())
    else:
        train, val, prior, _ = data_splits(cfg)
        records = train + val
        target_source = f'Live annotations: {cfg["data"]["annotation"]}; may differ from training-time labels'
    if not records:
        parser.error('No inspection records')
    by_token = {record['sample_token']: i for i, record in enumerate(records)}
    if len(by_token) != len(records):
        parser.error('Duplicate inspection sample tokens')
    default_record = next((row for row in records if len(set(row['image_paths'])) == len(row['image_paths'])),
                          records[0])
    selected = args.sample_token or [default_record['sample_token']]
    if any(token not in by_token for token in selected):
        parser.error('Selected sample_token is absent from checkpoint-config dataset')
    model = place_model(make_model(cfg, prior, checkpoint=checkpoint), device, cfg).eval()
    current_stat = checkpoint.stat()
    if (initial_stat.st_ino, initial_stat.st_mtime_ns, initial_stat.st_size) != (
            current_stat.st_ino, current_stat.st_mtime_ns, current_stat.st_size):
        raise RuntimeError('Checkpoint changed while loading; rerun against a stable checkpoint')
    dataset = loader(records, cfg, num_workers=0).dataset
    output = Path(args.output_dir) / datetime.now().strftime('inspection_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    links = []
    for token in selected:
        item = dataset[by_token[token]]
        inspector = ModelInspector(model, blocks=tuple(map(int, args.blocks.split(','))),
                                   heads=tuple(map(int, args.heads.split(','))))
        with inspector, torch.no_grad(), autocast_context(device, cfg['training']['amp']):
            result = model(item['images'].unsqueeze(0).to(device), debug=True)
        import hashlib
        filename = hashlib.sha256(token.encode()).hexdigest()[:16] + '.html'
        row = {key: records[by_token[token]][key] for key in ('sample_token', 'scene_token', 'image_paths')}
        target = item['target']
        row['target'] = int(target.argmax() if target.ndim else target)
        stats = inspector.write(output / filename, item['images'], row, list(ROAD_CLASSES.values()),
                                _array(result['road_logits'][0].float().softmax(-1)),
                                {'checkpoint': str(checkpoint), 'checkpoint_epoch': state.get('epoch'),
                                 'optimizer_steps': state.get('global_step'), 'target_source': target_source,
                                 'source': 'standalone checkpoint inference'})
        print(f'Inspection: {stats}', flush=True)
        links.append((filename, token))
    import html
    body = '<h1>VGGT model inspection</h1>' + ''.join(
        f'<p><a href="{filename}">{html.escape(token)}</a></p>' for filename, token in links)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">' + body, encoding='utf-8')
    print(f'Open: {output / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
