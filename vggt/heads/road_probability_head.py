"""Five-state road posterior from VGGT's last cached aggregator output."""
import math
import torch
from torch import nn
import torch.nn.functional as F

ROAD_CLASSES = {0: 'elevated_up', 1: 'elevated_down', 2: 'main_road', 3: 'side_road', 4: 'intersection'}


class RoadProbabilityHead(nn.Module):
    def __init__(self, feature_dim, road_dim=256, spatial_num_heads=8,
                 temporal_num_heads=8, temporal_layers=2, temporal_ffn_dim=1024,
                 dropout=0.1, num_classes=5, use_camera_token=False, max_frames=64):
        super().__init__()
        self.use_camera_token = use_camera_token
        self.token_projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, road_dim))
        self.spatial_ego_query = nn.Parameter(torch.randn(1, 1, road_dim) * 0.02)
        self.spatial_attention = nn.MultiheadAttention(road_dim, spatial_num_heads, dropout=dropout, batch_first=True)
        self.temporal_position = nn.Parameter(torch.randn(1, max_frames, road_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(road_dim, temporal_num_heads, temporal_ffn_dim, dropout,
                                           activation='gelu', batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(layer, temporal_layers)
        self.temporal_road_query = nn.Parameter(torch.randn(1, 1, road_dim) * 0.02)
        self.temporal_attention = nn.MultiheadAttention(road_dim, temporal_num_heads, dropout=dropout, batch_first=True)
        self.refine_in = nn.LayerNorm(road_dim)
        self.refine_mlp = nn.Sequential(nn.Linear(road_dim, 2 * road_dim), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(2 * road_dim, road_dim))
        self.refine_out = nn.LayerNorm(road_dim)
        self.classifier = nn.Sequential(nn.Linear(road_dim, road_dim // 2), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(road_dim // 2, num_classes))

    def forward(self, tokens, patch_start_idx, debug=False):
        b, t, p, d = tokens.shape
        patch_tokens = tokens[:, :, patch_start_idx:, :]
        selected = torch.cat((tokens[:, :, :1, :], patch_tokens), dim=2) if self.use_camera_token else patch_tokens
        projected = self.token_projection(selected)
        flat = projected.reshape(b * t, -1, projected.shape[-1])
        spatial, spatial_weights = self.spatial_attention(self.spatial_ego_query.expand(b*t, -1, -1),
                                                           flat, flat, need_weights=debug)
        frame_features = spatial.reshape(b, t, -1)
        if t > self.temporal_position.shape[1]:
            raise ValueError('num_frames exceeds max_frames')
        temporal_features = self.temporal_encoder(frame_features + self.temporal_position[:, :t])
        pooled, temporal_weights = self.temporal_attention(self.temporal_road_query.expand(b, -1, -1),
                                                            temporal_features, temporal_features, need_weights=debug)
        road_feat = pooled[:, 0]
        embedding = self.refine_out(road_feat + self.refine_mlp(self.refine_in(road_feat)))
        logits = self.classifier(embedding)
        out = {'road_logits': logits, 'road_embedding': embedding}
        if debug:
            out.update({'spatial_attention': spatial_weights.reshape(b, t, -1),
                        'temporal_attention': temporal_weights[:, 0],
                        'patch_tokens_shape': tuple(patch_tokens.shape),
                        'frame_road_features_shape': tuple(frame_features.shape),
                        'temporal_features_shape': tuple(temporal_features.shape)})
        return out


class VGGTRoadClassifier(nn.Module):
    def __init__(self, aggregator, head_config, finetune_mode='head_only', last_blocks=2,
                 temperature=1.0, class_prior=None, use_prior_correction=False):
        super().__init__()
        self.aggregator = aggregator
        # Last cached tensor concatenates frame/global features. Infer from actual module configuration.
        feature_dim = aggregator.frame_blocks[-1].norm1.normalized_shape[0] * 2
        self.road_head = RoadProbabilityHead(feature_dim, **head_config)
        self.temperature = float(temperature)
        self.use_prior_correction = use_prior_correction
        prior = torch.ones(5) / 5 if class_prior is None else torch.as_tensor(class_prior, dtype=torch.float32)
        self.register_buffer('class_prior', prior, persistent=False)
        self.finetune_mode = finetune_mode
        for p in self.aggregator.parameters():
            p.requires_grad_(False)
        if finetune_mode == 'last_blocks':
            if not 1 <= last_blocks <= len(self.aggregator.frame_blocks):
                raise ValueError('last_blocks must be between 1 and aggregator depth')
            for blocks in (self.aggregator.frame_blocks, self.aggregator.global_blocks):
                for block in blocks[-last_blocks:]:
                    for p in block.parameters():
                        p.requires_grad_(True)
        elif finetune_mode != 'head_only':
            raise ValueError(f'Unknown finetune_mode: {finetune_mode}')

    def train(self, mode=True):
        super().train(mode)
        if self.finetune_mode == 'head_only':
            self.aggregator.eval()
        return self

    def forward(self, images, debug=False):
        if self.finetune_mode == 'head_only':
            with torch.no_grad():
                cached, patch_start_idx = self.aggregator(images)
        else:
            cached, patch_start_idx = self.aggregator(images)
        tokens = cached[-1]
        if tokens is None:
            raise RuntimeError('Aggregator final cached output is None')
        feature_dim = tokens.shape[-1]
        if feature_dim != self.road_head.token_projection[0].normalized_shape[0]:
            raise RuntimeError(f'Runtime VGGT feature dim {feature_dim} differs from road head input dim')
        out = self.road_head(tokens, patch_start_idx, debug)
        logits = out['road_logits']
        prob = F.softmax(logits.float() / self.temperature, dim=-1)
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(-1)
        out.update({'road_prob': prob, 'entropy': entropy,
                    'normalized_entropy': entropy / math.log(prob.shape[-1]),
                    'entropy_confidence': 1 - entropy / math.log(prob.shape[-1]),
                    'class_prior': self.class_prior.to(prob.device)})
        if self.use_prior_correction:
            score = prob / self.class_prior.to(prob.device).clamp_min(1e-6)
            out['hmm_observation'] = score / score.sum(-1, keepdim=True)
        if debug:
            out.update({'images_shape': tuple(images.shape), 'aggregated_tokens_shape': tuple(tokens.shape),
                        'patch_start_idx': patch_start_idx})
        return out
