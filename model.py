"""Inference-only traffic encoder used by the released checkpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Architecture:
    traffic_hidden_dim: int = 640
    traffic_heads: int = 10
    traffic_layers: int = 8
    traffic_mlp_ratio: float = 4.0
    traffic_dropout: float = 0.1
    num_queries: int = 16
    stem_channels: int = 160
    text_dim: int = 2560
    logit_scale_init: float = 2.659
    logit_bias_init: float = -3.0


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    positions = torch.arange(end, device=freqs.device)
    return torch.polar(torch.ones((end, freqs.numel())), torch.outer(positions, freqs))


def apply_rotary_emb(
    query: torch.Tensor, key: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    query_complex = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    key_complex = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    query_out = torch.view_as_real(query_complex * freqs_cis).flatten(3)
    key_out = torch.view_as_real(key_complex * freqs_cis).flatten(3)
    return query_out.type_as(query), key_out.type_as(key)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.float().type_as(value) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class MultiScaleTrafficStem(nn.Module):
    def __init__(self, hidden_dim: int, stem_channels: int):
        super().__init__()
        self.dir_stem = nn.Sequential(
            nn.Conv1d(2, stem_channels, 4, stride=4, bias=False),
            nn.GroupNorm(8, stem_channels),
            nn.GELU(),
            nn.Conv1d(stem_channels, stem_channels * 2, 2, stride=2, bias=False),
            nn.GroupNorm(8, stem_channels * 2),
            nn.GELU(),
            nn.Conv1d(stem_channels * 2, hidden_dim, 2, stride=2, bias=False),
        )
        self.time_stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, 4, stride=4, bias=False),
            nn.GroupNorm(8, stem_channels),
            nn.GELU(),
            nn.Conv1d(stem_channels, stem_channels * 2, 2, stride=2, bias=False),
            nn.GroupNorm(8, stem_channels * 2),
            nn.GELU(),
            nn.Conv1d(stem_channels * 2, hidden_dim, 2, stride=2, bias=False),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        packet_values = value[:, :, 0]
        time_values = value[:, :, 1:2].transpose(1, 2).contiguous()
        direction = torch.cat(
            [F.relu(packet_values).unsqueeze(1), F.relu(-packet_values).unsqueeze(1)],
            dim=1,
        )
        features = self.fusion(
            torch.cat([self.dir_stem(direction), self.time_stem(time_values)], dim=1)
        )
        return features.transpose(1, 2).contiguous()


class TrafficTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.n_heads = heads
        self.head_dim = dim // heads
        self.norm1 = RMSNorm(dim)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout_p = dropout
        self.norm2 = RMSNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio, dropout)
        self.freqs_cis: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, channels = value.shape
        qkv = self.qkv_proj(self.norm1(value)).view(
            batch, sequence_length, 3, self.n_heads, self.head_dim
        )
        query, key, item = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if (
            self.freqs_cis is None
            or self.freqs_cis.shape[0] < sequence_length
            or self.freqs_cis.device != value.device
        ):
            self.freqs_cis = precompute_freqs_cis(self.head_dim, sequence_length * 2).to(
                value.device
            )
        query, key = apply_rotary_emb(query, key, self.freqs_cis[:sequence_length])
        query, key, item = (tensor.transpose(1, 2) for tensor in (query, key, item))
        attention = F.scaled_dot_product_attention(
            query,
            key,
            item,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attention = self.out_proj(attention.transpose(1, 2).reshape(batch, sequence_length, channels))
        value = value + attention
        return value + self.ffn(self.norm2(value))


class QueryAttentionPooler(nn.Module):
    def __init__(self, dim: int, heads: int, num_queries: int, dropout: float):
        super().__init__()
        self.num_queries = num_queries
        self.heads = heads
        self.head_dim = dim // heads
        self.query_tokens = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.query_norm = RMSNorm(dim)
        self.memory_norm = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, 2.0, dropout)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        batch = memory.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(batch, -1, -1)
        query = self.q_proj(self.query_norm(queries))
        key = self.k_proj(self.memory_norm(memory))
        value = self.v_proj(memory)
        query = query.view(batch, self.num_queries, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
        attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = self.dropout(torch.softmax(attention, dim=-1))
        pooled = torch.matmul(attention, value).transpose(1, 2).reshape(batch, self.num_queries, -1)
        queries = self.norm(queries + self.out_proj(pooled))
        return queries + self.ffn(self.ffn_norm(queries))


class LightweightTrafficEncoder(nn.Module):
    def __init__(self, config: Architecture):
        super().__init__()
        self.stem = MultiScaleTrafficStem(config.traffic_hidden_dim, config.stem_channels)
        self.layers = nn.ModuleList(
            TrafficTransformerLayer(
                config.traffic_hidden_dim,
                config.traffic_heads,
                config.traffic_mlp_ratio,
                config.traffic_dropout,
            )
            for _ in range(config.traffic_layers)
        )
        self.pooler = QueryAttentionPooler(
            config.traffic_hidden_dim,
            config.traffic_heads,
            config.num_queries,
            config.traffic_dropout,
        )
        self.final_norm = RMSNorm(config.traffic_hidden_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.stem(value)
        for layer in self.layers:
            value = layer(value)
        return self.final_norm(self.pooler(value))


class TrafficRetriever(nn.Module):
    def __init__(self, config: Architecture):
        super().__init__()
        self.traffic_encoder = LightweightTrafficEncoder(config)
        self.traffic_proj = nn.Sequential(
            nn.Linear(config.traffic_hidden_dim, config.traffic_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.traffic_dropout),
            nn.Linear(config.traffic_hidden_dim * 2, config.text_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * config.logit_scale_init)
        self.logit_bias = nn.Parameter(torch.tensor(config.logit_bias_init, dtype=torch.float32))

    def forward(self, traffic: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        traffic_embeddings = F.normalize(self.traffic_proj(self.traffic_encoder(traffic)), dim=-1)
        candidates = F.normalize(candidates, dim=-1)
        scale = self.logit_scale.clamp(max=4.6052).exp()
        multi_logits = scale * torch.einsum("bqd,nd->bqn", traffic_embeddings, candidates)
        return multi_logits.max(dim=1).values + self.logit_bias

