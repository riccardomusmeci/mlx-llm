import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class LayerNorm(nn.LayerNorm):
    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)


class RoPEAttention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        n_heads: int, 
        rotary_dim: int
    ):
        super().__init__()

        self.n_heads = n_heads

        self.rope = nn.RoPE(rotary_dim, traditional=False)
        self.Wqkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x, mask=None, cache=None):
        qkv = self.Wqkv(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        # Extract some shapes
        n_heads = self.n_heads
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = queries.astype(mx.float32)
        keys = keys.astype(mx.float32)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores, axis=-1).astype(values.dtype)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat), (keys, values)


class ParallelBlock(nn.Module):
    def __init__(
        self, 
        dim: int,
        n_heads: int,
        rotary_dim: int
    ):
        super().__init__()
        
        hidden_dim = dim * 4
        self.mixer = RoPEAttention(dim, n_heads, rotary_dim)
        self.ln = LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU(approx="precise")

    def __call__(self, x, mask, cache):
        h = self.ln(x)
        attn_h, cache = self.mixer(h, mask, cache)
        ff_h = self.fc2(self.act(self.fc1(h)))
        return attn_h + ff_h + x, cache


class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        dim: int,
        n_layers: int, # num_layers
        n_heads: int, # num_heads
        rotary_dim: int # rotary_dim
    ):
        super().__init__()
        self.h = [ParallelBlock(dim, n_heads, rotary_dim) for i in range(n_layers)]

    def __call__(self, x, mask, cache):
        if cache is None:
            cache = [None] * len(self.h)

        for e, layer in enumerate(self.h):
            x, cache[e] = layer(x, mask, cache[e])
        return x, cache


class OutputHead(nn.Module):
    def __init__(self, dim: int, vocab_size: int) -> None:
        self.ln = LayerNorm(dim)
        self.linear = nn.Linear(dim, vocab_size)

    def __call__(self, inputs):
        return self.linear(self.ln(inputs))
    

class Phi2(nn.Module):
    def __init__(
        self, 
        dim: int, # model_dim
        vocab_size: int, # num_vocab
        n_heads: int, # num_heads
        n_layers: int, # num_layers
        rotary_dim: int # rotary_dim
    ):
        
        self.wte = nn.Embedding(vocab_size, dim)
        self.transformer = TransformerDecoder(dim, n_layers, n_heads, rotary_dim)
        self.lm_head = OutputHead(dim, vocab_size)

    def embed(self, inputs: mx.array) -> mx.array:
        
        x = self.wte(inputs)

        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(x.dtype)

        y, cache = self.transformer(x, mask, cache)
        
        return y
    
    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache: mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        x = self.wte(inputs)

        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(x.dtype)

        y, cache = self.transformer(x, mask, cache)
        return self.lm_head(y), cache


def phi2() -> Phi2:
    return Phi2(
        dim=2560,
        vocab_size=51200,
        n_heads=32,
        n_layers=32,
        rotary_dim=32
    )