from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


class Attention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        n_heads: int, 
        n_kv_heads: int, 
        head_dim: int,
        rope_traditional: bool = True
    ):
        super().__init__()

        self.n_heads: int = n_heads
        self.n_kv_heads: int = n_kv_heads

        self.repeats = n_heads // n_kv_heads

        self.scale = head_dim**-0.5
        
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = nn.RoPE(dim // n_heads, traditional=rope_traditional)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:

        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        n_heads: int, 
        n_kv_heads: int, 
        head_dim: int, 
        hidden_dim: int,
        norm_eps: float,
        rope_traditional: bool = True
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope_traditional=rope_traditional
        )
        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        rope_traditional: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        assert self.vocab_size > 0
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = [
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                hidden_dim=hidden_dim,
                norm_eps=norm_eps,
                rope_traditional=rope_traditional
            )  for _ in range(n_layers)
        ]
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.tok_embeddings(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.output(self.norm(h)), cache
    
    
def mistral_7B_instruct_v01() -> Transformer:
    return Transformer(
        dim=4096,
        hidden_dim=14336,
        vocab_size=32000,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        norm_eps=1e-5
    )
    
def mistral_7B_instruct_v02() -> Transformer:
    return Transformer(
        dim=4096,
        hidden_dim=14336,
        vocab_size=32000,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        norm_eps=1e-5
    )

def openhermes_25_mistral_7B() -> Transformer:
    return Transformer(
        dim=4096,
        hidden_dim=14336,
        vocab_size=32002,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        norm_eps=1e-5
    )
      
def llama_2_7B_chat() -> Transformer:
    return Transformer(
        dim=4096,
        hidden_dim=11008,
        vocab_size=32000,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        head_dim=128,
        norm_eps=1e-5
    )
    
def tiny_llama_chat_v06() -> Transformer:
    return Transformer(
        dim=2048,
        hidden_dim=5632,
        n_heads=32,
        n_kv_heads=4,
        n_layers=22,
        vocab_size=32000,
        head_dim=64, # 2048 / 32,
        norm_eps=1e-5,
        rope_traditional=False
    )