from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Attention(nn.Module):
    """Attention module

    Args:
        dim (int): model dimension
        n_heads (int): number of attention heads
        n_kv_heads (int): number of key-value heads
        head_dim (Optional[int]): head dimension. If None, it is set to dim // n_heads. Defaults to None.
        rope_traditional (bool, optional): whether to use traditional RoPE. Defaults to False.
        rope_theta (float, optional): RoPE theta. Defaults to 1000.
        rope_scaling (Optional[Dict[str, Union[float, str]]], optional): RoPE scaling. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: Optional[int] = None,
        rope_traditional: bool = False,
        rope_theta: float = 1000,
        rope_scaling: Optional[Dict[str, Union[float, str]]] = None,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        head_dim = dim // n_heads if head_dim is None else head_dim

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        rope_scale = 1 / rope_scaling["factor"] if rope_scaling is not None and rope_scaling["type"] == "linear" else 1  # type: ignore

        self.rope = nn.RoPE(
            dims=head_dim,
            traditional=rope_traditional,
            base=rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        kv_cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """Forward pass

        Args:
            x (mx.array): input tokens
            mask (Optional[mx.array], optional): attention mask. Defaults to None.
            kv_cache (Optional[Tuple[mx.array, mx.array]], optional): key-value cache. Defaults to None.

        Returns:
            Tuple[mx.array, Tuple[mx.array, mx.array]]: output and kv-cache
        """
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            key_cache, value_cache = kv_cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)  # type: ignore
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)


class MLP(nn.Module):
    """MLP module

    Args:
        dim (int): model dimension
        hidden_dim (int): hidden dimension
        gemma (bool, optional): use Gemma activation. Defaults to False.
    """

    def __init__(self, dim: int, hidden_dim: int, gemma: bool = False):
        super().__init__()

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.gemma = gemma

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass

        Args:
            x (mx.array): input

        Returns:
            mx.array: output
        """
        if self.gemma:
            return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))
        else:
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Transformer block

    Args:
        dim (int): model dimension
        n_heads (int): number of attention heads
        n_kv_heads (int): number of key-value heads
        hidden_dim (int): hidden dimension
        norm_eps (float): normalization epsilon
        head_dim (Optional[int], optional): head dimension. Defaults to None.
        rope_traditional (bool, optional): whether to use traditional RoPE. Defaults to False.
        rope_theta (float, optional): RoPE theta. Defaults to 1000.
        rope_scaling (Optional[Dict[str, Union[float, str]]], optional): RoPE scaling. Defaults to None.
        gemma (bool, optional): whether using Gemma layers. Defaults to False.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        hidden_dim: int,
        norm_eps: float,
        head_dim: Optional[int] = None,
        rope_traditional: bool = False,
        rope_theta: float = 1000,
        rope_scaling: Optional[Dict[str, Union[float, str]]] = None,
        gemma: bool = False,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope_traditional=rope_traditional,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.mlp = MLP(dim=dim, hidden_dim=hidden_dim, gemma=gemma)

        if not gemma:
            self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
            self.mlp_norm = nn.RMSNorm(dim, eps=norm_eps)
        else:
            self.attention_norm = RMSNorm(dim, eps=norm_eps)
            self.mlp_norm = RMSNorm(dim, eps=norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        kv_cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """Forward pass

        Args:
            x (mx.array): input tokens
            mask (Optional[mx.array], optional): attention mask. Defaults to None.
            kv_cache (Optional[Tuple[mx.array, mx.array]], optional): key-value cache. Defaults to None.

        Returns:
            Tuple[mx.array, Tuple[mx.array, mx.array]]: output and key-value cache
        """
        r, kv_cache = self.attention(x=self.attention_norm(x), mask=mask, kv_cache=kv_cache)
        h = x + r
        r = self.mlp(self.mlp_norm(h))
        out = h + r
        return out, kv_cache


class Transformer(nn.Module):
    """Transformer model

    Args:
        dim (int): model dimension
        hidden_dim (int): hidden dimension
        vocab_size (int): vocabulary size
        n_layers (int): number of transformer blocks
        n_heads (int): number of attention heads
        n_kv_heads (Optional[int]): number of key-value heads. If None, it is set to num_heads. Defaults to None.
        head_dim (Optional[int], optional): head dimension. Defaults to None.
        norm_eps (float): normalization epsilon. Defaults to 1e-5.
        rope_traditional (bool, optional): whether to use traditional RoPE. Defaults to False.
        rope_theta (float, optional): RoPE theta. Defaults to 1000.
        rope_scaling (Optional[Dict[str, Union[float, str]]], optional): RoPE scaling. Defaults to None.
        gemma (bool, optional): whether to use Gemma Transformer. Defaults to False.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        norm_eps: float = 1e-5,
        rope_traditional: bool = False,
        rope_theta: float = 1000,
        rope_scaling: Optional[Dict[str, Union[float, str]]] = None,
        gemma: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.gemma = gemma
        assert self.vocab_size > 0
        if n_kv_heads is None:
            n_kv_heads = n_heads
        self.token_embed = nn.Embedding(vocab_size, self.dim)
        self.layers = [
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                hidden_dim=hidden_dim,
                norm_eps=norm_eps,
                head_dim=head_dim,
                rope_traditional=rope_traditional,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                gemma=self.gemma,
            )
            for _ in range(n_layers)
        ]
        if not self.gemma:
            self.norm = nn.RMSNorm(dim, eps=norm_eps)
            self.head = nn.Linear(dim, vocab_size, bias=False)
        else:
            self.norm = RMSNorm(dim, eps=norm_eps)
            self.head = None  # type: ignore

    def embed(
        self, x: mx.array, kv_cache: Optional[List[Tuple[mx.array, mx.array]]] = None, norm: bool = False
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        """Compute embedding for the input tokens.

        Args:
            x (mx.array): input tokens
            kv_cache (Optional[List[Tuple[mx.array, mx.array]]]): key-value cache
            norm (bool, optional): whether to normalize the output. Defaults to False.

        Returns:
            Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]: output and key-value cache
        """
        h = self.token_embed(x)

        if self.gemma:
            h = h * (self.dim**0.5)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if kv_cache is None:
            kv_cache = [None] * len(self.layers)  # type: ignore

        for e, layer in enumerate(self.layers):
            h, kv_cache[e] = layer(x=h, mask=mask, kv_cache=kv_cache[e])

        return self.norm(h) if norm else h, kv_cache

    def __call__(
        self, x: mx.array, kv_cache: Optional[List[Tuple[mx.array, mx.array]]] = None
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        """Forward pass

        Args:
            x (mx.array): input tokens
            kv_cache (Optional[List[mx.array]], optional): key-value cache. Defaults to None.

        Returns:
            Tuple[mx.array, List[mx.array]]: output and key-value cache
        """
        x, kv_cache = self.embed(x, kv_cache=kv_cache, norm=True)
        if self.gemma:
            out = self.token_embed.as_linear(x)
        else:
            out = self.head(x)
        return out, kv_cache  # type: ignore

    def generate(self, x: mx.array, temp: Optional[float] = 0.0):
        """Generate tokens from a given input

        Args:
            x (mx.array): input tokens
            temp (Optional[float], optional): model temperature. Defaults to 0.0.
        """

        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        logits, kv_cache = self(x[None])
        y = sample(logits[:, -1, :])
        yield y

        while True:
            logits, kv_cache = self(y[:, None], kv_cache=kv_cache)
            y = sample(logits.squeeze(1))
            yield y
