import mlx.core as mx
import mlx.nn as nn
import math

__all__ = ["Transformer"]

class Attention(nn.Module):
    
    def __init__(self, dim: int, n_heads: int):
        super().__init__()

        self.n_heads = n_heads

        self.rope = nn.RoPE(dim // n_heads, traditional=True)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, queries, keys, values, mask=None, cache=None):
        
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

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

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Note that we return the keys and values to possibly be used as a cache
        return self.out_proj(values_hat), (keys, values)


class Encoder(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, n_heads: int):
        super().__init__()

        self.attention = Attention(dim, n_heads)

        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        self.linear1 = nn.Linear(dim, mlp_dim, bias=False)
        self.linear2 = nn.Linear(dim, mlp_dim, bias=False)
        self.linear3 = nn.Linear(mlp_dim, dim, bias=False)

    def __call__(self, x, mask=None, cache=None):
        y = self.norm1(x)
        y, cache = self.attention(y, y, y, mask, cache)
        x = x + y

        y = self.norm2(x)
        a = self.linear1(y)
        b = self.linear2(y)
        y = a * mx.sigmoid(a) * b
        y = self.linear3(y)
        x = x + y

        return x, cache


class Transformer(nn.Module):
    
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        dim: int,
        mlp_dim: int,
    ) -> None:
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = [
            Encoder(dim=dim, mlp_dim=mlp_dim, n_heads=n_heads)
            for _ in range(n_layers)
        ]
        self.norm = nn.RMSNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)
          
    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        x = self.embedding(x)
        for l in self.layers:
            x, _ = l(x, mask)
        x = self.norm(x)
        return self.out_proj(x)
    
    def generate(self, x, temp=1.0):
        cache = []
        
        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        # First we process the prompt x the same was as in __call__ but
        # save the caches in cache
        x = self.embedding(x)
        for l in self.layers:
            x, c = l(x, mask=mask)
            # We store the per layer cache in a simple python list
            # if len(cache) == 0:
            #     print(x.shape, c[0].shape, c[1].shape, len(cache), None)
            # else:
            #     print(x.shape, c[0].shape, c[1].shape, len(cache), cache[-1][0].shape)
            
            cache.append(c)
        x = self.norm(x)
        # We only care about the last logits that generate the next token
        y = self.out_proj(x[:, -1])
        y = mx.random.categorical(y * (1 / temp))
        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y
        
        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None]

            x = self.embedding(x)
            for i in range(len(cache)):
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding the
                # old cache the moment it is not needed anymore.
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
            x = self.norm(x)
            y = self.out_proj(x[:, -1])
            y = mx.random.categorical(y * (1 / temp))
            
            # print(y)

            yield y