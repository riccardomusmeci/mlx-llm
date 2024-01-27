from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn


class TransformerEncoderLayer(nn.Module):
    """
    A transformer encoder layer with (the original BERT) post-normalization.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: Optional[int] = None,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        mlp_dim = mlp_dim or dim * 4
        self.attention = nn.MultiHeadAttention(dim, num_heads, bias=True)
        self.ln1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.linear1 = nn.Linear(dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, dim)
        self.gelu = nn.GELU()

    def __call__(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        add_and_norm = self.ln1(x + attention_out)

        ff = self.linear1(add_and_norm)
        ff_gelu = self.gelu(ff)
        ff_out = self.linear2(ff_gelu)
        x = self.ln2(ff_out + add_and_norm)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_layers: int, 
        num_heads: int, 
        mlp_dim: Optional[int] = None
    ):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(dim, num_heads, mlp_dim)
            for i in range(num_layers)
        ]

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x


class BertEmbeddings(nn.Module):
    def __init__(
        self, 
        dim: int, 
        vocab_size: int,
        layer_norm_eps: float,
        max_position_embeddings: int
    ):
        self.word_embeddings = nn.Embedding(vocab_size, dim)
        self.token_type_embeddings = nn.Embedding(2, dim)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, dim
        )
        self.norm = nn.LayerNorm(dim, eps=layer_norm_eps)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(
            mx.broadcast_to(mx.arange(input_ids.shape[1]), input_ids.shape)
        )
        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = position + words + token_types
        return self.norm(embeddings)


class Bert(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_attention_heads: int,
        num_hidden_layers: int,
        vocab_size: int,
        hidden_dropout_prob: float,
        layer_norm_eps: float,
        max_position_embeddings: int
    ):
        self.embeddings = BertEmbeddings(
            dim=dim,
            vocab_size=vocab_size,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=max_position_embeddings
        )
        self.encoder = TransformerEncoder(
            num_layers=num_hidden_layers,
            dim=dim,
            num_heads=num_attention_heads,
        )
        self.pooler = nn.Linear(dim, dim)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array,
        attention_mask: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        
        x = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            # convert 0's to -infs, 1's to 0's, and make it broadcastable
            attention_mask = mx.log(attention_mask)
            attention_mask = mx.expand_dims(attention_mask, (1, 2))

        y = self.encoder(x, attention_mask)
        return y, mx.tanh(self.pooler(y[:, 0]))
    
def bert_base_uncased():
    return Bert(
        dim=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        vocab_size=30522,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        max_position_embeddings=512
    )
    
def bert_base_cased():
    return Bert(
        dim=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        vocab_size=28996,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        max_position_embeddings=512
    )
    
def bert_large_uncased():
    return Bert(
        dim=1024,
        num_attention_heads=16,
        num_hidden_layers=24,
        vocab_size=30522,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        max_position_embeddings=512
    )
    
def bert_large_cased():
    return Bert(
        dim=1024,
        num_attention_heads=16,
        num_hidden_layers=24,
        vocab_size=28996,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        max_position_embeddings=512
    )
