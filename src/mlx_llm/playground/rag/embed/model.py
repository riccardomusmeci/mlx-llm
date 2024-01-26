import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_llm.model import create_model
from transformers import AutoTokenizer
from typing import Literal


class EmbeddingModel:
    
    def __init__(self, model: nn.Module, mode: Literal['last', 'avg']="last"):
        
        assert mode in ['last', 'avg'], "mode must be either 'last' or 'avg'"
        self.model = model
        self.model.eval()
        self.mode = mode
        
    
    def last_token_pool(self, embeds: mx.array, attn_mask: mx.array) -> mx.array:
        """Last token pool embeddings

        Args:
            embeds (mx.array): embeddings
            attn_mask (mx.array): attention mask

        Returns:
            mx.array: last token pooled embeddings
        """
        left_padding = (attn_mask[:, -1].sum() == attn_mask.shape[0])
        if left_padding:
            return embeds[:, -1]
        else:
            sequence_lengths = attn_mask.sum(axis=1) - 1
            batch_size = embeds.shape[0]
            return embeds[mx.arange(batch_size), sequence_lengths]
        
    def average_pool(self, embeds: mx.array, attn_mask: mx.array) -> mx.array:
        """Average pool embeddings

        Args:
            embeds (mx.array): embeddings
            attn_mask (mx.array): attention mask

        Returns:
            mx.array: average pooled embeddings
        """
        embeds = mx.multiply(embeds, attn_mask[..., None])
        return embeds.sum(axis=1) / attn_mask.sum(axis=1)[..., None]

    
    def normalize(self, embeds: mx.array):
        """Normalize embeddings

        Args:
            embeds (mx.array): embeddings

        Returns:
            mx.array: normalized embeddings
        """
        embeds = embeds / mx.linalg.norm(embeds, ord=2, axis=1)[..., None]
        return mx.array(embeds)
        
    def __call__(self, x: mx.array, attn_mask: mx.array):
        """Compute embedding for the input tokens.

        Args:
            x (mx.array): input tokens
            attn_mask (mx.array): attention mask

        Returns:
            mx.array: embedded tokens
        """
        embeds = self.model.embed(x)
        if self.mode == 'last':
            embeds = self.last_token_pool(embeds, attn_mask)
        if self.mode == 'avg':
            embeds = self.average_pool(embeds, attn_mask)
        embeds = self.normalize(embeds)        
        return embeds
