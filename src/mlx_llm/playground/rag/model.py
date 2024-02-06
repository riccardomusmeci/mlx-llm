from typing import Dict, List, Literal, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoTokenizer

from mlx_llm.model import create_model, create_tokenizer


class EmbeddingModel:
    """Embedding model

    Args:
        model_name (str): model name
        max_length (int): max length of the input sequence (embedding size)
        mode (Optional[Literal['last', 'avg']]): mode for pooling embeddings. Defaults to None.
    """

    def __init__(self, model_name: str, max_length: int, mode: Optional[Literal["last", "avg"]] = None):
        self.model = create_model(model_name=model_name, weights=True, strict=True)
        self.tokenizer = create_tokenizer(model_name)
        self.max_length = max_length
        self.model.eval()
        self.is_bert = "Bert" == self.model.__class__.__name__
        self.mode = mode

    def last_token_pool(self, embeds: mx.array, attn_mask: mx.array) -> mx.array:
        """Last token pool embeddings

        Args:
            embeds (mx.array): embeddings
            attn_mask (mx.array): attention mask

        Returns:
            mx.array: last token pooled embeddings
        """
        left_padding = attn_mask[:, -1].sum() == attn_mask.shape[0]
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

    def prepare_tokens(self, text: List) -> Dict[str, mx.array]:
        """Prepare tokens for the model

        Args:
            text (List): input text

        Returns:
            Dict[str, mx.array]: tokens for the model
        """
        tokens = self.tokenizer(
            text, max_length=self.max_length - 1, return_attention_mask=False, padding=False, truncation=True
        )

        if not self.is_bert:  # otherwise it puts None at the end with eos_token_id
            tokens["input_ids"] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in tokens["input_ids"]]

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        tokens = self.tokenizer.pad(tokens, padding=True, return_attention_mask=True, return_tensors="np")

        tokens = {key: mx.array(v.astype(np.int32)) for key, v in tokens.items()}
        return tokens

    def __call__(self, text: Union[List[str], str]) -> mx.array:
        """Compute embedding for the input tokens.

        Args:
            text (Union[List[str], str]): input text

        Returns:
            mx.array: embedded tokens
        """

        if isinstance(text, str):
            text = [text]

        tokens = self.prepare_tokens(text)
        output, embeds = self.model(**tokens)
        if self.mode == "last":
            embeds = self.last_token_pool(output, tokens["attention_mask"])
        if self.mode == "avg":
            embeds = self.average_pool(output, tokens["attention_mask"])
        embeds = self.normalize(embeds)
        return embeds
