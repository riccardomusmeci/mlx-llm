from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from ._registry import MODEL_ENTRYPOINTS, MODEL_QUANTIZED, MODEL_TOKENIZER
from ._utils import download_from_hf, load_weights, load_weights_from_hf, quantize

__all__ = ["list_models", "create_model"]


def list_models() -> None:
    """List all available LLM models."""
    print("Available models:")
    for model_name in list(MODEL_ENTRYPOINTS.keys()):
        print(f"\t- {model_name}")


def create_model(
    model_name: str,
    weights: Union[str, bool] = True,
    strict: bool = False,
    quantized: bool = False,
    verbose: bool = False,
) -> nn.Module:
    """Create a LLM model.

    Example:

    ```
    >>> from mlx_llm.model import create_model

    >>> # Create a Phi2 model with no pretrained weights.
    >>> model = create_model('Phi2')

    >>> # Create a Phi2 model with pretrained weights from HF.
    >>> model = create_model('Phi2', weights=True)

    >>> # Create a Phi2 model with custom weights.
    >>> model = create_model('Phi2', weights="path/to/weights.npz")
    ```

    Args:
        model_name (str): model name
        weights (Union[str, bool]): if True, load pretrained weights from HF. If str, load weights from the given path. Defaults to True.
        quantized (bool, optional): whether to quantize the model. Defaults to False.
        strict (bool, optional): whether to strictly enforce that the keys in weights match the keys of the model. Defaults to False.
        verbose (bool, optional): whether to print the model summary. Defaults to False.

    Returns:
        nn.Module: a LLM model

    Raises:
        ValueError: Unknown model name
    """

    if model_name not in MODEL_ENTRYPOINTS:
        raise ValueError(f"Unknown model name: {model_name}.")
    model = MODEL_ENTRYPOINTS[model_name]()

    # check quantization
    if quantized:
        quantize_args = MODEL_QUANTIZED.get(model_name, None)
        if quantize_args is not None:
            model = quantize(model, **quantize_args)
        else:
            raise ValueError(f"No quantization set for model {model_name}.")

    # loading weights
    if weights and isinstance(weights, bool):
        weights = download_from_hf(model_name)

    model = load_weights(model=model, weights=weights, strict=strict, verbose=verbose)

    return model


def create_tokenizer(model_name: str) -> AutoTokenizer:
    """Create a tokenizer for a LLM model.

    Args:
        model_name (str): model name

    Raises:
        ValueError: Unknown model name

    Returns:
        AutoTokenizer: tokenizer
    """
    if model_name not in MODEL_TOKENIZER:
        raise ValueError(f"Unknown model name: {model_name}.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER[model_name], legacy=True)

    return tokenizer


def generate(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.1,
):
    # generate answer
    x = mx.array([tokenizer.bos_token_id] + tokenizer.encode(prompt))

    skip = 0
    tokens = []

    print(prompt, end="", flush=True)
    for _i, token in enumerate(model.generate(x, temperature)):
        tokens.append(token.item())  # actually compute the token
        if len(tokens) >= max_tokens:
            break
        token_list = list(tokens)
        if token_list[-1] == tokenizer.eos_token_id:
            break
        answer = tokenizer.decode(token_list)
        print(answer[skip:], end="", flush=True)
        skip = len(answer)
