from ._registry import MODEL_ENTRYPOINTS, MODEL_TOKENIZER
from typing import Optional, Tuple, Union
import mlx.nn as nn
from ._utils import load_weights, load_weights_from_hf
from transformers import AutoTokenizer

__all__ = ["list_models", "create_model"]

def list_models() -> None:
    """List all available LLM models.
    """
    print("Available models:")
    for model_name in list(MODEL_ENTRYPOINTS.keys()):
        print(f"\t- {model_name}")

def create_model(model_name: str, weights: Union[str, bool] = True, strict: bool = False, verbose: bool = False) -> nn.Module:
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
        
    if weights and isinstance(weights, bool):
        model = load_weights_from_hf(
            model=model,
            model_name=model_name,
            strict=strict,
            verbose=verbose
        )
    elif isinstance(weights, str):
        model = load_weights(
            model=model,
            weights=weights,
            strict=strict,
            verbose=verbose
        )
        
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
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER[model_name])

    return tokenizer
