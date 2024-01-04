from .phi2 import phi2
from .transformer import (
    llama_2_7B_chat, 
    tiny_llama_chat_v06,
    openhermes_25_mistral_7B, 
    mistral_7B_instruct_v01, 
    mistral_7B_instruct_v02,
    e5_mistral_7b_instruct
)
from typing import Optional, Tuple, Union
import mlx.nn as nn
from .utils import load_weights

FACTORY = {
    "Phi2": phi2,
    "LLaMA-2-7B-chat": llama_2_7B_chat,
    "TinyLlama-1.1B-Chat-v0.6": tiny_llama_chat_v06,
    "Mistral-7B-Instruct-v0.1": mistral_7B_instruct_v01,
    "Mistral-7B-Instruct-v0.2": mistral_7B_instruct_v02,
    "OpenHermes-2.5-Mistral-7B": openhermes_25_mistral_7B,
    "e5-mistral-7b-instruct":  e5_mistral_7b_instruct
}

__all__ = ["list_models", "create_model"]

def list_models() -> None:
    """List all available LLM models.
    """
    print("Available models:")
    for model_name in FACTORY:
        print(f"\t- {model_name}")

def create_model(model_name: str, weights_path: Optional[str] = None, strict: bool = False) -> nn.Module:
    """Create a LLM model.

    Args:
        model_name (str): model name
        weights_path (Optional[str], optional): path to weights. Defaults to None.
        strict (bool, optional): whether to strictly enforce that the keys in weights match the keys of the model. Defaults to False.
        
    Returns:
        nn.Module: a LLM model

    Raises:
        ValueError: Unknown model name
    """
    
    if model_name not in FACTORY:
        raise ValueError(f"Unknown model name: {model_name}.")
    model = FACTORY[model_name]()
    
    if weights_path is not None:
        model = load_weights(
            model=model,
            weights_path=weights_path,
            strict=strict
        ) 
    return FACTORY[model_name]()


