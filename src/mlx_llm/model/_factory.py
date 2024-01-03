from .phi2 import phi2
from .transformer import (
    llama_2_7B_chat, 
    tiny_llama_chat_v06,
    openhermes_25_mistral_7B, 
    mistral_7B_instruct_v01, 
    mistral_7B_instruct_v02
)
from typing import Optional, Tuple, Union

FACTORY = {
    "Phi2": phi2,
    "LLaMA-2-7B-chat": llama_2_7B_chat,
    "TinyLlama-1.1B-Chat-v0.6": tiny_llama_chat_v06,
    "Mistral-7B-Instruct-v0.1": mistral_7B_instruct_v01,
    "Mistral-7B-Instruct-v0.2": mistral_7B_instruct_v02,
    "OpenHermes-2.5-Mistral-7B": openhermes_25_mistral_7B
}

__all__ = ["list_models", "create_model"]

def list_models() -> None:
    """List all available LLM models.
    """
    print("Available models:")
    for model_name in FACTORY:
        print(f"\t- {model_name}")

def create_model(model_name: str):
    """Create a LLM model.

    Args:
        model_name (str): model name

    Raises:
        ValueError: Unknown model name
    """
    
    if model_name not in FACTORY:
        raise ValueError(f"Unknown model name: {model_name}.")
    return FACTORY[model_name]()
