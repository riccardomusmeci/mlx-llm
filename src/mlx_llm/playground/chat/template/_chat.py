from typing import Dict, List, Optional, Union

from .llama import LLaMAChat
from .mistral import MistralChat
from .phi2 import Phi2Chat
from .teknium import OpenHermesChat
from .tiny_llama import TinyLLaMAChat

__all__ = ["create_chat"]

FACTORY = {
    "Phi2": Phi2Chat,
    "LLaMA-2-7B-chat": LLaMAChat,
    "Mistral-7B-Instruct-v0.1": MistralChat,
    "Mistral-7B-Instruct-v0.2": MistralChat,
    "OpenHermes-2.5-Mistral-7B": OpenHermesChat,
    "TinyLlama-1.1B-Chat-v0.6": TinyLLaMAChat,
    # "zephyr-7b-beta": TinyLLaMAChat,
}


def create_chat(
    model_name: str,
    personality: str = "",
    examples: List[Dict[str, str]] = (),
):
    """Create chat class based on model name

    Args:
        model_name (str): model name
        personality (str, optional): model personality (a descritpion of how the model should behave). Defaults to "".
        examples (List[Dict[str, str]], optional): a list of examples of dialog [{"user": ..., "model": ...}]. Defaults to [].

    Returns:
        Chat: chat class
    """

    assert model_name in list(FACTORY.keys()), f"Unknown model chat: {model_name}."

    return FACTORY[model_name](personality, examples)
