from .teknium import OpenHermesChat
from .mistral import MistralChat
from .llama import LLaMAChat
from .phi2 import Phi2Chat
from typing import Union, List, Dict, Optional

def create_chat(
    model_name: str,
    personality: str = "",
    examples: List[Dict[str, str]] = [],
) -> Union[MistralChat, OpenHermesChat, LLaMAChat, Phi2Chat]:
    """Create chat class based on model name

    Args:
        model_name (str): model name
        personality (str, optional): model personality (a descritpion of how the model should behave). Defaults to "".
        examples (List[Dict[str, str]], optional): a list of examples of dialog [{"user": ..., "model": ...}]. Defaults to [].

    Returns:
        Union[MistralChat, OpenHermesChat, LLaMAChat]: chat class
    """
    
    if "openhermes" in model_name.lower():
        return OpenHermesChat(personality, examples)
    
    if "mistral" in model_name.lower():
        return MistralChat(personality, examples)
    
    if "llama" in model_name.lower():
        return LLaMAChat(personality, examples)
    
    if "phi2" in model_name.lower():
        return Phi2Chat(personality, examples)
    