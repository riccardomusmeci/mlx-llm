from .llama import LLaMA2Prompt, LLaMA3Prompt, TinyLLaMAPrompt
from .phi import Phi3Prompt
from .mistral import MistralPrompt
from ._base import Prompt

PROMPT_ENTRYPOINTS = {
    "llama2": LLaMA2Prompt,
    "llama3": LLaMA3Prompt,
    "phi3": Phi3Prompt,
    "mistral": MistralPrompt,
    "tiny_llama": TinyLLaMAPrompt,
}


def create_prompt(model_family: str, system: str) -> Prompt:
    """Create prompt based on model family

    Args:
        model_family (str): model family

    Returns:
        Prompt: prompt class
    """
    if model_family not in PROMPT_ENTRYPOINTS:
        raise ValueError(f"Model family {model_family} not found in available prompts")

    return PROMPT_ENTRYPOINTS[model_family](system=system)
