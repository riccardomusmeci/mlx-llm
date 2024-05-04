from ._base import Prompt
from .gemma import GemmaPrompt
from .llama import LLaMA2Prompt, LLaMA3Prompt, TinyLLaMAPrompt
from .mistral import MistralPrompt, StarlingPrompt
from .openelm import OpenELMPrompt
from .phi import Phi3Prompt
from .hermes import HermesPrompt

PROMPT_ENTRYPOINTS = {
    "llama2": LLaMA2Prompt,
    "llama3": LLaMA3Prompt,
    "phi3": Phi3Prompt,
    "mistral": MistralPrompt,
    "tinyllama": TinyLLaMAPrompt,
    "gemma": GemmaPrompt,
    "openelm": OpenELMPrompt,
    "starling": StarlingPrompt,
    "hermes": HermesPrompt,
}

def list_prompts() -> None:
    """List all available prompts."""
    print("Available prompts based on model families:")
    for model_family in list(PROMPT_ENTRYPOINTS.keys()):
        print(f"\t- {model_family}")

def create_prompt(model_family: str, system: str = "") -> Prompt:
    """Create prompt based on model family

    Args:
        model_name (str): model name
        system (str): system prompt

    Returns:
        Prompt: prompt class
    """

    found = False
    assert model_family in PROMPT_ENTRYPOINTS.keys(), f"Model family {model_family} not found in available prompts."
    return PROMPT_ENTRYPOINTS[model_family](system=system)  # type: ignore
