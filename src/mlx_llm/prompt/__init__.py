from ._base import Prompt
from .llama import LLaMA2Prompt, LLaMA3Prompt, TinyLLaMAPrompt
from .gemma import GemmaPrompt
from .mistral import MistralPrompt
from .phi import Phi3Prompt

PROMPT_ENTRYPOINTS = {
    "llama2": LLaMA2Prompt,
    "llama3": LLaMA3Prompt,
    "phi3": Phi3Prompt,
    "mistral": MistralPrompt,
    "tinyllama": TinyLLaMAPrompt,
    "gemma": GemmaPrompt,
}


def create_prompt(model_name: str, system: str = "") -> Prompt:
    """Create prompt based on model family

    Args:
        model_name (str): model name
        system (str): system prompt

    Returns:
        Prompt: prompt class
    """

    model_family = model_name.replace("_", "").replace(".", "").replace("-", "").lower()
    found = False
    for k in PROMPT_ENTRYPOINTS.keys():
        if k in model_family:
            model_family = k
            found = True
            break
    if not found:
        raise ValueError(
            f"Model {model_name} not found in available prompts. Familes available: {list(PROMPT_ENTRYPOINTS.keys())}"
        )

    return PROMPT_ENTRYPOINTS[model_family](system=system)  # type: ignore
