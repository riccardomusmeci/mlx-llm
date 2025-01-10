from ._base import Prompt
from .gemma import GemmaPrompt
from .hermes import HermesPrompt
from .llama import LLaMA2Prompt, LLaMA3Prompt, TinyLLaMAPrompt
from .mistral import MistralPrompt, StarlingPrompt
from .openelm import OpenELMPrompt
from .phi import Phi3Prompt

PROMPT_ENTRYPOINTS = {
    "llama_2_7b_chat_hf": LLaMA2Prompt,
    "llama_2_7b_hf": LLaMA2Prompt,
    "llama_3_8b": LLaMA3Prompt,
    "llama_3_8b_instruct": LLaMA3Prompt,
    "llama_3_2_1b_instruct": LLaMA3Prompt,
    "llama_3_2_3b_instruct": LLaMA3Prompt,
    "hermes_2_pro_llama_3_8b": HermesPrompt,
    "phi_3_mini_4k_instruct": Phi3Prompt,
    "phi_3_mini_128k_instruct": Phi3Prompt,
    "phi_3.5_mini_instruct": Phi3Prompt,
    "mistral_7b_instruct_v0.2": MistralPrompt,
    "starling_lm_7b_beta": StarlingPrompt,
    "openhermes_2.5_mistral_7b": HermesPrompt,
    "tiny_llama_1.1B_chat_v1.0": TinyLLaMAPrompt,
    "gemma_1.1_2b_it": GemmaPrompt,
    "gemma_1.1_7b_it": GemmaPrompt,
    "gemma_2_2b_it": GemmaPrompt,
    "gemma_2_9b_it": GemmaPrompt,
    "openelm_3B_instruct": OpenELMPrompt,
    "openelm_1.1B_instruct": OpenELMPrompt,
    "openelm_450M_instruct": OpenELMPrompt,
    "openelm_270M_instruct": OpenELMPrompt,
    "smollm2_1.7B_instruct": HermesPrompt,
    "smollm2_360M_instruct": HermesPrompt,
    "smollm2_135M_instruct": HermesPrompt,
}


def list_prompts() -> None:
    """List all available prompts."""
    print("Available prompts based on model families:")
    for model_family in list(PROMPT_ENTRYPOINTS.keys()):
        print(f"\t- {model_family}")


def create_prompt(model_name: str, system: str = "") -> Prompt:
    """Create prompt based on model family

    Args:
        model_name (str): model name
        system (str): system prompt

    Returns:
        Prompt: prompt class
    """

    assert model_name in PROMPT_ENTRYPOINTS.keys(), f"Model {model_name} not found in available prompts."
    return PROMPT_ENTRYPOINTS[model_name](system=system)  # type: ignore
