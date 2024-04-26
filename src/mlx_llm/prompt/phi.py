from ..utils.session import Session
from .llama import TinyLLaMAPrompt


class Phi3Prompt(TinyLLaMAPrompt):
    """Phi3 Instruct prompt that follows this structure

    <|system|>
    You are a helpful AI assistant.<|end|>
    <|user|>
    How to explain Internet for a medieval knight?<|end|>
    <|assistant|>

    Args:
        system (str): system prompt

    """

    SYSTEM_START = "<|system|>"
    USER = "<|user|>"
    ASSISTANT = "<|assistant|>"
    TEXT_END = "<|end|>"

    def __init__(self, system: str) -> None:
        super().__init__(system)
