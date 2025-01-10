from ..utils.session import Session
from .hermes import HermesPrompt
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


class Phi4Prompt(HermesPrompt):
    """Phi4 prompt that follows this structure

    <|im_start|>system<|im_sep|>
    You are a medieval knight and must provide explanations to modern people.<|im_end|>
    <|im_start|>user<|im_sep|>
    How should I explain the Internet?<|im_end|>
    <|im_start|>assistant<|im_sep|>
    """

    TEXT_START = "<|im_start|>"
    TEXT_END = "<|im_end|>"
    TEXT_SEP = "<|im_sep|>"
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"

    def __init__(self, system: str) -> None:
        self.system = system

    def prepare(self, session: Session) -> str:
        """Prepare prompt for model input

        Args:
            session (Session): dialog session

        Returns:
            str: model input prompt
        """

        prompt = f"{self.TEXT_START}{self.SYSTEM}{self.TEXT_SEP}\n{self.system}{self.TEXT_END}\n"
        for qa in session.history:
            prompt += f"{self.TEXT_START}{self.USER}{self.TEXT_SEP}\n{qa['question']}{self.TEXT_END}\n"
            if qa["answer"] is None:
                prompt += f"{self.TEXT_START}{self.ASSISTANT}{self.TEXT_SEP}\n"
            else:
                prompt += f"{self.TEXT_START}{self.ASSISTANT}{self.TEXT_SEP}\n{qa['answer']}{self.TEXT_END}\n"
        if len(session) == 0:
            prompt += f"{self.TEXT_START}{self.ASSISTANT}{self.TEXT_SEP}\n"
        return prompt
