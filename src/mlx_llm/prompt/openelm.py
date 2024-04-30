from ..utils.session import Session
from ._base import Prompt


class OpenELMPrompt(Prompt):
    """OpenELM instruct prompt that follows this structure

    <|system|>
    You are a helpful AI assistant. </s>
    <|user|>
    How to explain Internet for a medieval knight? </s>
    <|assistant|>

    Args:
        system (str): system prompt

    """

    SYSTEM_START = "<|system|>"
    USER = "<|user|>"
    ASSISTANT = "<|assistant|>"
    TEXT_END = "</s>"

    def __init__(self, system: str) -> None:
        self.system = system

    def prepare(self, session: Session) -> str:
        """Prepare prompt for model input

        Args:
            session (Session): dialog session

        Returns:
            str: model input prompt
        """

        """
        <|system|>
        You are a helpful AI assistant.</s>
        <|user|>
        How to explain Internet for a medieval knight?</s>
        <|assistant|>
        """

        prompt = f"{self.SYSTEM_START}\n{self.system}{self.TEXT_END}"
        for qa in session.history:
            # qa --> {"question": ..., "answer": ...}
            prompt += f"\n{self.USER}\n{qa['question']}{self.TEXT_END}"
            if qa["answer"] is None:
                break
            else:
                prompt += f"\n{self.ASSISTANT}\n{qa['answer']}{self.TEXT_END}"

        prompt += f"\n{self.ASSISTANT}\n"

        return prompt
