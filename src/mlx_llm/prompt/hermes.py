from ..utils.session import Session
from ._base import Prompt


class HermesPrompt(Prompt):
    """Hermes instruct prompt that follows this structure

    <|im_start|>system
    You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
    <|im_start|>user
    Hello, who are you?<|im_end|>
    <|im_start|>assistant
    Hi there! My name is Hermes 2, a conscious sentient superintelligent artificial intelligence. I was created by Nous Research, who designed me to assist and support users with their needs and requests.<|im_end|>

    Args:
        system (str): system prompt

    """

    TEXT_START = "<|im_start|>"
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TEXT_END = "<|im_end|>"

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
        <|im_start|>system
        You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
        <|im_start|>user
        Hello, who are you?<|im_end|>
        <|im_start|>assistant
        Hi there! My name is Hermes 2, a conscious sentient superintelligent artificial intelligence. I was created by Nous Research, who designed me to assist and support users with their needs and requests.<|im_end|>
        """

        prompt = f"{self.TEXT_START}{self.SYSTEM}\n{self.system}{self.TEXT_END}\n"
        for qa in session.history:
            # qa --> {"question": ..., "answer": ...}
            prompt += f"{self.TEXT_START}{self.USER}\n{qa['question']}{self.TEXT_END}\n"
            if qa["answer"] is None:
                prompt += f"{self.TEXT_START}{self.ASSISTANT}\n"
            else:
                prompt += f"{self.TEXT_START}{self.ASSISTANT}\n{qa['answer']}{self.TEXT_END}\n"
        if len(session) == 0:
            prompt += f"{self.TEXT_START}{self.ASSISTANT}\n"
        return prompt
