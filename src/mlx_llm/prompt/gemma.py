from ..utils.session import Session
from ._base import Prompt


class GemmaPrompt(Prompt):
    """LLaMA Instruct prompt that follows this structure

    <bos><start_of_turn>user
    Write a hello world program<end_of_turn>
    <start_of_turn>model

    Args:
        system (str): system prompt
    """

    START = "<bos>"
    USER = "user"
    MODEL = "model"
    TURN_START = "<start_of_turn>"
    TEXT_END = "<end_of_turn>"

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
        <bos><start_of_turn>user
        {{ system }} + {{ question }} <end_of_turn>
        <start_of_turn>model
        """

        prompt = f"{self.START}{self.TURN_START}{self.USER}\n{self.system} "
        for i, qa in enumerate(session.history):
            # qa --> {"question": ..., "answer": ...}
            if i == 0:
                prompt += qa["question"] + " " + self.TEXT_END + "\n" + self.TURN_START + self.MODEL + "\n"
            else:
                prompt += (
                    self.TURN_START
                    + self.USER
                    + "\n"
                    + qa["question"]
                    + self.TEXT_END
                    + "\n"
                    + self.TURN_START
                    + self.MODEL
                    + "\n"
                )
            if qa["answer"] is None:
                break
            else:
                prompt += f"{qa['answer']} {self.TEXT_END}\n"

        return prompt
