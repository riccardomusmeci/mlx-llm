from ..utils.session import Session
from ._base import Prompt


class MistralPrompt(Prompt):
    """Mistral Instruct prompt that follows this structure

    [INST] {{ system_prompt }} + {{ user_prompt }} [/INST] {{ model_prompt }}</s>[INST] {{ user_prompt }} [/INST]

    Args:
        system (str): system prompt

    """

    INST_START = "[INST]"
    INST_END = "[/INST]"
    TEXT_START = "<s>"
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
        [INST] {{ system_prompt }} {{ user_prompt }} [/INST] {{ model_prompt }}</s>[INST] {{ user_prompt }} [/INST]
        """
        """
        "<s>[INST] What is your favourite condiment? [/INST]"
        "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
        "[INST] Do you have mayonnaise recipes? [/INST]"
        """

        prompt = f"{self.TEXT_START} {self.INST_START} {self.system} "
        for i, qa in enumerate(session.history):
            # qa --> {"question": ..., "answer": ...}
            if i == 0:
                prompt += f"{qa['question']} {self.INST_END}\n"
            else:
                prompt += f"{self.INST_START} {qa['question']} {self.INST_END}\n"
            if qa["answer"] is None:
                break
            else:
                prompt += f"{qa['answer']} {self.TEXT_END}\n"

        return prompt


class StarlingPrompt(Prompt):
    """Starling Chat prompt that follows this structure

    GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant: {response}<|end_of_turn|>GPT4 Correct User: {follow_up_question}<|end_of_turn|>GPT4 Correct Assistant:

    Args:
        system (str): system prompt
    """

    USER = "GPT4 Correct User:"
    ASSISTANT = "GPT4 Correct Assistant:"
    TEXT_END = "<|end_of_turn|>"

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
        GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant: {response}<|end_of_turn|>GPT4 Correct User: {follow_up_question}<|end_of_turn|>GPT4 Correct Assistant:
        """

        prompt = f"{self.USER} {self.system}"
        for i, qa in enumerate(session.history):
            if i == 0:
                prompt += f"{qa['question']}{self.TEXT_END}"
            else:
                prompt += f"{self.USER} {qa['question']}{self.TEXT_END}"
            if qa["answer"] is None:
                prompt += f"{self.ASSISTANT}"
            else:
                prompt += f"{self.ASSISTANT} {qa['answer']} {self.TEXT_END}"

        return prompt
