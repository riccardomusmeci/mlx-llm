from ..utils.session import Session
from ._base import Prompt


class LLaMA2Prompt(Prompt):
    """LLaMA Instruct prompt that follows this structure

    [INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_msg_1 }} [/INST] {{ model_answer_1 }} [INST] {{ user_msg_2 }} [/INST]

    Args:
        system (str): system prompt

    """

    SYS_START = "<<SYS>>"
    SYS_END = "<</SYS>>"
    INST_START = "[INST]"
    TEXT_END = "[/INST]"

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
        [INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_msg_1 }} [/INST] {{ model_answer_1 }} [INST] {{ user_msg_2 }} [/INST]
        """

        prompt = f"{self.INST_START} {self.SYS_START}\n{self.system}\n{self.SYS_END}\n\n"
        for qa in session.history:
            # qa --> {"question": ..., "answer": ...}
            prompt += qa["question"] + " "  # type: ignore
            if qa["answer"] is None:
                break
            else:
                prompt += f"{self.TEXT_END} " + qa["answer"] + f" {self.INST_START} "

        if not prompt.endswith(self.TEXT_END):
            prompt += f"{self.TEXT_END}"

        return prompt


class TinyLLaMAPrompt(Prompt):
    """TinyLLaMa instruct prompt that follows this structure

    <|system|>
    You are a helpful AI assistant.</s>
    <|user|>
    How to explain Internet for a medieval knight?</s>
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


class LLaMA3Prompt(Prompt):
    """LLaMA Instruct prompt that follows this structure

    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Args:
        system (str): system prompt

    """

    PROMPT_START = "<|begin_of_text|>"
    START_HEADER = "<|start_header_id|>"
    END_HEADER = "<|end_header_id|>"
    TEXT_END = "<|eot_id|>"

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
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

        prompt = f"{self.PROMPT_START} {self.START_HEADER}system{self.END_HEADER}\n\n{self.system}{self.TEXT_END}"
        for qa in session.history:
            # qa --> {"question": ..., "answer": ...}
            prompt += f"{self.START_HEADER}user{self.END_HEADER}\n\n{qa['question']}{self.TEXT_END}"
            prompt += f"{self.START_HEADER}assistant{self.END_HEADER}\n\n"
            if qa["answer"] is None:
                break
            else:
                prompt += qa["answer"] + self.TEXT_END

        return prompt
