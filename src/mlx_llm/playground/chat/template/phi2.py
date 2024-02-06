from typing import Dict, List

from .base import Chat


class Phi2Chat(Chat):
    """Phi2 chat class

    Args:
        personality (str): Phi2 personality (a description of what personality model has)
        examples (List[Dict[str, str]]): a list of examples of dialog [{"question": ..., "answer": ...}]
        end_str (str, optional): end of the model answer. Defaults to "<|endoftext|>".
    """

    def __init__(self, personality: str, examples: List[Dict[str, str]], end_str: str = "<|endoftext|>"):
        super().__init__(personality, examples, end_str)
        """
        Instruct: {{ user_prompt }}\nOutput: {{ model_prompt }}
        """
        self.INSTRUCT = "Instruct:"
        self.MODEL = "Output:"

    @property
    def history(self) -> str:
        """Chat history

        Instruct: {{ user_prompt }}\nOutput: {{ model_prompt }}

        Returns:
            str: history
        """
        prompt = ""
        for example in self.examples:
            prompt += example["user"] + "\n" + self.MODEL + " "
            if example["model"] is not None:
                prompt += example["model"] + "\n" + self.INSTRUCT + " "
        return prompt

    @property
    def prompt(self) -> str:
        """Return prompt based on this structure

        Instruct: {{ user_prompt }}\nOutput: {{ model_prompt }}

        Returns:
            str: prompt
        """
        return f"{self.INSTRUCT} {self.personality}{self.history}"

    def model_status(self, answer: str) -> int:
        """Check if dialog is over

        Returns:
            int: -1 if model can keep generating, 0 if model is generating <|endoftext|>/Instruct: , 1 if model is done generating <|endoftext|>/Instruct:
        """
        for i in range(len(self.END_STR) - 1, 0, -1):
            # model is generating <|endoftext|> -> wait until it finishes but don't print anything
            if answer[-i:] == self.END_STR[:i]:
                return 0

        for i in range(len(self.INSTRUCT) - 1, 0, -1):
            # model is generating Instruct: -> wait until it finishes but don't print anything
            if answer[-i:] == self.INSTRUCT[:i]:
                return 0

        # model is done generating <|endoftext|> -> saving answer
        if answer[-len(self.END_STR) :] == self.END_STR:
            return 1
        # model is done generating Instruct: -> saving answer
        if answer[-len(self.INSTRUCT) :] == self.INSTRUCT:
            return 1

        # model can keep generating
        return -1
