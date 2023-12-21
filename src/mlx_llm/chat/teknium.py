from typing import Dict, List
from .base import Chat

class OpenHermesChat(Chat):
    """OpenHermes chat class

    Args:
        personality (str): OpenHermes personality (a description of what personality model has)
        examples (List[Dict[str, str]]): a list of examples of dialog [{"question": ..., "answer": ...}]
    """
    
    def __init__(self, personality: str, examples: List[Dict[str, str]], end_str: str = "<|im_end|>"):
        
        super().__init__(personality, examples, end_str)
        
        self.SYSTEM = "<|im_start|>system"
        self.USER = "<|im_start|>user"
        self.ASSISTANT = "<|im_start|>assistant"
        self.END = "<|im_end|>"
        
    @property
    def history(self) -> str:
        """Chat history

        Returns:
            str: history
        """
        prompt = ""
        for example in self.examples:
            prompt += self.USER + "\n" + example["user"] + self.END + "\n" + self.ASSISTANT + "\n"
            if example["model"] is not None:
                prompt += example["model"] + self.END + "\n"
        return prompt
    
    @property
    def prompt(self) -> str:
        """Return OpenHermes prompt based on this structure
            <|im_start|>system
            {{ system_prompt }}
            <|im_start|>user
            {{ user_prompt }}<|im_end|>
            <|im_start|>assistant
            {{ assistant_prompt }}<|im_end|>
        
        Returns:
            str: prompt
        """
        return f"{self.SYSTEM}\n{self.personality}\n{self.history}"
    
    # def model_status(self, answer: str) -> int:
    #     """Check if dialog is over

    #     Returns:
    #         int: -1 if model can keep generating, 0 if model is generating [INST], 1 if model is done generating [INST]
    #     """
    #     print(answer)
    #     for i in range(len(self.END_STR) - 1, 0, -1):
    #         # model is generating [INST] -> wait until it finishes but don't print anything
    #         if answer[-i:] == self.END_STR[:i]:
    #             return 0
    #     for i in range(len(self.ASSISTANT) - 1, 0, -1):
    #         # model is generating [INST] -> wait until it finishes but don't print anything
    #         if answer[-i:] == self.ASSISTANT[:i]:
    #             return 0
    #     # model is done generating [INST] -> saving answer
    #     if answer[-len(self.END_STR) :] == self.END_STR:
    #         return 1
    #     # model can keep generating
    #     return -1