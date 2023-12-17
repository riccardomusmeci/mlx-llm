from typing import List, Dict
from ..llm.base import Chat

class MistralChat(Chat):
    """Mistral chat class

    Args:
        personality (str): Mistral personality (a description of what personality model has)
        examples (List[Dict[str, str]]): a list of examples of dialog [{"question": ..., "answer": ...}]
    """
    def __init__(self, personality: str, examples: List[Dict[str, str]]):
        
        super().__init__(personality, examples)
        
        """
        [INST] {{ system_prompt }} + {{ user_prompt }} [/INST] {{ model_prompt }}</s>[INST] {{ user_prompt }} [/INST
        """
        self.INST_START = "[INST]"
        self.INST_END = "[/INST]"
        self.END = "</s>"
        self.CHECK_STATUS = "</s>"
        
    # TODO: add check self.CHECK_STATUS in the middle of the answer and put in original class
    def add_answer(self, answer: str):
        """Add answer to dialog

        Args:
            answer (str): dialog answer
        """
        if self.CHECK_STATUS in answer:
            answer = answer.replace(self.CHECK_STATUS, "")
        self.examples[-1]["model"] = answer
        
    @property
    def history(self) -> str:
        """Chat history
        
        {{ user_prompt }} [/INST] {{ model_prompt }}</s>[INST] {{ user_prompt }} [/INST]

        Returns:
            str: history
        """
        prompt = ""
        for example in self.examples:
            prompt += example["user"] + " " + self.INST_END 
            if example["model"] is not None:
                prompt += " " + example["model"] + self.END + self.INST_START
        return prompt
    
    @property
    def prompt(self) -> str:
        """Return Mistral prompt based on this structure
        
        [INST] {{ system_prompt }} + {{ user_prompt }} [/INST] {{ model_prompt }}</s>[INST] {{ user_prompt }} [/INST]
        
        Returns:
            str: prompt
        """
        return f"{self.INST_START} {self.personality} {self.history}"
         
    def reset(self):
        """Reset dialog but not personality
        """
        self.examples = self.base_examples.copy()
    
    def model_status(self, answer: str) -> int:
        """Check if dialog is over

        Returns:
            int: -1 if model can keep generating, 0 if model is generating [INST], 1 if model is done generating [INST]
        """
        for i in range(len(self.CHECK_STATUS) - 1, 0, -1):
            # model is generating [INST] -> wait until it finishes but don't print anything
            if answer[-i:] == self.CHECK_STATUS[:i]:
                return 0
        # model is done generating [INST] -> saving answer
        if answer[-len(self.CHECK_STATUS) :] == self.CHECK_STATUS:
            return 1
        # model can keep generating
        return -1


class OpenHermesChat(Chat):
    """OpenHermes chat class

    Args:
        personality (str): OpenHermes personality (a description of what personality model has)
        examples (List[Dict[str, str]]): a list of examples of dialog [{"question": ..., "answer": ...}]
    """
    
    def __init__(self, personality: str, examples: List[Dict[str, str]]):
        
        super().__init__(personality, examples)
        
        self.SYSTEM = "<|im_start|>system"
        self.USER = "<|im_start|>user"
        self.ASSISTANT = "<|im_start|>assistant"
        self.END = "<|im_end|>"
        
        self.CHECK_STATUS = self.END
    
    # TODO: add check self.CHECK_STATUS in the middle of the answer and put in original class
    def add_answer(self, answer: str):
        """Add answer to dialog

        Args:
            answer (str): dialog answer
        """
        if self.USER in answer:
            answer = answer.replace(self.END, "")
        self.examples[-1]["model"] = answer
        
    @property
    def history(self) -> str:
        """Chat history

        Returns:
            str: history
        """
        prompt = ""
        for example in self.examples:
            prompt += self.USER + "\n" + example["user"] + self.END + "\n"
            if example["model"] is not None:
                prompt += self.ASSISTANT + "\n" + example["model"] + self.END + "\n"
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
         
    def reset(self):
        """Reset dialog but not personality
        """
        self.examples = self.base_examples.copy()
    
    def model_status(self, answer: str) -> int:
        """Check if dialog is over

        Returns:
            int: -1 if model can keep generating, 0 if model is generating [INST], 1 if model is done generating [INST]
        """
        for i in range(len(self.END) - 1, 0, -1):
            # model is generating [INST] -> wait until it finishes but don't print anything
            if answer[-i:] == self.END[:i]:
                return 0
        # model is done generating [INST] -> saving answer
        if answer[-len(self.END) :] == self.END:
            return 1
        # model can keep generating
        return -1
            