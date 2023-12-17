from typing import List, Dict
from .base import Chat

class MistralChat(Chat):
    """Mistral chat class

    Args:
        personality (str): Mistral personality (a description of what personality model has)
        examples (List[Dict[str, str]]): a list of examples of dialog [{"question": ..., "answer": ...}]
        end_str (str, optional): end of the model answer. Defaults to "</s>".
    """
    def __init__(self, personality: str, examples: List[Dict[str, str]], end_str: str = "</s>"):
        
        super().__init__(personality, examples, end_str)
        """
        [INST] {{ system_prompt }} + {{ user_prompt }} [/INST] {{ model_prompt }}</s>[INST] {{ user_prompt }} [/INST
        """
        self.INST_START = "[INST]"
        self.INST_END = "[/INST]"
        self.START = "<s>"
        self.END = "</s>"
        
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
                prompt += " " + example["model"] + self.END + self.INST_START + " "
        return prompt
    
    @property
    def prompt(self) -> str:
        """Return Mistral prompt based on this structure
        
        [INST] {{ system_prompt }} + {{ user_prompt }} [/INST] {{ model_prompt }}</s>[INST] {{ user_prompt }} [/INST]
        
        Returns:
            str: prompt
        """
        return f"{self.START}{self.INST_START} {self.personality} {self.history}"
