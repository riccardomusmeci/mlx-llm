from typing import List, Dict
from .base import Chat

class Phi2Chat(Chat):
    """Mistral chat class

    Args:
        personality (str): Mistral personality (a description of what personality model has)
        examples (List[Dict[str, str]]): a list of examples of dialog [{"question": ..., "answer": ...}]
        end_str (str, optional): end of the model answer. Defaults to "</s>".
    """
    def __init__(self, personality: str, examples: List[Dict[str, str]], end_str: str = "Istruct: "):
        
        super().__init__(personality, examples, end_str)
        """
        [INST] {{ system_prompt }} + {{ user_prompt }} [/INST] {{ model_prompt }}</s>[INST] {{ user_prompt }} [/INST
        """
        self.INSTRUCT = "Instruct: "
        self.MODEL = "Output: "
        
    @property
    def history(self) -> str:
        """Chat history
        
        {{ user_prompt }} [/INST] {{ model_prompt }}</s>[INST] {{ user_prompt }} [/INST]

        Returns:
            str: history
        """
        prompt = ""
        for example in self.examples:
            prompt += example["user"] + "\n" + self.MODEL
            if example["model"] is not None:
                prompt += example["model"] + "\n" + self.INSTRUCT
        return prompt
    
    @property
    def prompt(self) -> str:
        """Return Mistral prompt based on this structure
        
        [INST] {{ system_prompt }} + {{ user_prompt }} [/INST] {{ model_prompt }}</s>[INST] {{ user_prompt }} [/INST]
        
        Returns:
            str: prompt
        """
        return f"{self.INSTRUCT}{self.personality} {self.history}"
