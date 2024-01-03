from typing import List, Dict
from .base import Chat

class TinyLLaMAChat(Chat):
    """TinyLLaMAChat

    Args:
        personality (str): TinyLLaMA personality (a description of what personality model has)
        examples (List[Dict[str, str]]): a list of examples of dialog [{"question": ..., "answer": ...}]
        end_str (str, optional): end of the model answer. Defaults to "</s>\n<|user|>".
    """
    
    def __init__(self, personality: str, examples: List[Dict[str, str]], end_str: str = "</s>\n<|user|>"):
        
        """
        <|system|>
        {{ personality }}</s>
        <|user|>
        {{ user_prompt }}</s>
        <|assistant|>
        {{ model_prompt }}</s>
        """
        
        super().__init__(personality, examples, end_str)
        self.SYS = "<|system|>"
        self.END = "</s>"
        self.USER = "<|user|>"
        self.ASSISTANT = "<|assistant|>"
        
    @property
    def history(self) -> str:
        """Dialog history

        Returns:
            str: dialog history in TinyLLaMA format
        """
        prompt = ""
        for i, example in enumerate(self.examples):
            prompt += self.USER + "\n" + example["user"] + self.END + "\n" + self.ASSISTANT + "\n"
            if example["model"] is not None:
                prompt += example["model"] + self.END + "\n"
        return prompt

    @property
    def prompt(self) -> str:
        """Return TinyLLaMA prompt based on this structure
       
        <|system|>
        {{ personality }}</s>
        <|user|>
        {{ user_prompt }}</s>
        <|assistant|>
        {{ model_prompt }}</s>
        
        """
        return f"{self.SYS}\n{self.personality}{self.END}\n{self.history}"
 