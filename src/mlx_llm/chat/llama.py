from typing import List, Dict
from .base import Chat

class LLaMAChat(Chat):
    """LLaMAChat

    Args:
        personality (str): LLaMA personality (a description of what personality model has)
        examples (List[Dict[str, str]]): a list of examples of dialog [{"question": ..., "answer": ...}]
        end_str (str, optional): end of the model answer. Defaults to "[/INST]".
    """
    
    def __init__(self, personality: str, examples: List[Dict[str, str]], end_str: str = "[INST]"):
        
        super().__init__(personality, examples, end_str)
        self.SYS_START = "<<SYS>>"
        self.SYS_END = "<</SYS>>"
        self.INST_START = "[INST]"
        self.INST_END = "[/INST]"
    
    @property   
    def history(self) -> str:
        """Dialog history

        Returns:
            str: dialog history in LLaMA format
        """
        prompt = ""
        for i, example in enumerate(self.examples):
            prompt += example["user"]  + " " + self.INST_END
            if example["model"] is not None:
                prompt += " " + example["model"] + " " + self.INST_START + " "
        return prompt

    @property
    def prompt(self) -> str:
        """Retunr LLaMA prompt based on this structure
        [INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_msg_1 }} [/INST] {{ model_answer_1 }} [INST] {{ user_msg_2 }} [/INST]
        """
        return f"{self.INST_START} {self.SYS_START}\n{self.personality}\n{self.SYS_END}\n\n{self.history}"
 