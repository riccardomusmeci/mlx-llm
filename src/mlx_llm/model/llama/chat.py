from typing import List, Dict

class LLaMAChat:
    """LLaMAChat

    Args:
        personality (str): LLaMA personality (a description of what personality model has)
        examples (List[Dict[str, str]]): a list of examples of dialog [{"question": ..., "answer": ...}]
    """
    
    def __init__(self, personality: str, examples: List[Dict[str, str]]):
        
        self.personality = personality
        self.base_examples = self._load_examples(examples)
        self.examples = self.base_examples.copy()
        
        self.SYS_START = "<<SYS>>"
        self.SYS_END = "<</SYS>>"
        self.INST_START = "[INST]"
        self.INST_END = "[/INST]"
        
    def _load_examples(self, examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Load valid examples from ones provided

        Args:
            examples (List[Dict[str, str]]): a list of examples of dialog [{"user": ..., "model": ...}]

        Returns:
            List[Dict[str, str]]: a list of valid examples of dialog [{"user": ..., "model": ...}]
        """
        valid = []
        for example in examples:
            keys = list(example.keys())
            if "user" not in keys or "model" not in keys:
                print(f"Some key is missing (must have user/model keys). Found only: {keys}")
            else:
                valid.append(example)
        return valid
        
    def add_question(self, question: str):
        """Add question to dialog

        Args:
            question (str): dialog question
        """
        self.examples.append({"user": question, "model": None})
        
    def add_answer(self, answer: str):
        """Add answer to dialog

        Args:
            answer (str): dialog answer
        """
        if self.INST_START in answer:
            answer = answer.replace(self.INST_START, "")
        self.examples[-1]["model"] = answer
    
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
         
    def reset(self):
        """Reset dialog but not personality
        """
        self.examples = self.base_examples.copy()
    
    def model_status(self, answer: str) -> int:
        """Check if dialog is over

        Returns:
            int: -1 if model can keep generating, 0 if model is generating [INST], 1 if model is done generating [INST]
        """
        for i in range(len(self.INST_START) - 1, 0, -1):
            # model is generating [INST] -> wait until it finishes but don't print anything
            if answer[-i:] == self.INST_START[:i]:
                return 0
        # model is done generating [INST] -> saving answer
        if answer[-len(self.INST_START) :] == self.INST_START:
            return 1
        # model can keep generating
        return -1
 