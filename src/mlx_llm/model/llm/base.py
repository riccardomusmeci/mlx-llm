from typing import Dict, List

class Chat:
    
    def __init__(self, personality: str, examples: List[Dict[str, str]]):
        self.personality = personality
        self.base_examples = self._load_examples(examples)
        self.examples = self.base_examples.copy()
    
    def add_question(self, question: str):
        """Add question to dialog

        Args:
            question (str): dialog question
        """
        self.examples.append({"user": question, "model": None})
    
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
    
    def add_answer(self, answer: str):
        """Add answer to dialog

        Args:
            answer (str): dialog answer
        """
        self.examples[-1]["model"] = answer
        


class BaseLLM:
    
    def __init__(self, model, tokenizer, personality, examples):
        self.model = model
        self.tokenizer = tokenizer
        self.personality = personality
        self.examples = examples
        
    @staticmethod
    def build():
        raise NotImplementedError()
    
    @staticmethod
    def convert():
        raise NotImplementedError()
    
    def generate(self):
        raise NotImplementedError()
    
    def chat(self):
        raise NotImplementedError()
    
    