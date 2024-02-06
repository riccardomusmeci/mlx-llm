from typing import Dict, List


class Chat:
    """BaseChat

    Args:
        personality (str): LLaMA personality (a description of what personality model has)
        examples (List[Dict[str, str]]): a list of examples of dialog [{"question": ..., "answer": ...}]
        end_str (str, optional): end of the model answer. Defaults to "</s>".
    """

    def __init__(self, personality: str, examples: List[Dict[str, str]], end_str: str = "</s>"):
        self.personality = personality
        self.base_examples = self._load_examples(examples)
        self.examples = self.base_examples.copy()
        self.END_STR = end_str

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

    def add_question(self, question: str) -> None:
        """Add question to dialog

        Args:
            question (str): dialog question
        """
        self.examples.append({"user": question, "model": None})

    def add_answer(self, answer: str) -> None:
        """Add answer to dialog

        Args:
            answer (str): dialog answer
        """
        if self.END_STR in answer:
            answer = answer.replace(self.END_STR, "")
        self.examples[-1]["model"] = answer

    @property
    def history(self) -> str:
        """Dialog history

        Returns:
            str: dialog history in LLaMA format
        """
        raise NotImplementedError("history property must be implemented in child class")

    @property
    def prompt(self) -> str:  # noqa: D102
        raise NotImplementedError("prompt property must be implemented in child class")

    def reset(self) -> None:
        """Reset dialog but not personality"""
        self.examples = self.base_examples.copy()

    def model_status(self, answer: str) -> int:
        """Check if dialog is over

        Returns:
            int: -1 if model can keep generating, 0 if model is generating self.END_STR, 1 if model is done generating self.END_STR
        """
        for i in range(len(self.END_STR) - 1, 0, -1):
            # model is generating self.END_STR -> wait until it finishes but don't print anything
            if answer[-i:] == self.END_STR[:i]:
                return 0
        # model is done generating self.END_STR -> saving answer
        if answer[-len(self.END_STR) :] == self.END_STR:
            return 1
        # model can keep generating
        return -1
