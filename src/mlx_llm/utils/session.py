from typing import Dict, List, Optional, Tuple


class Session:
    """Session class.

    Args:
        questions (List[str], optional): questions. Defaults to [].
        answers (List[str], optional): answers. Defaults to [].
    """

    def __init__(
        self, questions: Optional[List[str]] = None, answers: Optional[List[str]] = None
    ) -> None:  # noqa: D107  # noqa: B006
        questions = questions or []
        answers = answers or []

        self.questions, self.answers = self._load(questions, answers)
        self.q_reset_len = len(self.questions)
        self.a_reset_len = len(self.answers)

    def _load(self, questions: List[str], answers: List[str]) -> Tuple[List[str], List[str]]:
        """Load questions and answers.

        Args:
            questions (List[str]): questions
            answers (List[str]): answers

        Returns:
            Tuple[List[str], List[str]]: valid questions and answers
        """
        assert isinstance(questions, list), f"questions must be a list. Got {type(questions)} instead."

        assert isinstance(answers, list), f"answers must be a list. Got {type(answers)} instead."

        if len(questions) < len(answers):
            print(
                f"[WARNING] Found {len(answers) - len(questions)} more answers than questions. Extra {len(answers) - len(questions)} answers will be discarded."
            )
            up_to = len(questions)
        elif len(answers) < len(questions) - 1:
            print(
                f"[WARNING] Found {len(questions) - len(answers)} more questions than answers. Extra {len(questions) - len(answers) - 1} questions will be discarded."
            )
            up_to = len(answers) + 1
        else:
            up_to = len(questions)

        questions = questions[:up_to]
        answers = answers[: up_to - 1] if len(answers) == up_to - 1 else answers[:up_to]

        return questions, answers

    def reset(self) -> None:
        """Reset conversation."""
        self.questions = self.questions[: self.q_reset_len]
        self.answers = self.answers[: self.a_reset_len]

    def add_question(self, question: str) -> None:
        """Add question to conversation.

        Args:
            question (str): question

        Raises:
            ValueError: if there is nos answer for the previous question
        """
        if len(self.questions) != len(self.answers):
            raise ValueError("Cannot add question. There is no answer for the previous question.")
        self.questions.append(question)

    def add_answer(self, answer: str) -> None:
        """Add answer to conversation.

        Args:
            answer (str): answer

        Raises:
            ValueError: if there is no question for the answer
        """
        if len(self.questions) == len(self.answers):
            raise ValueError("Cannot add answer. There is no question for the answer.")
        self.answers.append(answer)

    @property
    def history(self) -> List[Dict[str, Optional[str]]]:
        """Conversation history.

        Returns:
            List[Dict[str, Optional[str]]]: session history in the form of a list of dictionaries with keys "question" and "answer"
        """
        history = []
        for i in range(len(self.questions)):
            q = self.questions[i]
            a = self.answers[i] if i < len(self.answers) else None
            history.append({"question": q, "answer": a})
        return history

    def to_string(self, last_k: int = 0) -> str:
        """String representation of conversation.

        Args:
            last_k (int): number of questions and answers to consider. If last_k==0, all questions and answers are considered. Defaults to 0.

        Returns:
            str: conversation
        """

        assert isinstance(last_k, int), f"last_k must be an int. Got {type(last_k)} instead."

        if last_k > len(self.history):
            print(
                f"[WARNING] last_k={last_k} is greater than the number of answers ({len(self.history)}). last_k will be set to {len(self.history)}."
            )
            last_k = 0  # all history

        _history = self.history[-last_k:]
        out = ""
        for qa in _history:
            out += f"Question: {qa['question']}\n"
            if qa["answer"] is not None:
                out += f"Answer: {qa['answer']}\n"
        return out

    def __len__(self) -> int:
        """Length of conversation.

        Returns:
            int: length of conversation
        """
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, Optional[str]]:
        """Get item from conversation.

        Args:
            idx (int): index

        Returns:
            Dict[str, Optional[str]]: item
        """
        return self.history[idx]
