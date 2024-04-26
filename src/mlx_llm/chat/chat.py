from dataclasses import dataclass
from typing import Dict, List, Optional

import mlx.core as mx

from mlx_llm.model import create_model, create_tokenizer, quantize
from mlx_llm.prompt import create_prompt
from mlx_llm.utils.session import Session

from .utils import GO, STOP, WAIT, answer_status


@dataclass
class ChatSetup:
    """Dataclass for chat setup

    Raises:
        ValueError: Each element in history must contain a question and an answer.
    """

    system: str
    history: Optional[List[Dict[str, str]]] = None

    def session(self) -> Session:
        """Create session

        Raises:
            ValueError: Each element in history must contain a question and an answer.

        Returns:
            Session: session
        """
        for el in self.history:  # type: ignore
            if "question" not in el.keys() or "answer" not in el.keys():
                raise ValueError("Each element in history must contain a question and an answer.")
        if self.history is None:
            return Session()
        else:
            return Session(
                questions=[el["question"] for el in self.history], answers=[el["answer"] for el in self.history]
            )


class LLMChat:
    """Chat with LLM class

    Args:
        model_name (str): model name
        chat_setup (ChatSetup): chat setup
        max_tokens (int, optional): max tokens. Defaults to 1024.
        temperature (float, optional): temperature. Defaults to 0.1.
        quantized (bool, optional): quantized. Defaults to False.
        group_size (int, optional): group size. Defaults to 64.
        bits (int, optional): bits. Defaults to 8.
    """

    def __init__(
        self,
        model_name: str,
        chat_setup: ChatSetup,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        quantized: bool = False,
        group_size: int = 64,
        bits: int = 8,
    ) -> None:
        self.model = create_model(model_name)
        self.tokenizer = create_tokenizer(model_name)
        self.prompt = create_prompt(model_name=model_name, system=chat_setup.system)
        self.session = chat_setup.session()
        self.max_tokens = max_tokens
        self.temperature = temperature

        if quantized:
            print(f"[INFO] Quantizing model {model_name} with group_size={group_size} and bits={bits}")
            self.model = quantize(self.model, group_size=group_size, bits=bits)

        self.model.eval()

    def start(self) -> None:
        """Start chat"""
        print("\n[INFO] Start chatting! Press 'q' to quit or 'r' to reset conversation.\n")

        print("******* CHAT *******")
        print(f"[SYSTEM] {self.prompt.system}")
        print(f"[HISTORY]\n{self.session.to_string()}[/HISTORY]")

        while True:
            user = input("User: ")
            if user == "q":
                break
            if user == "r":
                self.session.reset()
                continue

            self.session.add_question(user)
            prompt = self.prompt.prepare(session=self.session)

            x = mx.array([self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt))  # type: ignore
            tokens = []
            skip = 0
            print("LLM: ", end="", flush=True)
            for _, token in enumerate(self.model.generate(x, self.temperature)):
                tokens.append(token.item())  # actually compute the token
                # checking if we reached max tokens
                if len(tokens) >= self.max_tokens:
                    break

                token_list = list(tokens)
                # checking if we reached end of sentence
                if token_list[-1] == self.tokenizer.eos_token_id:
                    break

                answer = self.tokenizer.decode(token_list)

                # checking if we reached end of sentence for LLM
                if answer_status(answer, self.prompt.TEXT_END) == STOP:
                    break
                if answer_status(answer, self.prompt.TEXT_END) == WAIT:
                    continue

                print(answer[skip:], end="", flush=True)
                skip = len(answer)

            self.session.add_answer(answer)
            if not answer.endswith("\n"):
                print("\n")
