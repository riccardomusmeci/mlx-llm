from abc import ABC, abstractmethod

from ..utils.session import Session


class Prompt(ABC):
    TEXT_END: str = ""

    def __init__(self, system: str) -> None:
        self.system = system
        pass

    @abstractmethod
    def prepare(self, session: Session) -> str:
        pass
