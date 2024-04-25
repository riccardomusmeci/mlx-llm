from abc import ABC
from ..utils.session import Session


class Prompt(ABC):
    TEXT_END = None

    def __init__(self, system: str) -> None:
        pass

    def prepare(self, session: Session) -> str:
        pass
