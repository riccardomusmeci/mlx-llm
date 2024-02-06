import json
from typing import Dict, Iterable, Iterator, List, Tuple

from transformers import AutoTokenizer


class KnowledgeBase:
    """RAGKnowledgeBase dataset

    Dataset must be structures as follows:
    ```
    [
        {
            "content": "This is the first document",
            "metadata": {
                "source": "source1",
                "page": 1
        }
    ]

    ```

    Args:
        kb_path (str): path to knowledge base
        tokenizer (AutoTokenizer): tokenizer
        max_length (int): max length of the input sequence

    """

    def __init__(self, kb_path: str, limit: int = None):
        self.kb = json.load(open(kb_path, "r"))
        self.limit = limit
        if self.limit:
            self.kb = self.kb[: self.limit]

    def __getitem__(self, idx: int) -> Tuple[List[int], Dict]:
        return self.kb[idx]

    def __len__(self) -> int:
        return len(self.kb)


class KnowledgeBaseLoader:
    """KnowledgeBase DataLoader

    Args:
        kb (RAGKnowledgeBase): RAGKnowledgeBase instance
        batch_size (int): batch size
    """

    def __init__(self, kb: KnowledgeBase, batch_size: int):
        assert batch_size > 0, "Batch size must be >0"

        self.kb = kb
        self.batch_size = batch_size
        iter_offset = 1 if (len(self.kb) % self.batch_size) > 0 else 0
        self.num_iters = len(self.kb) // self.batch_size + iter_offset

    def __iter__(self) -> Dict:
        for i in range(1, self.num_iters + 1):
            start = (i - 1) * self.batch_size
            stop = len(self.kb) if i == self.num_iters else i * self.batch_size
            batch = [self.kb[i] for i in range(start, stop)]
            yield batch

    def __len__(self) -> int:
        return self.num_iters
