from typing import List, Optional

from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer


class Tokenizer:
    """A simple tokenizer class that is able to load both HF tokenizers and SentencePiece tokenizers.
    It performes easy encoding. For facny onece, go with custom HF tokenizers.

    Args:
        tokenizer (str): path to tokenizer / HF tokenizer name
        return_tensors (Optional[str], optional): return tensors type. Defaults to "np".
        return_attention_mask (Optional[bool], optional): return attention mask. Defaults to False.
    """

    def __init__(
        self,
        tokenizer: str,
        return_tensors: Optional[str] = "np",
        return_attention_mask: Optional[bool] = False,
    ) -> None:
        # loading tokenizer
        if tokenizer.endswith(".model"):
            self.tokenizer = SentencePieceProcessor(tokenizer)
            self.from_hf = False
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
            self.from_hf = True

        self.return_tensors = return_tensors
        self.return_attention_mask = return_attention_mask

    def encode(self, text: str) -> List[int]:
        """Encode a text to a list of token ids.

        Args:
            text (str): text to encode

        Returns:
            List[int]: list of token ids
        """
        if self.from_hf:
            tokens = self.tokenizer(
                text,
                return_tensors=self.return_tensors,
                return_attention_mask=self.return_attention_mask,
            )["input_ids"]
            return tokens[0].tolist()
        else:
            return self.tokenizer.encode(text)

    def bos_id(self) -> int:
        """Get the beginning of sentence token id.

        Returns:
            int: bos token id
        """
        if self.from_hf:
            return self.tokenizer.bos_token_id
        else:
            return self.tokenizer.bos_id()

    def eos_id(self) -> int:
        """Get the end of sentence token id.

        Returns:
            int: eos token id
        """
        if self.from_hf:
            return self.tokenizer.eos_token_id
        else:
            return self.tokenizer.eos_id()

    def vocab_size(self) -> int:
        """Get the vocabulary size.

        Returns:
            int: vocabulary size
        """
        if self.from_hf:
            return self.tokenizer.vocab_size
        else:
            return self.tokenizer.vocab_size()

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token ids to a text.

        Args:
            ids (List[int]): list of token ids

        Returns:
            str: decoded text
        """
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> List[int]:
        """Encode a text to a list of token ids.

        Args:
            text (str): text to encode

        Returns:
            List[int]: list of token ids
        """
        return self.encode(text)
