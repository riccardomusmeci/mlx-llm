from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer
from typing import Optional, List

class ChatTokenizer:
    
    def __init__(
        self, 
        tokenizer,
        return_tensors: Optional[str] = "np",
        return_attention_mask: Optional[bool] = False,
    ):
        # loading tokenizer    
        if tokenizer.endswith(".model"):
            self.tokenizer = SentencePieceProcessor(tokenizer)
            self.from_hf = False
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
            self.from_hf = True
            
        self.return_tensors = return_tensors
        self.return_attention_mask = return_attention_mask
        
    def encode(self, text: str):
        if self.from_hf:
            tokens = self.tokenizer(
                text,
                return_tensors=self.return_tensors,
                return_attention_mask=self.return_attention_mask,
            )["input_ids"]
            return tokens[0].tolist()
        else:
            return self.tokenizer.encode(text)
    
    def bos_id(self):
        if self.from_hf:
            return self.tokenizer.bos_token_id
        else:
            return self.tokenizer.bos_id()
    
    def eos_id(self):
        if self.from_hf:
            return self.tokenizer.eos_token_id
        else:
            return self.tokenizer.eos_id()
        
    def vocab_size(self):
        if self.from_hf:
            return self.tokenizer.vocab_size
        else:
            return self.tokenizer.vocab_size()
    
    def decode(self, ids: List[int]):
        return self.tokenizer.decode(ids)
    
    def __call__(self, text: str):
        return self.encode(text)