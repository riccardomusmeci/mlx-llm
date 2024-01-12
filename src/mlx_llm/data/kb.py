from transformers import AutoTokenizer
from typing import List, Tuple, Iterable, Iterator, Dict
import json

class KnowledgeBase:
    """KnowledgeBase Dataset
    
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
    
    def __init__(self, kb_path: str, tokenizer: AutoTokenizer, max_length: int):

        self.kb = json.load(open(kb_path, "r"))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx: int) -> Tuple[List[int], Dict]:
        
        text = self.kb[idx]["content"]
        tokens = self.tokenizer(
            text, 
            max_length=self.max_length - 1, 
            return_attention_mask=False, 
            padding=False, 
            truncation=True
        )
        tokens['input_ids'] += [self.tokenizer.eos_token_id]
        self.kb[idx]["input_ids"] = tokens['input_ids']
        return self.kb[idx]
    
    def __len__(self) -> int:
        return len(self.kb)
