from typing import List, Tuple, Iterable, Iterator, Dict
from .kb import KnowledgeBase
import json

class KBLoader:
    """KnowledgeBase DataLoader 
    
    Args:
        kb (List[Dict]): knowledge base 
        batch_size (int): batch size
    """
    
    def __init__(self, kb: KnowledgeBase, batch_size: int):
        assert batch_size > 0, "Batch size must be >0"
        
        self.kb = kb
        self.batch_size = batch_size
        self.num_iters = len(self.kb) // self.batch_size + 1 if (len(self.kb) % self.batch_size)>0 else 0
    
    def __iter__(self) -> Dict:
        for i in range(1, self.num_iters+1):    
            
            start = (i-1)*self.batch_size
            stop = len(self.kb) if i == self.num_iters else i*self.batch_size
            batch = [self.kb[i] for i in range(start, stop)]                              
            yield batch
            
    def __len__(self) -> int:
        return self.num_iters