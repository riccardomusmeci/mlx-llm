# from .model import Transformer
from typing import Union, List, Dict, Optional
from pathlib import Path
import os
from ...utils import Timing
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_unflatten
from safetensors import safe_open
import torch
import numpy as np
from tqdm import tqdm

from ...model import create_model
from .tokenizer import Tokenizer
from .template import create_chat


class ChatLLM:
    """LLM class

    Args:
        model (Transformer): a Transformer model
        tokenizer: tokenizer
        personality (str, optional): model personality (a description of what personality model has). Defaults to "".
        examples (List[Dict[str, str]], optional): a list of examples of dialog [{"user": ..., "model": ...}]. Defaults to [].
        model_name (str, optional): model name. Defaults to "".
    """
    def __init__(
        self, 
        model, 
        tokenizer, 
        personality: str = "", 
        examples: List[Dict[str, str]] = [], 
        model_name: str = ""
    ):
        
        self.model = model
        self.tokenizer = tokenizer
        self.personality = personality
        self.examples = examples
        self.model_name = model_name
        
    
    @staticmethod
    def build(
        model_name: str,
        tokenizer: str,
        personality: str = "",
        examples: List[Dict[str, str]] = [],
        weights: Union[str, bool] = True,
    ):
        """Build an LLM model from a given model name, weights path and tokenizer path.

        Args:
            model_name (str): Mistral model name
            tokenizer (str): path to tokenizer
            personality (str, optional): Mistral personality for chat mode. Defaults to "".
            examples (List[Dict[str, str]], optional): Mistral examples (list of {"user": ..., "model": ...} examples) for chat mode. Defaults to [].
            weights (Union[str, bool], optional): if True, load pretrained weights from HF. If str, load weights from the given path. Defaults to True.
        
        Returns:
            LLM: LLM class instance with model and tokenizer
        """    
        print(f"************ Building LLM ({model_name}) ************")
        
        tokenizer = Tokenizer(tokenizer)
        model = create_model(model_name, weights=weights)

        print("\n" + "-"*20 + "\n")
    
        return ChatLLM(model, tokenizer, personality, examples, model_name=model_name)
    

    def generate(self, x: mx.array, temp: Optional[float] = 0.0):
        """Generate tokens from a given input

        Args:
            x (mx.array): input tokens
            temp (Optional[float], optional): model temperature. Defaults to 0.0.
        """
        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))
        logits, cache = self.model(x[None])
        y = sample(logits[:, -1, :])
        yield y

        while True:
            logits, cache = self.model(y[:, None], cache=cache)
            y = sample(logits.squeeze(1))
            yield y

    
    def run(self, temp: float = .1,  max_tokens: int = 1000):
        """Chat with model

        Args:
            temp (float, optional): model temperature. Defaults to .1.
            max_tokens (int, optional): max number of tokens to generate. Defaults to 1000.
        """
        
        chat = create_chat(self.model_name, self.personality, self.examples)
        
        print("************ LLM Chat ('q' to quit, 'r' to reset) ************\n")
        
        while True:        
            question = input("\nUser: ")
            if question == "q": quit()
            if question == "r": 
                chat.reset()
                continue
            
            # adding question to dialog and getting prompt to model
            chat.add_question(question)
            prompt = chat.prompt
            
            x = mx.array([self.tokenizer.bos_id()] + self.tokenizer.encode(prompt))

            tokens = []
            skip = 0
            print("Model: ", end="", flush=True)
            
            for token in self.generate(x, temp):
                tokens.append(token)
                if len(tokens) >= max_tokens: 
                    break
                mx.eval(tokens)
                token_list = [t.item() for t in tokens]
                # tokenizer sometimes fails to decode - this fixes it (it's not fancy but it works)
                try:
                    answer = self.tokenizer.decode(token_list)
                except:
                    if token == self.tokenizer.vocab_size(): 
                        tokens[-1] = mx.array([self.tokenizer.eos_id()])
                        token_list[-1] = self.tokenizer.eos_id()
                # if answer is still prompt, continue
                status = chat.model_status(answer)

                if status == 0: continue
                if status == 1: 
                    skip = len(answer)
                    break
                print(answer[skip:], end="", flush=True)
                skip = len(answer)
                if token_list[-1] == self.tokenizer.eos_id(): 
                    break
            mx.eval(tokens)
            answer = self.tokenizer.decode([t.item() for t in tokens])
            chat.add_answer(answer)

            
            