from .model import Transformer
from typing import Union, List, Dict, Optional
from pathlib import Path
import os
from ..utils import Timing
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_unflatten
from sentencepiece import SentencePieceProcessor
from safetensors import safe_open
import torch
import numpy as np
from tqdm import tqdm

from .config import CONFIG
from ..chat import create_chat

class LLM:
    """LLM class

    Args:
        model (Transformer): a Transformer model
        tokenizer: SentencePieceProcessor tokenizer
        personality (str, optional): model personality (a description of what personality model has). Defaults to "".
        examples (List[Dict[str, str]], optional): a list of examples of dialog [{"user": ..., "model": ...}]. Defaults to [].
        model_name (str, optional): model name. Defaults to "".
    """
    def __init__(
        self, 
        model: Transformer, 
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
        weights_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        personality: str = "",
        examples: List[Dict[str, str]] = [],
        no_rope: bool = True
    ):
        """Build an LLM model from a given model name, weights path and tokenizer path.

        Args:
            model_name (str): Mistral model name
            weights_path (Union[str, Path]): path to mlx weights
            tokenizer_path (Union[str, Path]): path to tokenizer
            personality (str, optional): Mistral personality for chat mode. Defaults to "".
            examples (List[Dict[str, str]], optional): Mistral examples (list of {"user": ..., "model": ...} examples) for chat mode. Defaults to [].
        
        Returns:
            LLM: LLM class instance with model and tokenizer
        """
        
        assert model_name in CONFIG.keys(), f"Model name {model_name} not found in CONFIG. Available models are {list(CONFIG.keys())}"
        assert os.path.exists(weights_path), f"Weights path {weights_path} does not exist."
        assert os.path.exists(tokenizer_path), f"Tokenizer path {tokenizer_path} does not exist."
        
        print(f"************ Building LLM ({model_name}) ************")
            
        with Timing("> Loading weights"):
            model = Transformer(**CONFIG[model_name])
            weights = mx.load(weights_path)
            weights = tree_unflatten(list(weights.items()))
            weights = tree_map(lambda p: p.astype(mx.float16), weights)
            model.update(weights)
            
        with Timing("> Loading tokenizer"):
            tokenizer = SentencePieceProcessor(tokenizer_path)
            
        print("\n" + "-"*20 + "\n")
    
        return LLM(model, tokenizer, personality, examples, model_name=model_name)
    

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
            logits, cache = self.model(y[:, None], cache)
            y = sample(logits.squeeze(1))
            yield y

    
    def chat(self, temp: float = .1,  max_tokens: int = 1000):
        """Chat with model

        Args:
            temp (float, optional): model temperature. Defaults to .1.
            max_tokens (int, optional): max number of tokens to generate. Defaults to 1000.
        """
        
        chat = create_chat(self.model_name, self.personality, self.examples)
        
        print("************ Mistral Chat ('q' to quit, 'r' to reset) ************\n")
        
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
                answer = self.tokenizer.decode(token_list)
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
            
        
            
            