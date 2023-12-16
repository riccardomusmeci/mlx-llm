from .model import Transformer
from typing import List, Tuple, Union, Optional, Dict
from pathlib import Path
from .config import llama_config
import os
import mlx.core as mx
from mlx.utils import tree_unflatten
from ...utils import Timing
from sentencepiece import SentencePieceProcessor
import sys 
from ..llm.base import BaseLLM
import numpy as np
import torch
from itertools import starmap

from .chat import LLaMAChat
 
class LLaMA(BaseLLM): 
    
    @staticmethod
    def build(
        model_name: str,
        weights_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        personality: str = "",
        examples: List[Dict[str, str]] = [],
    ):
        """Builds a LLaMA model from a given model name, weights path and tokenizer path.

        Args:
            model_name (str): LLaMA model name
            weights_path (Union[str, Path]): path to mlx weights
            tokenizer_path (Union[str, Path]): path to tokenizer
            personality (str, optional): LLaMA personality for chat mode. Defaults to "".
            examples (List[Dict[str, str]], optional): LLaMA examples (list of {"user": ..., "model": ...} examples) for chat mode. Defaults to [].

        Returns:
            LLaMA: LLaMA class instance with model and tokenizer
        """
        
        assert model_name in llama_config.keys(), f"Model name {model_name} not found in llama_config. Available models are {list(llama_config.keys())}"
        assert os.path.exists(weights_path), f"Weights path {weights_path} does not exist."
        assert os.path.exists(tokenizer_path), f"Tokenizer path {tokenizer_path} does not exist."
        
        print(f"************ Building LLaMA ************")
        with Timing(f"> Loading weights from {weights_path}"):
            weights = mx.load(weights_path)
        
        with Timing(f"> Creating {model_name}"):
            params = llama_config[model_name]
            model = Transformer(**params)
            
        with Timing("> Loading weights into model"):
            model.update(tree_unflatten(list(weights.items())))
            mx.eval(model.parameters()) 
            
        with Timing("> Loading tokenizer"):
            tokenizer = SentencePieceProcessor(tokenizer_path)
            
        print("\n" + "-"*20 + "\n")
               
        return LLaMA(model, tokenizer, personality, examples) 
    
    @staticmethod
    def convert(
        ckpt_path: Union[str, Path],
        output_path: Union[str, Path],
    ):
        """Convert a PyTorch checkpoint to a MLX checkpoint.

        Args:
            ckpt_path (Union[str, Path]): path to PyTorch checkpoint
            output_path (Union[str, Path]): path to output MLX checkpoint
        """
        def map_torch_to_mlx(key: str, value: torch.Tensor):
            """Maps a PyTorch key to a MLX key.
            
            Args:
                key (str): PyTorch key
                value (torch.Tensor): PyTorch value
            """
            if "tok_embedding" in key:
                key = "embedding.weight"

            elif "norm" in key:
                key = key.replace("attention_norm", "norm1").replace("ffn_norm", "norm2")

            elif "wq" in key or "wk" in key or "wv" in key or "wo" in key:
                key = key.replace("wq", "query_proj")
                key = key.replace("wk", "key_proj")
                key = key.replace("wv", "value_proj")
                key = key.replace("wo", "out_proj")

            elif "w1" in key or "w2" in key or "w3" in key:
                # The FFN is a separate submodule in PyTorch
                key = key.replace("feed_forward.w1", "linear1")
                key = key.replace("feed_forward.w3", "linear2")
                key = key.replace("feed_forward.w2", "linear3")

            elif "output" in key:
                key = key.replace("output", "out_proj")

            elif "rope" in key:
                return None, None

            return (
                key,
                value.numpy()
                if value.dtype != torch.bfloat16
                else value.to(torch.float16).numpy(),
            )
        
        with Timing(f"> Loading checkpoint from {ckpt_path}"):
            state = torch.load(ckpt_path, map_location="cpu")
        
        with Timing(f"> Saving weights to {output_path}"):
            np.savez(
                output_path,
                **{k: v for k, v in starmap(map_torch_to_mlx, state.items()) if k is not None}
            )
    
    def generate(
        self,
        write_every: int = 1, 
        max_tokens: int = 100,
        temp: float = 1.0,
    ):
        """Generate text from a given prompt.

        Args:
            write_every (int, optional): write every n tokens. Defaults to 1.
            max_tokens (int, optional): maximum number of tokens to generate. Defaults to 100.
            temp (float, optional): temperature. Defaults to 1.0.
        """
        print("************ LLaMA Text Completion ************\n")
        prompt = input("> prompt: ")
        
        x = mx.array([[self.tokenizer.bos_id()] + self.tokenizer.encode(prompt)])
        skip = x.shape[0]
        tokens = []
        for token in self.model.generate(x, temp):
            tokens.append(token)
            if len(tokens) == 1:
                # Actually perform the computation to measure the prompt processing time
                mx.eval(token)
            if len(tokens) >= max_tokens:
                break
            elif (len(tokens) % write_every) == 0:
                # It is perfectly ok to eval things we have already eval-ed.
                mx.eval(tokens)
                s = self.tokenizer.decode([t.item() for t in tokens])
                print(s[skip:], end="", flush=True)
                skip = len(s)
                if token.item() == self.tokenizer.eos_id():
                    print("End token: ", self.tokenizer.eos_id())
                    break
    
    def chat(self, temp: float = .1,  max_tokens: int = 1000):
        """Chat with the model.

        Args:
            temp (float, optional): model temperature. Defaults to .1.
            max_tokens (int, optional): max tokens to generate. Defaults to 1000.
        """
        # Start chat with a personality 
        chat = LLaMAChat(self.personality, self.examples)
        
        print("************ LLaMA Chat ('q' to quit, 'r' to reset) ************\n")
        
        while True:        
            question = input("\nUser: ")
            if question == "q": quit()
            if question == "r": 
                chat.reset()
                continue
            
            # adding question to dialog and getting prompt to model
            chat.add_question(question)
            prompt = chat.prompt
            
            x = mx.array([[self.tokenizer.bos_id()] + self.tokenizer.encode(prompt)])
            tokens = []
            skip = 0
            print("Model: ", end="", flush=True)
            for token in self.model.generate(x, temp):
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
            
        