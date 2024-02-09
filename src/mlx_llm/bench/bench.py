import gc
import os
import time
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pandas as pd
import psutil
from transformers import AutoTokenizer

from ..model import create_model, create_tokenizer

FAIL = "FAIL"


class Results:
    """Results class to store benchmarking results"""

    def __init__(self, apple_silicon: str, model_name: str, max_tokens: int = 100, temperature: float = 0.1):
        self.apple_silicon = apple_silicon
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_name = model_name
        self.datetime = time.strftime("%Y-%m-%d %H:%M:%S")
        self.mlx_version = mx.__version__
        self.data = {
            "datetime": [],
            "apple_silicon": [],
            "model_name": [],
            "max_tokens": [],
            "temperature": [],
            "generation_tps [token/sec]": [],
            "generation_time [s]": [],
            "memory_usage [MB/token]": [],
            "mlx_version": [],
        }

    def update(
        self,
        generation_tps: Optional[float] = None,
        generation_time: Optional[float] = None,
        memory_usage: Optional[float] = None,
        failed: bool = False,
    ) -> None:
        """Update results

        Args:
            generation_tps (Optional[float], optional): generation TPS. Defaults to None.
            generation_time (Optional[float], optional): generation time. Defaults to None.
            memory_usage (Optional[float], optional): memory usage. Defaults to None.
            failed (bool, optional): whether the test failed. Defaults to False.
        """
        self.data["datetime"].append(self.datetime)
        self.data["apple_silicon"].append(self.apple_silicon)
        self.data["model_name"].append(self.model_name)
        self.data["max_tokens"].append(self.max_tokens)
        self.data["temperature"].append(self.temperature)
        self.data["mlx_version"].append(self.mlx_version)
        if failed:
            self.data["generation_tps [token/sec]"].append(FAIL)
            self.data["generation_time [s]"].append(FAIL)
            self.data["memory_usage [MB/token]"].append(FAIL)
        else:
            self.data["generation_tps [token/sec]"].append(generation_tps)
            self.data["generation_time [s]"].append(generation_time)
            self.data["memory_usage [MB/token]"].append(memory_usage)
            self._log_results()

    def _log_results(self) -> None:
        """Log results"""
        generation_tps = self.data["generation_tps [token/sec]"][-1]
        generation_time = self.data["generation_time [s]"][-1]
        memory_usage = self.data["memory_usage [MB/token]"][-1]
        print(f"> [SUCCESS] Test on {self.data['model_name'][-1]} completed. Results recap:")
        print(f"\t> Generation TPS: {generation_tps:.2f} token/sec")
        print(f"\t> Generation time: {generation_time:.2f} s")
        print(f"\t> Memory usage: {memory_usage:.2f} MB")

    def save(self, output_dir: str) -> None:
        """Save results to a CSV file

        Args:
            output_dir (str): output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "benchmark.csv")
        if os.path.exists(output_path):
            print(f"> Results file {output_path} already exists. Appending results...")
            df = pd.read_csv(output_path)
            df = pd.concat([df, pd.DataFrame(self.data)], ignore_index=True)
        else:
            df = pd.DataFrame(self.data)
            output_path = os.path.join(output_dir, "benchmark.csv")
        df.to_csv(output_path, index=False)
        print(f"> Results saved to {output_path}")


class Benchmark:
    """LLM Benchmarking class

    Args:
        apple_silicon (str): Apple silicon version (e.g. m1_pro, m1_max, etc.)
        model_name (str): model name to save benchmarking results
        prompt (str, optional): prompt for the model. Defaults to "What is the meaning of life?".
        max_tokens (int, optional): maximum tokens to generate. Defaults to 100.
        temperature (float, optional): temperature for generation. Defaults to 0.1.
        verbose (bool, optional): whether to print verbose output. Defaults to False.
    """

    def __init__(
        self,
        apple_silicon: str,
        model_name: str,
        prompt: str = "What is the meaning of life?",
        max_tokens: int = 100,
        temperature: float = 0.1,
        verbose: bool = False,
    ) -> None:
        self.apple_silicon = apple_silicon
        self.model_name = model_name
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.results = Results(
            apple_silicon=apple_silicon, model_name=model_name, max_tokens=max_tokens, temperature=temperature
        )
        self.verbose = verbose

    def _generate(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
    ) -> Tuple[float, float, float]:
        """Generate text using a given model and tokenizer and compute TPS, generation time, memory usage

        Args:
            model (nn.Module): LLM model
            tokenizer (AutoTokenizer): tokenizer
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            Tuple[float, float, float]: generation TPS, generation time, avg memory usage
        """
        process = psutil.Process(os.getpid())
        # generate answer
        x = mx.array([tokenizer.bos_token_id] + tokenizer.encode(self.prompt))
        prompt_size = x.size

        tic = time.perf_counter()
        skip = 0
        tokens = []
        memory_usage = []
        if self.verbose:
            print("\n ======= \n")
            print(self.prompt, end="", flush=True)
        for i, token in enumerate(model.generate(x, self.temperature)):
            if i == 0:
                prompt_time = time.perf_counter() - tic
                tic = time.perf_counter()
            tokens.append(token.item())  # actually compute the token
            memory_usage.append(process.memory_info().rss / 1024 / 1024)  # Convert to MB
            if len(tokens) >= self.max_tokens:
                break
            if self.verbose:
                token_list = list(tokens)
                if token_list[-1] == tokenizer.eos_token_id:
                    break
                answer = tokenizer.decode(token_list)
                print(answer[skip:], end="", flush=True)
                skip = len(answer)

        if self.verbose:
            print("\n ======= \n")

        # print stats
        token_count = len(tokens)
        generation_time = time.perf_counter() - tic
        if token_count == 0:
            print("No tokens generated for this prompt")
            return 0.0, 0.0
        prompt_size / prompt_time
        generation_tps = (token_count - 1) / generation_time
        memory_usage = np.mean(memory_usage)

        return generation_tps, generation_time, memory_usage

    def start(self) -> None:
        """Run benchmarking tests for each model"""
        try:
            print(f"\n> Running test for {self.model_name}")

            model = create_model(self.model_name)

            tokenizer = create_tokenizer(self.model_name)

            generation_tps, generation_time, memory_usage = self._generate(model, tokenizer)

            self.results.update(
                generation_tps=generation_tps, generation_time=generation_time, memory_usage=memory_usage
            )
        except Exception as e:
            print(f"> [ERROR] Failed test on {self.model_name} - error {e}")
            self.results.update(failed=True)

        del model
        gc.collect()

    def save(self, output_dir: str) -> None:
        """Save results to a CSV file

        Args:
            output_path (str): path to save the results
        """
        self.results.save(output_dir=output_dir)
