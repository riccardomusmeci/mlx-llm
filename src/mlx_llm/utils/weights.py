import os
from pathlib import Path
from typing import Dict, Generator, List, Union

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from .time import Timing


def save_weights(weights: Dict[str, mx.array], output_path: str) -> None:
    """Save MLX weights to a given path.

    Args:
        weights (Dict[str, mx.array]): MLX weights dict (key, mx.array)
        output_path (str): path to save weights
    """
    output_dir = os.path.dirname(output_path)
    if len(output_dir) > 0:
        os.makedirs(output_dir, exist_ok=True)
    if output_path.endswith(".safetensors"):
        mx.save_safetensors(output_path, weights, metadata={"format": "mlx"})
    else:
        mx.savez(output_path, **weights)


def smart_load(ckpt_paths: Union[List[str], str]) -> Generator:
    """Load a checkpoint from a list of paths.

    Args:
        ckpt_paths (Union[List[str], str]): path to checkpoint or multiple paths

    Yields:
        Generator: a generator of state_dict
    """

    if isinstance(ckpt_paths, str):
        ckpt_paths = [ckpt_paths]

    for ckpt_path in ckpt_paths:
        if ckpt_path.endswith(".safetensors"):
            state_dict = mx.load(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        yield state_dict


def weights_to_mlx(
    ckpt_paths: Union[List[str], str],
    show_kv: bool = False,
) -> Dict[str, mx.array]:
    """Convert a checkpoint from PyTorch or safetensors to a MLX weights (Dict[str, mx.array]).

    Args:
        ckpt_paths (Union[List[str], str]): path to checkpoint
        show_kv (bool, optional): whether to show key-value pairs. Defaults to False.

    Returns:
        Dict[str, mx.array]: MLX weights dict (key, mx.array)
    """
    weights = {}
    for state_dict in smart_load(ckpt_paths):
        for k, w in state_dict.items():
            if show_kv:
                print(k, w.shape)
            if isinstance(w, torch.Tensor):
                w = w.to(torch.float16).numpy()
                w = mx.array(w)
            weights[k] = w
    return weights
