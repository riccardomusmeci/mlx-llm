import torch
import os
from .time import Timing
from pathlib import Path
from typing import Union, List
from tqdm import tqdm
import numpy as np
from safetensors import safe_open

def weights_to_npz(
    ckpt_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    show_kv: bool = False,
):
    """Convert a checkpoint of PyTorch or safetensors to a MLX checkpoint (npz file).

    Args:
        ckpt_path (List[Union[str, Path]]): list of paths to PyTorch checkpoint
        output_path (Union[str, Path]): path to output MLX checkpoint
        show_kv (bool, optional): show key and value shape. Defaults to False.
        replace (str, optional): replace string in key. Defaults to "model.".
    """
    state_dict = {}
    for ckpt_path in ckpt_paths:
        print(f"> Loading {ckpt_path}..")
        # safetensors
        if ckpt_path.endswith(".safetensors"):
            with safe_open(ckpt_path, framework="pt", device="cpu") as state:
                for k in tqdm(state.keys(), total=len(state.keys()), desc="Converting.."):
                    v = state.get_tensor(k)
                    if show_kv: print(k, v.shape)
                    v = v.to(torch.float16).numpy()
                    state_dict[k.replace("model.", "")] = v     
        else:
            state = torch.load(ckpt_path, map_location="cpu")
            for k, v in tqdm(state.items(), total=len(state.keys()), desc="Converting.."):
                if show_kv: print(k, v.shape)
                v = v.to(torch.float16).numpy()
                state_dict[k.replace("model.", "")] = v
    
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with Timing(f"> Saving npz file at {output_path}"):
        np.savez(
            output_path,
            **state_dict
        )
        
def hf_to_npz(
    
):
    pass