import os
import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from huggingface_hub import hf_hub_download
from ._registry import MODEL_WEIGHTS


def load_weights(
    model: nn.Module,
    weights: str,
    strict: bool = True,
) -> nn.Module:
    """Load weights from a given path.

    Args:
        model (nn.Module): a LLM model
        weights (str): path to weights
        strict (bool, optional): whether to strictly enforce that the keys in weights match the keys of the model. Defaults to True.

    Returns:
        nn.Module: an nn.Module with loaded weights
    """
    
    assert os.path.exists(weights), f"Weights path {weights} does not exist."
    
    print(f"Loading weights from {weights}")
    
    weights = list(mx.load(weights).items())
    
    new_state = dict(weights)
    # create a torch-like state dict { layer_name: weights }
    model_state = dict(tree_flatten(model.parameters()))
    
    # check if new_state does not have more keys
    extras = set(new_state.keys()) - set(model_state.keys())
    if extras:
        extras = " ".join(list(extras))
        if strict:
            raise ValueError(f"Found extra keys in weights file: {extras}")
        else:
            print(f"[WARNING] Found extra keys in weights file: {extras}")
    
    # check if new_state does not have less keys
    missing = set(model_state.keys()) - set(new_state.keys())
    if missing:
        missing = " ".join(list(missing))
        if strict:
            raise ValueError(f"Missing keys in weights file: {missing}")
        else:
            print(f"[WARNING] Missing keys in weights file: {missing}")
    
    for k, w in model_state.items():
        try:
            new_w = new_state[k]
        except KeyError:
            if strict:
                raise ValueError(f"Missing key {k} in weights file")
            else:
                print(f"[WARNING] Missing key {k} in weights file")
            continue
        
        # checking if new_w is an mx.array first
        if not isinstance(new_w, mx.array):
            if strict:
                raise ValueError(f"Expected mx.array for key {k}, got {type(new_w)}")
            else:
                print(f"[WARNING] Expected mx.array for key {k}, got {type(new_w)}")
        # checking if new_w has the same shape as w
        if new_w.shape != w.shape:
            if strict:
                raise ValueError(f"Expected shape {w.shape} for key {k}, got {new_w.shape}")
            else:
                print(f"[WARNING] Expected shape {w.shape} for key {k}, got {new_w.shape}")
    
    model.update(tree_unflatten(weights))
    
    return model

def load_weights_from_hf(
    model: nn.Module,
    model_name: str,
    strict: bool = True
) -> nn.Module:
    """Load weights from HuggingFace Hub.

    Args:
        model (nn.Module): an LLM model
        model_name (str): model namw
        strict (bool, optional): whether to strictly enforce that the keys in weights match the keys of the model. Defaults to True.

    Returns:
        nn.Module: an LLM with loaded weights
    """
    try:
        repo_id = MODEL_WEIGHTS[model_name]["repo_id"]
        filename = MODEL_WEIGHTS[model_name]["filename"]
        weights_path = hf_hub_download(repo_id=repo_id, repo_type="model", filename=filename)
    except Exception as e:
        print(f"Error while downloading weights from HuggingFace Hub: {e}. Weights won't be loaded.")
        weights_path = None
        
    if weights_path is not None:
        model = load_weights(
            model=model,
            weights=weights_path,
            strict=strict
        )
    return model    
        
        
    