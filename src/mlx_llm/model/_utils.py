import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten
from safetensors.numpy import load_file

from ._registry import MODEL_WEIGHTS


def load_weights(model: nn.Module, weights: str, strict: bool = True, verbose: bool = False) -> nn.Module:
    """Load weights from a given path.

    Args:
        model (nn.Module): a LLM model
        weights (str): path to weights
        strict (bool, optional): whether to strictly enforce that the keys in weights match the keys of the model. Defaults to True.
        verbose (bool, optional): whether to print information during loading. Defaults to False.

    Returns:
        nn.Module: an nn.Module with loaded weights
    """

    assert os.path.exists(weights), f"Weights path {weights} does not exist."

    if verbose:
        print(f"> Loading weights from {weights}")

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
            if verbose:
                print(f"\t- [WARNING] Found extra keys in weights file: {extras}")

    # check if new_state does not have less keys
    missing = set(model_state.keys()) - set(new_state.keys())
    if missing:
        missing = " ".join(list(missing))
        if strict:
            raise ValueError(f"Missing keys in weights file: {missing}")
        else:
            if verbose:
                print(f"\t- [WARNING] Missing keys in weights file: {missing}")

    for k, w in model_state.items():
        try:
            new_w = new_state[k]
        except KeyError:
            if strict:
                raise KeyError(f"Missing key {k} in weights file")  # noqa: B904
            else:
                if verbose:
                    print(f"\t- [WARNING] Missing key {k} in weights file")
            continue

        # checking if new_w is an mx.array first
        if not isinstance(new_w, mx.array):
            if strict:
                raise ValueError(f"Expected mx.array for key {k}, got {type(new_w)}")
            else:
                if verbose:
                    print(f"\t- [WARNING] Expected mx.array for key {k}, got {type(new_w)}")
        # checking if new_w has the same shape as w
        if new_w.shape != w.shape:
            if strict:
                raise ValueError(f"Expected shape {w.shape} for key {k}, got {new_w.shape}")
            else:
                if verbose:
                    print(f"\t- [WARNING] Expected shape {w.shape} for key {k}, got {new_w.shape}")

    model.update(tree_unflatten(weights))

    return model


def load_weights_from_hf(model: nn.Module, model_name: str, strict: bool = True, verbose: bool = False) -> nn.Module:
    """Load weights from HuggingFace Hub.

    Args:
        model (nn.Module): an LLM model
        model_name (str): model name
        strict (bool, optional): whether to strictly enforce that the keys in weights match the keys of the model. Defaults to True.
        verbose (bool, optional): whether to print information during loading. Defaults to False.

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
        model = load_weights(model=model, weights=weights_path, strict=strict, verbose=verbose)
    return model


def download_from_hf(model_name: str) -> str:
    """Download weights from HuggingFace Hub.

    Args:
        model_name (str): model name

    Returns:
        str: path to downloaded weights
    """
    try:
        repo_id = MODEL_WEIGHTS[model_name]["repo_id"]
        filename = MODEL_WEIGHTS[model_name]["filename"]
        weights_path = hf_hub_download(repo_id=repo_id, repo_type="model", filename=filename)
    except Exception as e:
        print(f"[ERROR] Downloading weights from HuggingFace Hub failed: {e}.")
        quit()

    return weights_path


def save_weights(model: nn.Module, path: str) -> None:
    """Save weights to a given path.

    Args:
        model (nn.Module): a LLM model
        path (str): path to save weights
    """

    dir_path = os.path.dirname(path)
    if len(dir_path) > 0 and not os.path.exists(dir_path):
        print(f"> Creating directory {dir_path}")
        os.makedirs(dir_path, exist_ok=True)

    print(f"> Saving weights to {path}")
    weights_dict = get_weights_dict(model)
    state_dict = {}
    for k, v in weights_dict.items():
        state_dict[k] = np.array(v)
    np.savez(path, **state_dict)


def get_weights_dict(model: nn.Module) -> dict:
    """Return the model weights dict.

    Args:
        model (nn.Module): a model

    Returns:
        dict: model weights dict
    """
    state_dict = dict(tree_flatten(model.parameters()))
    return state_dict


def quantize(model: nn.Module, group_size: int, bits: int) -> nn.Module:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): model to be quantized.
        group_size (int): group size for quantization.
        bits (int): bits per weight for quantization.

    Returns:
        Tuple: model
    """

    def linear_class_predicate(m):
        return isinstance(m, nn.Linear) and m.weight.shape[0] != 8

    nn.QuantizedLinear.quantize_module(model, group_size, bits, linear_class_predicate=linear_class_predicate)

    return model
