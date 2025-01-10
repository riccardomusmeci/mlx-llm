import os
import time
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from transformers import AutoTokenizer


def apply_weights(
    model: nn.Module,
    weights: Dict[str, mx.array],  # type: ignore
) -> nn.Module:
    """Apply weights to a model.

    Args:
        model (nn.Module): a model
        weights (Dict[str, mx.array]): weights dict

    Returns:
        nn.Module: a model with weights applied
    """
    model.update(tree_unflatten(list(weights.items())))
    return model


def load_weights(
    model: nn.Module, weights: Union[str, List[str]], strict: bool = True, verbose: bool = False
) -> nn.Module:
    """Load weights from a given path.

    Args:
        model (nn.Module): a LLM model
        weights (Union[str, List[str]]): path to weights file or list of paths to weights files
        strict (bool, optional): whether to strictly enforce that the keys in weights match the keys of the model. Defaults to True.
        verbose (bool, optional): whether to print information during loading. Defaults to False.

    Returns:
        nn.Module: an nn.Module with loaded weights
    """

    if verbose:
        print(f"\n> Loading weights from {weights}")

    # loading weights
    if isinstance(weights, str):
        weights = [weights]
    for weight in weights:
        assert os.path.exists(weight), f"Weights path {weight} does not exist."

    pretrained_weights = {}
    for weight in weights:
        pretrained_weights.update(dict(list(mx.load(weight).items())))

    # create a torch-like state dict { layer_name: weights }
    model_weights = dict(tree_flatten(model.parameters()))

    if strict:
        if len(model_weights) != len(pretrained_weights):
            raise ValueError(f"Expected {len(model_weights)} keys, got {len(pretrained_weights)}")
        if set(model_weights.keys()) != set(pretrained_weights.keys()):
            diff = set(model_weights.keys()) ^ set(pretrained_weights.keys())
            raise ValueError(f"Found model keys not in pretrained weights: {diff}")
        if set(pretrained_weights.keys()) != set(model_weights.keys()):
            diff = set(pretrained_weights.keys()) ^ set(model_weights.keys())
            raise ValueError(f"Found pretrained keys not in model weights: {diff}")

    for k, w in model_weights.items():
        if k not in pretrained_weights:
            if strict:
                raise KeyError(f"Missing key {k} in weights file")
            elif verbose:
                print(f"> [WARNING] Missing key {k} in weights file")
            continue
        else:
            pretrained_w = pretrained_weights[k]
            if verbose:
                print(f"> [SUCCESS] Found key {k} in weights file")
            # checking if pretrained_w has the same shape as w
            if pretrained_w.shape != w.shape:
                if strict:
                    raise ValueError(f"Expected shape {w.shape} for key {k}, got {pretrained_w.shape}")
                elif verbose:
                    print(f"> [WARNING] Expected shape {w.shape} for key {k}, got {pretrained_w.shape}")
                    pretrained_w = w
            model_weights[k] = pretrained_w

    model = apply_weights(model, model_weights)
    return model


def get_weights(model: nn.Module, trainable: bool = False) -> dict:
    """Return the model weights dict. If trainable is True, return only trainable parameters.

    Args:
        model (nn.Module): a model
        trainable (bool, optional): whether to return trainable parameters. Defaults to False.

    Returns:
        dict: model weights dict
    """
    if trainable:
        state_dict = dict(tree_flatten(model.trainable_parameters()))
    else:
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

    nn.quantize(model, group_size, bits)  # type: ignore
    return model


def generate(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.1,
    stats: bool = True,
) -> None:
    """Generate text from a given prompt.

    Args:
        model (nn.Module): model
        tokenizer (AutoTokenizer): tokenizer
        prompt (str): prompt
        max_tokens (int, optional): max number of tokens to generate. Defaults to 100.
        temperature (float, optional): temperature. Defaults to 0.1.
    """
    # generate answer
    x = mx.array([tokenizer.bos_token_id] + tokenizer.encode(prompt))

    skip = 0
    tokens = []

    print(prompt, end="", flush=True)

    for _i, token in enumerate(model.generate(x, temperature)):
        if _i == 0:
            tick = time.time()
        tokens.append(token.item())  # actually compute the token
        if len(tokens) >= max_tokens:
            break
        token_list = list(tokens)
        if token_list[-1] == tokenizer.eos_token_id:
            break
        answer = tokenizer.decode(token_list)
        print(answer[skip:], end="", flush=True)
        skip = len(answer)
    tock = time.time()
    tokens_per_second = len(tokens) / (tock - tick)
    if stats:
        print(f"\n\n[STATS] Tokens/sec: {tokens_per_second:.3f}")


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by the divisor
    It can be seen at:
    https://github.com/tensorflow/models/blob/2cfc99eff5e5eb729c6793d2f3d03aa1c9be2b15/research/slim/nets/mobilenet/mobilenet.py#L62
    Args:
        v: input value
        divisor: default to 8
        min_value: minimum divisor value
    Returns:
        new_v: new divisible value
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)  # type: ignore
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:  # type: ignore
        new_v += divisor  # type: ignore
    return new_v  # type: ignore
