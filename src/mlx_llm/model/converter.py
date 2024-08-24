from typing import Dict, List

import mlx.core as mx
import torch

from ..utils.weights import weights_to_mlx


def llama_to_mlxllm(weights_paths: List[str], verbose: bool = False) -> Dict[str, mx.array]:
    """Convert LLaMA-type weights to MLX format.

    Args:
        weights_paths (List[str]): list of paths to LLaMA-type weights
        verbose (bool, optional): whether to print information during conversion. Defaults to False.

    Returns:
        Dict[str, mx.array]: a dict of weights in MLX format
    """

    model_weights = {}
    weights = weights_to_mlx(weights_paths)
    if verbose:
        print("Converting LLaMA-type weights to mlx-llm format.")
    for k, w in weights.items():
        if k.startswith("model."):
            k = k.replace("model.", "")
        k_split = k.split(".")
        if "layers" in k_split:
            if "rotary_emb" in k_split:
                continue
            if "self_attn" in k_split:
                # ['layers', '0', 'attention', 'q_proj' | 'k_proj' | 'v_proj' | 'out_proj', weight]
                model_k = f"{k_split[0]}.{k_split[1]}.attention.{k_split[3]}.{k_split[4]}"
                model_weights[model_k] = w
            elif "mlp" in k_split:
                model_k = k
                model_weights[model_k] = w
            else:
                key_map = {
                    "input_layernorm": "attention_norm",
                    "post_attention_layernorm": "mlp_norm",
                    "pre_feedforward_layernorm": "pre_mlp_norm",
                    "post_feedforward_layernorm": "post_mlp_norm",
                }
                model_k = f"{k_split[0]}.{k_split[1]}.{key_map[k_split[2]]}.{k_split[3]}"
                model_weights[model_k] = w
        else:
            key_map = {"embed_tokens": "token_embed", "lm_head": "head", "norm": "norm"}
            model_k = f"{key_map[k_split[0]]}.{k_split[1]}"
            model_weights[model_k] = w
    return model_weights


def phi3_to_mlxllm(weights_paths: List[str], verbose: bool = False) -> Dict[str, mx.array]:
    """Convert original Phi3 weights to MLX format.

    Args:
        weights_paths (List[str]): list of paths to original Phi3 weights
        verbose (bool, optional): whether to print information during conversion. Defaults to False.

    Returns:
        Dict[str, mx.array]: a dict of weights in MLX format
    """
    model_weights = {}
    weights = weights_to_mlx(weights_paths)
    if verbose:
        print("Converting LLaMA 3 weights to mlx-llm format.")
    for k, w in weights.items():
        if k.startswith("model."):
            k = k.replace("model.", "")
        k_split = k.split(".")

        if "layers" in k_split:
            if "qkv_proj" in k_split:
                # ['layers', '0', 'attention', 'q_proj' | 'k_proj' | 'v_proj' | 'out_proj', weight]
                qkv = w.reshape(3, -1, w.shape[-1])
                q, k, v = qkv[0], qkv[1], qkv[2]
                q_key = f"{k_split[0]}.{k_split[1]}.attention.q_proj.{k_split[4]}"
                k_key = f"{k_split[0]}.{k_split[1]}.attention.k_proj.{k_split[4]}"
                v_key = f"{k_split[0]}.{k_split[1]}.attention.v_proj.{k_split[4]}"
                model_weights[q_key] = q
                model_weights[k_key] = k
                model_weights[v_key] = v
            elif "o_proj" in k_split:
                # - ['layers', '0', 'self_attn', 'o_proj', 'weight']
                model_k = f"{k_split[0]}.{k_split[1]}.attention.o_proj.{k_split[4]}"
                model_weights[model_k] = w
            elif "gate_up_proj" in k_split:
                gate_up_proj = w.reshape(2, -1, w.shape[-1])
                gate_proj, up_proj = gate_up_proj[0], gate_up_proj[1]
                gate_k = f"{k_split[0]}.{k_split[1]}.mlp.gate_proj.{k_split[4]}"
                up_k = f"{k_split[0]}.{k_split[1]}.mlp.up_proj.{k_split[4]}"
                model_weights[gate_k] = gate_proj
                model_weights[up_k] = up_proj
            elif "down_proj" in k_split:
                model_k = k
                model_weights[model_k] = w
            else:
                key_map = {"input_layernorm": "attention_norm", "post_attention_layernorm": "mlp_norm"}
                model_k = f"{k_split[0]}.{k_split[1]}.{key_map[k_split[2]]}.{k_split[3]}"
                model_weights[model_k] = w
        else:
            key_map = {"embed_tokens": "token_embed", "lm_head": "head", "norm": "norm"}
            model_k = f"{key_map[k_split[0]]}.{k_split[1]}"
            model_weights[model_k] = w
    return model_weights


def mistral_to_mlxllm(weights_paths: List[str], verbose: bool = False) -> Dict[str, mx.array]:
    """Convert Mistral weights to MLX format.

    Args:
        weights_paths (List[str]): list of paths to Mistral weights
        verbose (bool, optional): whether to print information during conversion. Defaults to False.

    Returns:
        Dict[str, mx.array]: a dict of weights in MLX format
    """
    model_weights = {}
    weights = weights_to_mlx(weights_paths)
    if verbose:
        print("Converting Mistral weights to mlx-llm format.")
    for k, w in weights.items():
        if k.startswith("model."):
            k = k.replace("model.", "")
        k_split = k.split(".")
        if "layers" in k_split:
            if "self_attn" in k_split:
                # ['layers', '0', 'self_attn', 'q_proj' | 'k_proj' | 'v_proj' | 'out_proj', weight]
                model_k = f"{k_split[0]}.{k_split[1]}.attention.{k_split[3]}.{k_split[4]}"
                model_weights[model_k] = w
            elif "mlp" in k_split:
                model_k = k
                model_weights[model_k] = w
            else:
                key_map = {"input_layernorm": "attention_norm", "post_attention_layernorm": "mlp_norm"}
                model_k = f"{k_split[0]}.{k_split[1]}.{key_map[k_split[2]]}.{k_split[3]}"
                model_weights[model_k] = w
        else:
            key_map = {"embed_tokens": "token_embed", "lm_head": "head", "norm": "norm"}
            model_k = f"{key_map[k_split[0]]}.{k_split[1]}"
            model_weights[model_k] = w
    return model_weights


def openelm_to_mlxllm(
    weights_paths: List[str],
    head_dim: int,
    n_kv_heads: List[int],
    tensor_type: torch.dtype = torch.float32,
    verbose: bool = False,
) -> Dict[str, mx.array]:
    """Convert OpenELM weights to MLX format.

    Args:
        weights_paths (List[str]): list of paths to OpenELM weights
        tensor_type (torch.dtype, optional): tensor type. Defaults to torch.float32.
        verbose (bool, optional): whether to print information during conversion. Defaults to False.

    Returns:
        Dict[str, mx.array]: a dict of weights in MLX format
    """
    model_weights = {}
    weights = weights_to_mlx(weights_paths, tensor_type=tensor_type)
    if verbose:
        print("Converting OpenELM weights to mlx-llm format.")
    for k, w in weights.items():
        if k.startswith("transformer."):
            k = k.replace("transformer.", "")
        k_split = k.split(".")
        if "qkv_proj" in k_split:
            idx = int(k_split[1])
            kv_dim = n_kv_heads[idx] * head_dim
            q_dim = w.shape[0] - 2 * kv_dim
            k_dim = kv_dim
            query, key, value = mx.split(w, [q_dim, q_dim + k_dim], axis=0)
            query_key = f"layers.{k_split[1]}.attention.q_proj.weight"
            key_key = f"layers.{k_split[1]}.attention.k_proj.weight"
            value_key = f"layers.{k_split[1]}.attention.v_proj.weight"
            model_weights[query_key] = query
            model_weights[key_key] = key
            model_weights[value_key] = value
        elif "out_proj" in k_split:
            model_key = f"layers.{k_split[1]}.attention.o_proj.weight"
            model_weights[model_key] = w
        elif "k_norm" in k_split or "q_norm" in k_split:
            model_key = f"layers.{k_split[1]}.attention.{k_split[3]}.weight"
            model_weights[model_key] = w
        elif "proj_1" in k_split:
            weight = w.reshape([2, -1, w.shape[-1]])
            gate_proj, up_proj = weight[0], weight[1]
            gate_key = f"layers.{k_split[1]}.mlp.gate_proj.weight"
            up_key = f"layers.{k_split[1]}.mlp.up_proj.weight"
            model_weights[gate_key] = gate_proj
            model_weights[up_key] = up_proj
        elif "proj_2" in k_split:
            model_key = f"layers.{k_split[1]}.mlp.down_proj.weight"
            model_weights[model_key] = w
        elif "attn_norm" in k_split:
            model_key = f"layers.{k_split[1]}.attention_norm.weight"
            model_weights[model_key] = w
        elif "ffn_norm" in k_split:
            model_key = f"layers.{k_split[1]}.mlp_norm.weight"
            model_weights[model_key] = w
        else:
            key_map = {"token_embeddings": "token_embed", "norm": "norm"}
            model_key = f"{key_map[k_split[0]]}.weight"
            model_weights[model_key] = w
    return model_weights
