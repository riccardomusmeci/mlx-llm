from typing import Callable, Dict, Optional, Tuple

from ._config import HFConfig, ModelConfig, QuantizeConfig
from ._convert import llama_to_mlxllm, mistral_to_mlxllm, phi3_to_mlxllm
from .transformer import Transformer

MODEL_ENTRYPOINTS = {}


def register_model(name: Optional[str] = None) -> Callable:
    """Register a model entrypoint.

    Args:
        name (Optional[str], optional): mlx_llm model name. If None, is the same as the fn name. Defaults to None.

    Returns:
        Callable: model entrypoint
    """

    def wrapper(fn: Callable) -> Callable:
        key = name if name is not None else fn.__name__
        if key in MODEL_ENTRYPOINTS:
            raise ValueError(f"Model entrypoints already with '{key}' model.")
        MODEL_ENTRYPOINTS[key] = fn
        return fn

    return wrapper


@register_model("llama_2_7b_chat")
def llama_2_7b_chat() -> Tuple[Transformer, ModelConfig]:
    """Create a LLaMA 2 7B chat model.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=4096,
        hidden_dim=11008,
        vocab_size=32000,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        norm_eps=1e-5,
        rope_theta=10000.0,
        rope_traditional=False,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="meta-llama/Llama-2-7b-chat-hf",
        ),
        converter=llama_to_mlxllm,
    )
    return model, config


@register_model("llama_2_7b")
def llama_2_7b() -> Tuple[Transformer, ModelConfig]:
    """Create a LLaMA 2 7B chat model.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=4096,
        hidden_dim=11008,
        vocab_size=32000,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        norm_eps=1e-5,
        rope_theta=10000.0,
        rope_traditional=False,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="meta-llama/Llama-2-7b",
        ),
        converter=llama_to_mlxllm,
    )
    return model, config


@register_model("llama_3_8b")
def llama_3_8b() -> Tuple[Transformer, ModelConfig]:
    """Create a LLaMA 3 8B model.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=4096,
        hidden_dim=14336,
        vocab_size=128256,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        norm_eps=1e-5,
        rope_theta=500000.0,
        rope_traditional=True,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="meta-llama/Meta-Llama-3-8B",
        ),
        converter=llama_to_mlxllm,
    )
    return model, config


@register_model("llama_3_8b_instruct")
def llama_3_8b_instruct() -> Tuple[Transformer, ModelConfig]:
    """Create a LLaMA 3 8B Instruct model.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=4096,
        hidden_dim=14336,
        vocab_size=128256,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        norm_eps=1e-5,
        rope_theta=500000.0,
        rope_traditional=True,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        ),
        converter=llama_to_mlxllm,
    )
    return model, config


@register_model("phi_3_mini_4k_instruct")
def phi3_mini_4k_instruct() -> Tuple[Transformer, ModelConfig]:
    model = Transformer(
        dim=3072,
        hidden_dim=8192,
        vocab_size=32064,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        rope_theta=10000.0,
        rope_traditional=False,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="microsoft/Phi-3-mini-4k-instruct",
        ),
        converter=phi3_to_mlxllm,
    )
    return model, config


@register_model("phi_3_mini_128k_instruct")
def phi3_mini_128k_instruct() -> Tuple[Transformer, ModelConfig]:
    model = Transformer(
        dim=3072,
        hidden_dim=8192,
        vocab_size=32064,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        rope_theta=10000.0,
        rope_traditional=False,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="microsoft/Phi-3-mini-128k-instruct",
        ),
        converter=phi3_to_mlxllm,
    )
    return model, config


@register_model("mistral_7b_instruct_v0.2")
def mistral_7b_instruct_v02() -> Tuple[Transformer, ModelConfig]:
    model = Transformer(
        dim=4096,
        hidden_dim=14336,
        vocab_size=32000,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        norm_eps=1e-5,
        rope_theta=10000.0,
        rope_traditional=False,
    )

    config = ModelConfig(hf=HFConfig(repo_id="mistralai/Mistral-7B-Instruct-v0.2"), converter=mistral_to_mlxllm)

    return model, config


@register_model("openhermes_2.5_mistral_7b")
def openhermes_25_mistral_7b() -> Tuple[Transformer, ModelConfig]:
    model = Transformer(
        dim=4096,
        hidden_dim=14336,
        vocab_size=32000,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        norm_eps=1e-5,
        rope_theta=10000.0,
        rope_traditional=False,
    )

    config = ModelConfig(hf=HFConfig(repo_id="teknium/OpenHermes-2.5-Mistral-7B"), converter=mistral_to_mlxllm)

    return model, config


@register_model("tiny_llama_1.1B_chat_v1.0")
def tiny_llama_11B_chat_v10() -> Tuple[Transformer, ModelConfig]:
    model = Transformer(
        dim=2048,
        hidden_dim=5632,
        vocab_size=32000,
        n_layers=22,
        n_heads=32,
        n_kv_heads=4,
        norm_eps=1e-5,
        rope_theta=10000.0,
        rope_traditional=False,
    )

    config = ModelConfig(hf=HFConfig(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"), converter=llama_to_mlxllm)

    return model, config
