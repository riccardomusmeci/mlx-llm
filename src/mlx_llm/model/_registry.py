from typing import Callable, Dict, Optional, Tuple

from ._config import HFConfig, ModelConfig, QuantizeConfig
from .converter import llama_to_mlxllm, mistral_to_mlxllm, phi3_to_mlxllm
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


@register_model("llama_2_7b_chat_hf")
def llama_2_7b_chat(
    vocab_size: int = 32000, norm_eps: float = 1e-5, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a LLaMA 2 7B chat model.

    Args:
        vocab_size (int, optional): vocab size. Defaults to 32000.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-5.
        rope_theta (float, optional): rope theta. Defaults to 10000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=4096,
        hidden_dim=11008,
        vocab_size=vocab_size,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="meta-llama/Llama-2-7b-chat-hf",
        ),
        converter=llama_to_mlxllm,
    )
    return model, config


@register_model("llama_2_7b_hf")
def llama_2_7b(
    vocab_size: int = 32000, norm_eps: float = 1e-5, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a LLaMA 2 7B chat model.

    Args:
        vocab_size (int, optional): vocab size. Defaults to 32000.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-5.
        rope_theta (float, optional): rope theta. Defaults to 10000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=4096,
        hidden_dim=11008,
        vocab_size=vocab_size,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="meta-llama/Llama-2-7b-hf",
        ),
        converter=llama_to_mlxllm,
    )
    return model, config


@register_model("llama_3_8b")
def llama_3_8b(
    vocab_size: int = 128256, norm_eps: float = 1e-5, rope_theta: float = 500000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a LLaMA 3 8B model.

    Args:
        vocab_size (int, optional): vocab size. Defaults to 128256.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-5.
        rope_theta (float, optional): rope theta. Defaults to 500000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=4096,
        hidden_dim=14336,
        vocab_size=vocab_size,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="meta-llama/Meta-Llama-3-8B",
        ),
        converter=llama_to_mlxllm,
    )
    return model, config


@register_model("llama_3_8b_instruct")
def llama_3_8b_instruct(
    vocab_size: int = 128256, norm_eps: float = 1e-5, rope_theta: float = 500000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a LLaMA 3 8B Instruct model.

    Args:
        vocab_size (int, optional): vocab size. Defaults to 128256.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-5.
        rope_theta (int, optional): rope theta. Defaults to 500000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """

    model = Transformer(
        dim=4096,
        hidden_dim=14336,
        vocab_size=vocab_size,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        ),
        converter=llama_to_mlxllm,
    )
    return model, config


@register_model("phi_3_mini_4k_instruct")
def phi3_mini_4k_instruct(
    vocab_size: int = 32064, norm_eps: float = 1e-5, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a Phi3 Mini 4k Instruct model.

    Args:
        vocab_size (int, optional): vocab size. Defaults to 32064.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-5.
        rope_theta (float, optional): rope theta. Defaults to 10000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=3072,
        hidden_dim=8192,
        vocab_size=vocab_size,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="microsoft/Phi-3-mini-4k-instruct",
        ),
        converter=phi3_to_mlxllm,
    )
    return model, config


@register_model("phi_3_mini_128k_instruct")
def phi3_mini_128k_instruct(
    vocab_size: int = 32064, norm_eps: float = 1e-5, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a Phi3 Mini 128k Instruct model.

    Args:
        vocab_size (int, optional): vocab size. Defaults to 32064.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-5.
        rope_theta (float, optional): rope theta. Defaults to 10000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=3072,
        hidden_dim=8192,
        vocab_size=vocab_size,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="microsoft/Phi-3-mini-128k-instruct",
        ),
        converter=phi3_to_mlxllm,
    )
    return model, config


@register_model("mistral_7b_instruct_v0.2")
def mistral_7b_instruct_v02(
    vocab_size: int = 32064, norm_eps: float = 1e-5, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a Mistral Instruct v0.2 Instruct model.

    Args:
        vocab_size (int, optional): vocab size. Defaults to 32000.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-5.
        rope_theta (float, optional): rope theta. Defaults to 10000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=4096,
        hidden_dim=14336,
        vocab_size=vocab_size,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
    )

    config = ModelConfig(hf=HFConfig(repo_id="mistralai/Mistral-7B-Instruct-v0.2"), converter=mistral_to_mlxllm)

    return model, config


@register_model("openhermes_2.5_mistral_7b")
def openhermes_25_mistral_7b(
    vocab_size: int = 32064, norm_eps: float = 1e-5, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a OpenHermes 2.5 Mistral 7B model.

    Args:
        vocab_size (int, optional): vocab size. Defaults to 32000.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-5.
        rope_theta (float, optional): rope theta. Defaults to 10000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=4096,
        hidden_dim=14336,
        vocab_size=vocab_size,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
    )

    config = ModelConfig(hf=HFConfig(repo_id="teknium/OpenHermes-2.5-Mistral-7B"), converter=mistral_to_mlxllm)

    return model, config


@register_model("tiny_llama_1.1B_chat_v1.0")
def tiny_llama_11B_chat_v10(
    vocab_size: int = 32000, norm_eps: float = 1e-5, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a Tiny Llama 1.1B Chat v1.0 model.

    Args:
        vocab_size (int, optional): vocab size. Defaults to 32000.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-5.
        rope_theta (float, optional): rope theta. Defaults to 10000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=2048,
        hidden_dim=5632,
        vocab_size=vocab_size,
        n_layers=22,
        n_heads=32,
        n_kv_heads=4,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
    )

    config = ModelConfig(hf=HFConfig(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"), converter=llama_to_mlxllm)

    return model, config


@register_model("gemma_1.1_2b_it")
def gemma_2b_it(
    vocab_size: int = 256000, norm_eps: float = 1e-6, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a Gemma 2B (v1.1) model

    Args:
        vocab_size (int, optional): vocab size. Defaults to 256000.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-6.
        rope_theta (float, optional): rope theta. Defaults to 10000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=2048,
        hidden_dim=16384,
        vocab_size=vocab_size,
        n_layers=18,
        n_heads=8,
        n_kv_heads=1,
        head_dim=256,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
        gemma=True,
    )

    config = ModelConfig(hf=HFConfig(repo_id="google/gemma-1.1-2b-it"), converter=llama_to_mlxllm)

    return model, config


@register_model("gemma_1.1_7b_it")
def gemma_7b_it(
    vocab_size: int = 256000, norm_eps: float = 1e-6, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a Gemma /B (v1.1) model

    Args:
        vocab_size (int, optional): vocab size. Defaults to 256000.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-6.
        rope_theta (float, optional): rope theta. Defaults to 10000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=3072,
        hidden_dim=24576,
        vocab_size=vocab_size,
        n_layers=28,
        n_heads=16,
        n_kv_heads=16,
        head_dim=256,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
        gemma=True,
    )

    config = ModelConfig(hf=HFConfig(repo_id="google/gemma-1.1-7b-it"), converter=llama_to_mlxllm)

    return model, config
