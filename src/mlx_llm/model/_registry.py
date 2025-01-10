from functools import partial
from typing import Callable, Dict, Optional, Tuple

from ._config import HFConfig, ModelConfig, QuantizeConfig
from ._utils import make_divisible
from .converter import llama_to_mlxllm, mistral_to_mlxllm, openelm_to_mlxllm, phi3_to_mlxllm
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


@register_model("llama_3_2_1b_instruct")
def llama_3_2_1b_instruct(
    vocab_size: int = 128256, norm_eps: float = 1e-5, rope_theta: float = 500000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a LLaMA 3.2 1B Instruct model.

    Args:
        vocab_size (int, optional): vocab size. Defaults to 128256.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-5.
        rope_theta (int, optional): rope theta. Defaults to 500000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """

    model = Transformer(
        dim=2048,
        hidden_dim=8192,
        vocab_size=vocab_size,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        head_dim=64,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        embed_as_head=True,
        rope_traditional=rope_traditional,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="meta-llama/Llama-3.2-1B-Instruct",
        ),
        converter=llama_to_mlxllm,
    )
    return model, config


@register_model("llama_3_2_3b_instruct")
def llama_3_2_3b_instruct(
    vocab_size: int = 128256, norm_eps: float = 1e-5, rope_theta: float = 500000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a LLaMA 3.2 3B Instruct model.

    Args:
        vocab_size (int, optional): vocab size. Defaults to 128256.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-5.
        rope_theta (int, optional): rope theta. Defaults to 500000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """

    model = Transformer(
        dim=3072,
        hidden_dim=8192,
        vocab_size=vocab_size,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        head_dim=128,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        embed_as_head=True,
        rope_traditional=rope_traditional,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="meta-llama/Llama-3.2-3B-Instruct",
        ),
        converter=llama_to_mlxllm,
    )
    return model, config


@register_model("hermes_2_pro_llama_3_8b")
def hermes_2_pro_llama_3_8b(
    vocab_size: int = 128288, norm_eps: float = 1e-5, rope_theta: float = 500000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a Hermes 2 Pro Llama 3 8B model.

    Args:
        vocab_size (int, optional): vocab size. Defaults to 128288.
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
            repo_id="NousResearch/Hermes-2-Pro-Llama-3-8B",
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


@register_model("phi_3.5_mini_instruct")
def phi35_mini_instruct(
    vocab_size: int = 32064, norm_eps: float = 1e-5, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a Phi3.5 Mini Instruct model.

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
            repo_id="microsoft/Phi-3.5-mini-instruct",
        ),
        converter=phi3_to_mlxllm,
    )
    return model, config


@register_model("phi_4_14b")
def phi4_14b(
    vocab_size: int = 100352, norm_eps: float = 1e-5, rope_theta: float = 250000, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a Phi4 14B Mini Instruct model.

    Args:
        vocab_size (int, optional): vocab size. Defaults to 100352.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-5.
        rope_theta (float, optional): rope theta. Defaults to 250000.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=5120,
        hidden_dim=17920,
        vocab_size=vocab_size,
        n_layers=40,
        n_heads=40,
        n_kv_heads=10,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="microsoft/phi-4",
        ),
        converter=phi3_to_mlxllm,
    )
    return model, config


@register_model("mistral_7b_instruct_v0.2")
def mistral_7b_instruct_v02(
    vocab_size: int = 32000, norm_eps: float = 1e-5, rope_theta: float = 1000000.0, rope_traditional: bool = False
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


@register_model("starling_lm_7b_beta")
def starling_lm_7b_beta(
    vocab_size: int = 32002, norm_eps: float = 1e-5, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a Starling-LM-7B-beta model.

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

    config = ModelConfig(
        hf=HFConfig(repo_id="Nexusflow/Starling-LM-7B-beta"),
        tokenizer=HFConfig(repo_id="openchat/openchat-3.5-0106"),
        converter=mistral_to_mlxllm,
    )

    return model, config


@register_model("openhermes_2.5_mistral_7b")
def openhermes_25_mistral_7b(
    vocab_size: int = 32002, norm_eps: float = 1e-5, rope_theta: float = 10000.0, rope_traditional: bool = False
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
def tiny_llama_1_1B_chat_v10(
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
        embed_as_head=True,
    )

    config = ModelConfig(hf=HFConfig(repo_id="google/gemma-1.1-2b-it"), converter=llama_to_mlxllm)

    return model, config


@register_model("gemma_1.1_7b_it")
def gemma_7b_it(
    vocab_size: int = 256000, norm_eps: float = 1e-6, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a Gemma 7B (v1.1) model

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
        embed_as_head=True,
    )

    config = ModelConfig(hf=HFConfig(repo_id="google/gemma-1.1-7b-it"), converter=llama_to_mlxllm)

    return model, config


@register_model("gemma_2_2b_it")
def gemma_2_2b_it(
    vocab_size: int = 256000, norm_eps: float = 1e-6, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a Gemma 2 2B model

    Args:
        vocab_size (int, optional): vocab size. Defaults to 256000.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-6.
        rope_theta (float, optional): rope theta. Defaults to 10000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=2304,
        hidden_dim=9216,
        vocab_size=vocab_size,
        n_layers=26,
        n_heads=8,
        n_kv_heads=4,
        head_dim=256,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
        gemma=True,
        gemma2=True,
        embed_as_head=True,
    )

    config = ModelConfig(hf=HFConfig(repo_id="google/gemma-2-2b-it"), converter=llama_to_mlxllm)

    return model, config


@register_model("gemma_2_9b_it")
def gemma_2_9b_it(
    vocab_size: int = 256000, norm_eps: float = 1e-6, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    """Create a Gemma 2 9B model

    Args:
        vocab_size (int, optional): vocab size. Defaults to 256000.
        norm_eps (float, optional): norm epsilon. Defaults to 1e-6.
        rope_theta (float, optional): rope theta. Defaults to 10000.0.
        rope_traditional (bool, optional): whether to use traditional rope. Defaults to False.

    Returns:
        Tuple[Transformer, ModelConfig]: model, config
    """
    model = Transformer(
        dim=3584,
        hidden_dim=14336,
        vocab_size=vocab_size,
        n_layers=42,
        n_heads=16,
        n_kv_heads=8,
        head_dim=256,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
        gemma=True,
        gemma2=True,
        embed_as_head=True,
    )

    config = ModelConfig(hf=HFConfig(repo_id="google/gemma-2-9b-it"), converter=llama_to_mlxllm)

    return model, config


@register_model("openelm_3B_instruct")
def openelm_3B(
    vocab_size: int = 32000, norm_eps: float = 1e-5, rope_theta: float = 10000, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    dim = 3072
    head_dim = 128
    num_gqa_groups = 4
    divisble_by = 256
    n_kv_heads = [
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
    ]
    mlp_scales = [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
        2.6,
        2.7,
        2.8,
        2.9,
        3.0,
        3.1,
        3.2,
        3.3,
        3.4,
        3.5,
        3.6,
        3.7,
        3.8,
        3.9,
        4.0,
    ]

    n_heads = [num_gqa_groups * el for el in n_kv_heads]
    hidden_dim = [make_divisible(v=mlp_scale * dim, divisor=divisble_by) for mlp_scale in mlp_scales]

    model = Transformer(
        dim=dim,
        hidden_dim=hidden_dim,  # type: ignore
        vocab_size=vocab_size,
        n_layers=36,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        norm_eps=norm_eps,
        rope_traditional=rope_traditional,
        rope_theta=rope_theta,
        rope_scaling=None,
        norm_qk_proj=True,
        attention_norm_eps=norm_eps,
        embed_as_head=True,
    )

    config = ModelConfig(
        hf=HFConfig(repo_id="apple/OpenELM-3B-Instruct"),
        tokenizer=HFConfig(repo_id="meta-llama/Llama-2-7b-hf"),
        converter=partial(
            openelm_to_mlxllm,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
        ),
    )

    return model, config


@register_model("openelm_1.1B_instruct")
def openelm_1_1B(
    vocab_size: int = 32000, norm_eps: float = 1e-5, rope_theta: float = 10000, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    dim = 2048
    head_dim = 64
    num_gqa_groups = 4
    divisble_by = 256
    n_kv_heads = [4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8]
    mlp_scales = [
        0.5,
        0.63,
        0.76,
        0.89,
        1.02,
        1.15,
        1.28,
        1.41,
        1.54,
        1.67,
        1.8,
        1.93,
        2.06,
        2.19,
        2.31,
        2.44,
        2.57,
        2.7,
        2.83,
        2.96,
        3.09,
        3.22,
        3.35,
        3.48,
        3.61,
        3.74,
        3.87,
        4.0,
    ]
    n_heads = [num_gqa_groups * el for el in n_kv_heads]
    hidden_dim = [make_divisible(v=mlp_scale * dim, divisor=divisble_by) for mlp_scale in mlp_scales]

    model = Transformer(
        dim=dim,
        hidden_dim=hidden_dim,  # type: ignore
        vocab_size=vocab_size,
        n_layers=28,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        norm_eps=norm_eps,
        rope_traditional=rope_traditional,
        rope_theta=rope_theta,
        rope_scaling=None,
        norm_qk_proj=True,
        attention_norm_eps=norm_eps,
        embed_as_head=True,
    )

    config = ModelConfig(
        hf=HFConfig(repo_id="apple/OpenELM-1_1B-Instruct"),
        tokenizer=HFConfig(repo_id="meta-llama/Llama-2-7b-hf"),
        converter=partial(
            openelm_to_mlxllm,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
        ),
    )

    return model, config


@register_model("openelm_450M_instruct")
def openelm_450M(
    vocab_size: int = 32000, norm_eps: float = 1e-5, rope_theta: float = 10000, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    dim = 1536
    head_dim = 64
    num_gqa_groups = 4
    divisble_by = 256
    n_kv_heads = [3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6]
    mlp_scales = [
        0.5,
        0.68,
        0.87,
        1.05,
        1.24,
        1.42,
        1.61,
        1.79,
        1.97,
        2.16,
        2.34,
        2.53,
        2.71,
        2.89,
        3.08,
        3.26,
        3.45,
        3.63,
        3.82,
        4.0,
    ]
    n_heads = [num_gqa_groups * el for el in n_kv_heads]
    hidden_dim = [make_divisible(v=mlp_scale * dim, divisor=divisble_by) for mlp_scale in mlp_scales]

    model = Transformer(
        dim=dim,
        hidden_dim=hidden_dim,  # type: ignore
        vocab_size=vocab_size,
        n_layers=20,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        norm_eps=norm_eps,
        rope_traditional=rope_traditional,
        rope_theta=rope_theta,
        rope_scaling=None,
        norm_qk_proj=True,
        attention_norm_eps=norm_eps,
        embed_as_head=True,
    )

    config = ModelConfig(
        hf=HFConfig(repo_id="apple/OpenELM-450M-Instruct"),
        tokenizer=HFConfig(repo_id="meta-llama/Llama-2-7b-hf"),
        converter=partial(
            openelm_to_mlxllm,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
        ),
    )

    return model, config


@register_model("openelm_270M_instruct")
def openelm_270M(
    vocab_size: int = 32000, norm_eps: float = 1e-5, rope_theta: float = 10000, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    dim = 1280
    head_dim = 64
    num_gqa_groups = 4
    divisble_by = 256
    n_kv_heads = [3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5]
    mlp_scales = [0.5, 0.73, 0.97, 1.2, 1.43, 1.67, 1.9, 2.13, 2.37, 2.6, 2.83, 3.07, 3.3, 3.53, 3.77, 4.0]
    n_heads = [num_gqa_groups * el for el in n_kv_heads]
    hidden_dim = [make_divisible(v=mlp_scale * dim, divisor=divisble_by) for mlp_scale in mlp_scales]

    model = Transformer(
        dim=dim,
        hidden_dim=hidden_dim,  # type: ignore
        vocab_size=vocab_size,
        n_layers=16,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        norm_eps=norm_eps,
        rope_traditional=rope_traditional,
        rope_theta=rope_theta,
        rope_scaling=None,
        norm_qk_proj=True,
        attention_norm_eps=norm_eps,
        embed_as_head=True,
    )

    config = ModelConfig(
        hf=HFConfig(repo_id="apple/OpenELM-270M-Instruct"),
        tokenizer=HFConfig(repo_id="meta-llama/Llama-2-7b-hf"),
        converter=partial(
            openelm_to_mlxllm,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
        ),
    )

    return model, config


@register_model("smollm2_1.7B_instruct")
def smollm2_1_7B_instruct(
    vocab_size: int = 49152, norm_eps: float = 1e-5, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    model = Transformer(
        dim=2048,
        hidden_dim=8192,
        vocab_size=vocab_size,
        n_layers=24,
        n_heads=32,
        n_kv_heads=32,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
        embed_as_head=True,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        ),
        converter=llama_to_mlxllm,
    )
    return model, config


@register_model("smollm2_360M_instruct")
def smollm2_360M_instruct(
    vocab_size: int = 49152, norm_eps: float = 1e-5, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    model = Transformer(
        dim=960,
        hidden_dim=2560,
        vocab_size=vocab_size,
        n_layers=32,
        n_heads=15,
        n_kv_heads=5,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
        embed_as_head=True,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        ),
        converter=llama_to_mlxllm,
    )
    return model, config


@register_model("smollm2_135M_instruct")
def smollm2_135M_instruct(
    vocab_size: int = 49152, norm_eps: float = 1e-5, rope_theta: float = 10000.0, rope_traditional: bool = False
) -> Tuple[Transformer, ModelConfig]:
    model = Transformer(
        dim=576,
        hidden_dim=1536,
        vocab_size=vocab_size,
        n_layers=30,
        n_heads=9,
        n_kv_heads=3,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
        embed_as_head=True,
    )

    config = ModelConfig(
        hf=HFConfig(
            repo_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        ),
        converter=llama_to_mlxllm,
    )
    return model, config
