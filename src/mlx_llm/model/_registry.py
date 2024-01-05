from .phi2 import phi2
from .transformer import (
    llama_2_7B_chat, 
    tiny_llama_chat_v06,
    openhermes_25_mistral_7B, 
    mistral_7B_instruct_v01, 
    mistral_7B_instruct_v02,
    e5_mistral_7b_instruct
)

MODEL_ENTRYPOINTS = {
    "Phi2": phi2,
    "LLaMA-2-7B-chat": llama_2_7B_chat,
    "TinyLlama-1.1B-Chat-v0.6": tiny_llama_chat_v06,
    "Mistral-7B-Instruct-v0.1": mistral_7B_instruct_v01,
    "Mistral-7B-Instruct-v0.2": mistral_7B_instruct_v02,
    "OpenHermes-2.5-Mistral-7B": openhermes_25_mistral_7B,
    "e5-mistral-7b-instruct":  e5_mistral_7b_instruct
}

MODEL_WEIGHTS = {
    "Phi2": { "repo_id": "mlx-community/phi-2", "filename": "weights.npz" },
    # "LLaMA-2-7B-chat": { "repo_id": "mlx-community/LLaMA-2-7B-chat-mlx_llm", "filename": None },
    "TinyLlama-1.1B-Chat-v0.6": { "repo_id": "mlx-community/TinyLlama-1.1B-Chat-v0.6", "filename": "weights.npz" },
    # "Mistral-7B-Instruct-v0.1": { "repo_id": "mlx-community/", "filename": None },
    # "Mistral-7B-Instruct-v0.2": { "repo_id": "mlx-community/", "filename": None },
    "OpenHermes-2.5-Mistral-7B": { "repo_id": "mlx-community/OpenHermes-2.5-Mistral-7B", "filename": "weights.npz" },
    "e5-mistral-7b-instruct": { "repo_id": "mlx-community/e5-mistral-7b-instruct-mlx", "filename": "weights.npz" }
}