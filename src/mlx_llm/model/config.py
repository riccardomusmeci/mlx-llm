CONFIG = {
    "Mistral-7B-v0.2-Instruct": {
        "dim": 4096,
        "n_layers": 32,
        "head_dim": 128,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8,
        # "rope_theta": 1000000.0,
        "norm_eps": 1e-05,
        "vocab_size": 32000
    },
    "llama-2-7b-chat": {
        "dim": 4096,
        "n_layers": 32,
        "head_dim": 128,
        "hidden_dim": 11008,
        "n_heads": 32,
        "n_kv_heads": 32,
        "norm_eps": 1e-05,
        # "sliding_window": 4096,
        "vocab_size": 32000
    },
}