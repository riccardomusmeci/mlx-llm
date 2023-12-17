mistral_config = {
    "Mistral-7B-v0.1": {
        "dim": 4096,
        "n_layers": 32,
        "head_dim": 128,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8,
        "norm_eps": 1e-05,
        # "sliding_window": 4096,
        "vocab_size": 32000
    },
    # TODO: verify
    "OpenHermes-2.5-Mistral-7B": {
        "dim": 4096,
        "n_layers": 32,
        "head_dim": 128,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8,
        "norm_eps": 1e-05,
        # "sliding_window": 4096,
        "vocab_size": 32000
    },

}