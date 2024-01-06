# Models Summary

mlx-llm models are available in HuggingFace and can be used with the following snippet:

```python
from mlx_llm.model import create_model
model = create_model(
    model_name=...,
    weights=True | "path/to/weights.npz", # True if weights are in HuggingFace
)
```

Here's the table with available models and their tokenizers:

| Family | Type | Model Name | Weights | Tokenizer | Notes |
|--------|------|------------|---------|-----------|-------|
|   LLaMa     |  chat        |   LLaMA-2-7B-chat              |  [link](https://huggingface.co/mlx-community/Llama-2-7b-chat-mlx/tree/main)          |  mlx-community/Llama-2-7b-chat-mlx         |       |
|   LLaMa     |  chat        |   TinyLlama-1.1B-Chat-v0.6     |  [link](https://huggingface.co/mlx-community/TinyLlama-1.1B-Chat-v0.6/tree/main)     |  mlx-community/TinyLlama-1.1B-Chat-v0.6          |  Download tokenizer from [🤗 link](https://huggingface.co/mlx-community/TinyLlama-1.1B-Chat-v0.6/blob/main/tokenizer.model) and use it locally for chat applications  |
|   Mistral   |  chat        |   Mistral-7B-Instruct-v0.2     |  [link](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.2/tree/main)     |  mlx-community/Mistral-7B-Instruct-v0.2         |       |
|   Mistral   |  chat        |   OpenHermes-2.5-Mistral-7B    |  [link](https://huggingface.co/mlx-community/OpenHermes-2.5-Mistral-7B/tree/main)    |  mlx-community/OpenHermes-2.5-Mistral-7B         |       |
|   Mistral   |  embedding   |   e5-mistral-7b-instruct       |  [link](https://huggingface.co/mlx-community/e5-mistral-7b-instruct-mlx/tree/main)   |  mlx-community/e5-mistral-7b-instruct-mlx        |       |
|   Phi2      |  chat        |   Phi2                         |  [link](https://huggingface.co/mlx-community/phi-2/tree/main)                        |  microsoft/phi-2        |  Chat application currently not working (🤷)     |

