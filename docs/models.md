# Models Summary

mlx-llm models are available in HuggingFace and can be used with the following snippet:

```python
from mlx_llm.model import create_model, create_tokenizer
model = create_model(
    model_name=...,
    weights=True | "path/to/weights.npz", # True if weights are in HuggingFace
)

# You can also get the tokenizer from
model = create_tokenizer(
    model_name=...,
)

```

Here's the table with available models and their tokenizers:

| Family | Type | Model Name | Weights | Tokenizer | Notes |
|--------|------|------------|---------|-----------|-------|
|   LLaMa     |  text generation        |   LLaMA-2-7B-chat              |  [link](https://huggingface.co/mlx-community/Llama-2-7b-chat-mlx/tree/main)          |  mlx-community/Llama-2-7b-chat-mlx         |       |
|   LLaMa     |  text generation        |   TinyLlama-1.1B-Chat-v0.6     |  [link](https://huggingface.co/mlx-community/TinyLlama-1.1B-Chat-v0.6/tree/main)     |  mlx-community/TinyLlama-1.1B-Chat-v0.6          |    |
|   Mistral   |  text generation        |   Mistral-7B-Instruct-v0.2     |  [link](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.2/tree/main)     |  mlx-community/Mistral-7B-Instruct-v0.2         |       |
|   Mistral   |  text generation        |   OpenHermes-2.5-Mistral-7B    |  [link](https://huggingface.co/mlx-community/OpenHermes-2.5-Mistral-7B/blob/main/weights.npz)    |  mlx-community/OpenHermes-2.5-Mistral-7B         |       |
|   Mistral   |  text generation        |   OpenHermes-2.5-Mistral-7B-4bit    |  [link](https://huggingface.co/mlx-community/OpenHermes-2.5-Mistral-7B/blob/main/weights_4bit.npz)    |  mlx-community/OpenHermes-2.5-Mistral-7B         |       |
|   Mistral   |  embedding   |   e5-mistral-7b-instruct       |  [link](https://huggingface.co/mlx-community/e5-mistral-7b-instruct-mlx/tree/main)   |  intfloat/e5-mistral-7b-instruct        |       |
|   Phi2      |  text generation        |   Phi2                         |  [link](https://huggingface.co/mlx-community/phi-2/tree/main)                        |  microsoft/phi-2        |       |
|   BERT     |  embedding        |   bert-base-uncased              |  [link](https://huggingface.co/mlx-community/bert-base-uncased-mlx/tree/main)          |  bert-base-uncased-mlx         |       |
|   BERT     |  embedding        |   bert-large-uncased              |  [link](https://huggingface.co/mlx-community/bert-large-uncased-mlx/tree/main)          |  bert-large-uncased-mlx         |       |
