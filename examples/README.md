
# **Examples 🧑‍💻**

## **Chat**
Supported personalities:
- Dwight K Schrute (The Office)
- Michael Scott (The Office)
- Kanye West (Rapper)

Here's an example of how to chat with Dwight K Schrute and OpenHermes-2.5-Mistral-7B:

```bash
python examples/chat/chat_llm.py \
    --personality dwight \
    --model OpenHermes-2.5-Mistral-7B \
    --tokenizer mlx-community/OpenHermes-2.5-Mistral-7B \ # can be a local path to a tokenizer.model file
    --max_tokens 500 \
```