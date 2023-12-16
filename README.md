# mlx-llm
LLM applications running on Apple Silicon thanks to [Apple MLX framework](https://github.com/ml-explore/mlx).

## **How to install 🔨**
```
git clone https://github.com/riccardomusmeci/mlx-llm
cd mlx-llm
pip install .
```

## **LLM Chat 📱**
mlx-llm comes with tools to easily run your LLM chats on Apple Silicon.

### **LLaMA v2 🦙**
Before running LLaMA v2, you need to download the model and the tokenizer from [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/). 

> ⚠️ **Warning:** Currently only LLaMA2-7B is supported.

Then you have to convert the model weights (PyTorch) to Apple MLX format with the following command:
```python
from mlx_llm.model import LLaMA

LLaMA.convert(
    ckpt_path="path/to/llama.ckpt",
    output_path="path/to/llama.npz",
)
```
> ⚠️ **Warning:** Currently only single file weights are supported. If you have a multi-file checkpoint, you can use the following command to merge them:
> ```bash
>  cat llama.ckpt.* > llama.ckpt
> ```

Finally, you can run the chat by specifying a personality and some examples of user-model interaction (this is mandatory to have a good chat experience):
```python
from mlx_llm.model import LLaMA

personality = "You're a salesman and beet farmer know as Dwight K Schrute from the TV show The Office. Dwight replies just as he would in the show. You always reply as Dwight would reply. If you don't know the answer to a question, please don't share false information."

# examples must be structured as below
examples = [
    {
        "user": "What is your name?",
        "model": "Dwight K Schrute",
    },
    {
        "user": "What is your job?",
        "model": "Assistant Regional Manager. Sorry, Assistant to the Regional Manager.",
    },
    {
        "user": "What is your favorite color?",
        "model": "Brown. Beets are brown. Bears are brown. Bears eat beets. Bears, beets, Battlestar Galactica.",
    }
]

llama = LLaMA.build(
    model_name="llama-7B",
    weights_path="path/to/llama.npz",
    tokenizer_path="path/to/llama.tokenizer",
    personality=personality,
    examples=examples,
)
    
llama.chat(max_tokens=500)
```

## 📧 Contact

If you have any questions, please email `riccardomusmeci92@gmail.com`

