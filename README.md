# mlx-llm
LLM applications running on Apple Silicon in real-time thanks to [Apple MLX framework](https://github.com/ml-explore/mlx).

[![Watch the video](https://img.youtube.com/vi/vB7tk6W6VIw/hqdefault.jpg)](https://www.youtube.com/embed/vB7tk6W6VIw)


## **How to install 🔨**
```
git clone https://github.com/riccardomusmeci/mlx-llm
cd mlx-llm
pip install .
```

## **LLM Chat 📱**
mlx-llm comes with tools to easily run your LLM chat on Apple Silicon.

### **Supported models**

| Model Family | Weights | Supported Models |
|----------|----------|----------|
|   LLaMA-2  |  [link](ttps://ai.meta.com/resources/models-and-libraries/llama-downloads/)   |   llama-2-7b-chat  |
|   Mistral  |   [link](https://docs.mistral.ai/models)  |   Mistral-7B-v0.2-Instruct  |

> ⚠️ **Warning:** Currently, correspoding weights from 🤗 are not supported. This because 🤗 weights have different names and shapes. You need to download the weights from the links above.


### **How to run**
Once downloaded the weights, you need to convert the tokenizer to Apple MLX format (.npz file) with the following command:
```python
from mlx_llm.utils import weights_to_npz

# also supported .safetensors files
weights_to_npz(
    ckpt_paths=[
        "path/to/model_1.bin", # if model weights are split in multiple files
        "path/to/model_2.bin",
    ]
    output_path="path/to/model.npz",
)
```

Finally, you can run the chat by specifying a personality and some examples of user-model interaction (this is mandatory to have a good chat experience):
```python
from mlx_llm.llm import LLM

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
    }
]

llm = LLM.build(
    model_name="llama-2-7B-chat",
    weights_path="path/to/llama.npz",
    tokenizer_path="path/to/llama.tokenizer",
    personality=personality,
    examples=examples,
)
    
llm.chat(max_tokens=500)
```

## **Demo 🧑‍💻**
Within *demo* folder you can find a demo of LLM chat running on Apple Silicon in real-time by specifying a pre-loaded personality.

Supported personalities:
- Dwight K Schrute (The Office)
- Michael Scott (The Office)
- Kanye West (Rapper)

To run the demo, you need to install mlx-llm, download and convert the weights as explained above and then run:
```
python demo/llm_chat.py \
    --personality dwight|michael|kanye
    --model llama-2-7b-chat|Mistral-7B-v0.2-instruct
    --weights path/to/weights.npz
    --tokenizer path/to/tokenizer.model
    --max_tokens 500
```

## 📧 Contact

If you have any questions, please email `riccardomusmeci92@gmail.com`

