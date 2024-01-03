# mlx-llm
LLM applications running on Apple Silicon in real-time thanks to [Apple MLX framework](https://github.com/ml-explore/mlx).

![Alt Text](static/mlx-llm-demo.gif)


Here's also a [Youtube Video](https://www.youtube.com/watch?v=vB7tk6W6VIw).


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
|   LLaMA-2  |  [link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)   |  LLaMA-2-7B-chat |
|   Mistral  |  [link](https://docs.mistral.ai/models)  |   Mistral-7B-Instruct-v0.1, Mistral-7B-Instruct-v0.2  |
|   OpenHermes-Mistral  |  [link](https://huggingface.co/mlx-community/OpenHermes-2.5-Mistral-7B/tree/main)  |   OpenHermes-2.5-Mistral-7B  |
|   Microsoft Phi2  |  [link](https://huggingface.co/mlx-community/phi-2/tree/main)  |   Phi2  |
|   Tiny-LLaMA |  [link](https://huggingface.co/mlx-community/TinyLlama-1.1B-Chat-v0.6/tree/main)  |  TinyLlama-1.1B-Chat-v0.6  |

To list all available models:
```python
from mlx_llm.model import list_models

print(list_models())
```

### **How to run**
Weights from mlx-community in HuggingFace can be used once downloaded, while weights from original sources must be converted into Apple MLX format (.npz). 

Use the snippet below to convert weights from original source:

```python
from mlx_llm.utils import weights_to_npz

# if weights are original ones (from raw sources)
weights_to_npz(
    ckpt_paths=[
        "path/to/model_1.bin", # also support safetensor
        "path/to/model_2.bin",
    ]
    output_path="path/to/model.npz",
)
```

Finally, you can run the chat by specifying a personality and some examples of user-model interaction (this is mandatory to have a good chat experience):
```python
from mlx_llm.playground import LLM

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
    model_name="LLaMA-2-7B-chat",
    weights_path="path/to/weights.npz",
    tokenizer_path="path/to/llama.tokenizer",
    personality=personality,
    examples=examples,
)
    
llm.chat(max_tokens=500, temp=0.1)
```

## **Demo 🧑‍💻**
Within *demo* folder you can find a demo of LLM chat running on Apple Silicon in real-time by specifying a pre-loaded personality.

Supported personalities:
- Dwight K Schrute (The Office)
- Michael Scott (The Office)
- Kanye West (Rapper)
- Astro (An astrophysicist that likes to keypoints)

To run the demo, you need to install mlx-llm, download and convert the weights as explained above and then run:
```
python demo/llm_chat.py \
    --personality dwight|michael|kanye|astro
    --model llama-2-7b-chat|Mistral-7B-v0.2-instruct
    --weights path/to/weights.npz
    --tokenizer path/to/tokenizer.model
    --max_tokens 500
```

## **ToDos**

[ ] Make it installable from PyPI

[ ] Add tests

[ ] One class to rule them all (LLaMA, Phi2 and Mixtral)

## 📧 Contact

If you have any questions, please email `riccardomusmeci92@gmail.com`

