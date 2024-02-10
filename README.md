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

## **Models 🧠**

Go check [models](docs/models.md) for a summary of available models.

To create a model with weights:
```python
from mlx_llm.model import create_model

# loading weights from HuggingFace
model = create_model("TinyLlama-1.1B-Chat-v0.6")

# loading weights from local file
model = create_model("TinyLlama-1.1B-Chat-v0.6", weights="path/to/weights.npz")
```

To list all available models:
```python
from mlx_llm.model import list_models

print(list_models())
```

### **Quantization 📉**
You can quantize a model with `mlx-llm`:
```python
from mlx_llm.model import create_model, quantize, save_weights

# create the model from original weights
model = create_model("TinyLlama-1.1B-Chat-v0.6")
# quantize the model
model = quantize(model, group_size=64, bits=4)
# save the model
save_weights(model, "TinyLlama-1.1B-Chat-v0.6-4bit.npz")

```

quantize("TinyLlama-1.1B-Chat-v0.6", "tiny-llama-1.1b-chat-v0.6-quantized")
```


### **Benchmarks 📊**
You can run benchmarks with `mlx-llm` to compare mlx versions, models, and devices:
```python
from mlx_llm.bench import Benchmark

benchmark = Benchmark(
    apple_silicon="m1_pro_32GB",
    model_name="TinyLlama-1.1B-Chat-v0.6",
    prompt="What is the meaning of life?",
    max_tokens=100,
    temperature=0.1,
    verbose=False
)

benchmark.start()
# just the output dir, the file name will be benchmark.csv
benchmark.save("results") # if benchmark.csv is already there, it will append the new results
```
> [!WARNING]
> Download first the model weights before running the benchmark (just use `create_model` and then run the test).

Go to [benchmark.csv](results/benchmark.csv) to check my experiments.

If you want to run benchmarks for all available LLMs:
```bash
cd scripts
./run_benchmarks.sh
```
> [!WARNING]
> The test will take a while since it will download all the models if not already present. Also, once test for a model is done, all the 🤗 hub cache will be deleted.

> [!NOTE]
> Run the benchmarks on your Apple Silicon device and then PR-me the results. I will be happy to add them to the [benchmark.csv](results/benchmark.csv) file.


### **Model Embeddings ✴️**
Models in `mlx-llm` are able to extract embeddings from a given text.

```python
import mlx.core as mx
from mlx_llm.model import create_model
from transformers import AutoTokenizer

model = create_model("e5-mistral-7b-instruct")
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
text = ["I like to play basketball", "I like to play tennis"]
tokens = tokenizer(text)
x = mx.array(tokens["input_ids"])
embeds = model.embed(x)
```

> **For a better example go check [🤗 e5-mistral-7b-instruct page](https://huggingface.co/mlx-community/e5-mistral-7b-instruct-mlx).**

## **Applications 📁**

With `mlx-llm` you can run a variety of applications, such as:
- Chat with an LLM
- Retrieval Augmented Generation (RAG) running locally

Below an example of how to chat with an LLM, but for more details go check the [examples](examples/README.md) documentation.

### **Chat with LLM 📱**
`mlx-llm` comes with tools to easily run your LLM chat on Apple Silicon.

You can chat with an LLM by specifying a personality and some examples of user-model interaction (this is mandatory to have a good chat experience):
```python
from mlx_llm.playground.chat import ChatLLM

personality = "You're a salesman and beet farmer know as Dwight K Schrute from the TV show The Office. Dwight replies just as he would in the show. You always reply as Dwight would reply. If you don't know the answer to a question, please don't share false information."

# examples must be structured as below
examples = [
    {
        "user": "What is your name?",
        "model": "Dwight K Schrute",
    },
    {
        "user": "What is your job?",
        "model": "Assistant Regional Manager. Sorry, Assistant to the Regional Manager."
    }
]

chat_llm = ChatLLM.build(
    model_name="LLaMA-2-7B-chat",
    tokenizer="mlx-community/Llama-2-7b-chat-mlx", # HF tokenizer or a local path to a tokenizer
    personality=personality,
    examples=examples,
)

chat_llm.run(max_tokens=500, temp=0.1)
```

## **ToDos**

[ ] Chat and RAG with streamlit???

[ ] Test with quantized models

[ ] LoRA and QLoRA

## 📧 Contact

If you have any questions, please email `riccardomusmeci92@gmail.com`
