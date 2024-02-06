import os
from argparse import ArgumentParser

import mlx.core as mx
import numpy as np

from mlx_llm.model import create_model, create_tokenizer
from mlx_llm.playground.rag import EmbeddingModel, KnowledgeBase, KnowledgeBaseLoader, OpenHermesPrompt, VectorDB
from mlx_llm.utils import Timing

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--vector-db", default="data/vector-store", type=str, help="Path to vector DB folder")
    parser.add_argument(
        "--embedding-model", default="bert-large-uncased", type=str, help="Name of the embedding model to use"
    )
    parser.add_argument("--max-length", type=int, default=512, help="Embedding dimension of the model")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0, help="Temperature for sampling")
    args = parser.parse_args()
    return args


SYSTEM_PROMPT = "You are an assistant and you will use the context below to answer the question the user will provide. Your answer should be in your own words and should not be copied from the context. If you don't know the answer, just type 'I don't know'."


def main(args):
    print(f"[INFO] Loading vector DB from {args.vector_db}")
    vector_db = VectorDB()
    vector_db.load(args.vector_db)

    print(f"[INFO] Loading embedding model: {args.embedding_model}")
    embed_model = EmbeddingModel(model_name=args.embedding_model, max_length=args.max_length, mode="avg")

    print("[INFO] Loading LLM model: OpenHermes-2.5-Mistral-7B")
    llm_model = create_model("OpenHermes-2.5-Mistral-7B")
    llm_tokenizer = create_tokenizer("OpenHermes-2.5-Mistral-7B")

    # setup prompter
    prompter = OpenHermesPrompt(system_prompt=SYSTEM_PROMPT, end_strs=["assistant", "<|im_end|>"])
    # getting question from user
    question = input("\nQ: ")

    # just to be sure that the prompt is correct for e5-mistral-7b-instruct
    if args.embedding_model == "e5-mistral-7b-instruct":
        query = (
            f"Instruct: Given a web search query, retrieve relevant passages that answer the query'\nQuesry: {question}"
        )
    else:
        query = question

    with Timing("Embedding + Context Retrieval:"):
        embedding = embed_model(text=query)
        # getting context
        out = vector_db.query(embedding=embedding[0], top_k=10)
        context = "\n -- \n".join([o["metadata"]["text"] for o in out])

    del embed_model
    # preparing prompt
    prompt = prompter.prepare(question=question, context=context)

    # generate answer
    x = mx.array([llm_tokenizer.bos_token_id] + llm_tokenizer.encode(prompt))
    skip = 0
    tokens = []
    print("A: ", end="", flush=True)
    for token in llm_model.generate(x, args.temp):
        tokens.append(token)
        if len(tokens) >= args.max_tokens:
            break
        mx.eval(tokens)
        token_list = [t.item() for t in tokens]
        answer = llm_tokenizer.decode(token_list)
        status = prompter.model_status(answer)
        if status == 0:
            continue
        if status == 1:
            skip = len(answer)
            break
        print(answer[skip:], end="", flush=True)
        skip = len(answer)
        if token_list[-1] == llm_tokenizer.eos_token_id:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
