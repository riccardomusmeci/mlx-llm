import argparse
import os

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_llm.model import create_model, create_tokenizer
from mlx_llm.playground.rag import EmbeddingModel, KnowledgeBase, KnowledgeBaseLoader, VectorDB
from mlx_llm.utils import Timing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/vector-store", help="Collection path")
    parser.add_argument("--data", type=str, default="data/kohesio_PO_budget_0-10000.json", help="Knowledge base path")
    parser.add_argument("--model", type=str, default="bert-large-uncased", help="Model name")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    return args


def main(args):
    # create vector DB
    vector_db = VectorDB()

    # Knowledge base
    kb = KnowledgeBase(args.data)
    kb_loader = KnowledgeBaseLoader(kb, batch_size=4)
    # tokenizer
    create_tokenizer(args.model)
    # embedding model
    model = EmbeddingModel(model_name=args.model, max_length=args.max_length, mode="avg")
    # embeddings
    for _batch_idx, batch in tqdm(enumerate(kb_loader), total=len(kb_loader)):
        text = [el["content"] for el in batch]
        embeds = model(text)
        for i, el in enumerate(batch):
            vector_db.add(embedding=embeds[i], text=el["content"], source=el["source"], page=el["page"])

        vector_db.save(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
