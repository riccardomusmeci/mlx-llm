import argparse

from personality import personalities

from mlx_llm.playground import ChatLLM


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="LLM chat arguments")

    parser.add_argument(
        "--personality",
        default=None,
        type=str,
        choices=list(personalities.keys()),
    )

    parser.add_argument("--model", default="OpenHermes-2.5-Mistral-7B", required=False, type=str, help="Model name.")

    parser.add_argument(
        "--weights",
        required=False,
        default=True,
        type=str,
        help="if True, load pretrained weights from HF. If str, load weights from the given path.",
    )

    parser.add_argument(
        "--tokenizer",
        default="mlx-community/OpenHermes-2.5-Mistral-7B",
        # required=True,
        type=str,
        help="HF Tokenizer name or local tokenizer path.",
    )

    parser.add_argument("--max_tokens", default=500, type=int, help="Max tokens for the chat.")

    # Parse the arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    print(f"> LLM with personality: {args.personality.upper() if args.personality else 'None'}")

    chat_llm = ChatLLM.build(
        model_name=args.model,
        tokenizer=args.tokenizer,
        personality=personalities[args.personality]["personality"] if args.personality else "",
        examples=personalities[args.personality]["examples"] if args.personality else [],
        weights=args.weights,
    )

    chat_llm.run(max_tokens=args.max_tokens)
