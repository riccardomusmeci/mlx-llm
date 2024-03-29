from argparse import ArgumentParser

from mlx_llm.bench.bench import Benchmark


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--apple-silicon", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--quantized", default="false", choices=["true", "false"])
    parser.add_argument("--weights", type=str, default="true")
    parser.add_argument("--prompt", type=str, default="What is the meaning of life?")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--verbose", type=str, default="false", choices=["true", "false"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.quantized = True if args.quantized == "true" else False
    args.weights = True if args.weights == "true" else args.weights
    args.verbose = True if args.verbose == "true" else False
    benchmark = Benchmark(
        apple_silicon=args.apple_silicon,
        quantized=args.quantized,
        weights=args.weights,
        model_name=args.model_name,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        verbose=args.verbose,
    )
    benchmark.start()
    benchmark.save("../results")
