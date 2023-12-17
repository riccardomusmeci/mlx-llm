import argparse
from mlx_llm.model import LLM
from personality import personalities
  
def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="LLM chat arguments")
    
    parser.add_argument(
      "--personality",
      default="dwight",
      type=str,
      choices=list(personalities.keys()),
    )
    
    # parser.add_argument(
    #   "--model", 
    #   default="Mistral-7B-Instruct-v0.1",
    #   required=False,
    #   type=str, 
    #   help="Model name."
    # )
    
    # parser.add_argument(
    #   "--weights", 
    #   required=True,
    #   type=str, 
    #   help="Mistral weights path (npz file)."
    # )
    
    # parser.add_argument(
    #   "--tokenizer", 
    #   required=True,
    #   type=str, 
    #   help="Mistral tokenizer path (model file)."
    # )
    
    # parser.add_argument(
    #   "--max_tokens", 
    #   default=500,
    #   type=int, 
    #   help="Max tokens for the chat."
    # )

    # Parse the arguments
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    
    args = parse_args()
    
    print(f"> LLM with personality: {args.personality.upper()}")
    
    # Mistral-7B-v0.2-Instruct, llama-2-7b-chat
    model_name = "Mistral-7B-v0.2-Instruct"

    weights = f"/Users/riccardomusmeci/Developer/data/github/mlx_llm/weights/{model_name}/weights.npz"
    tokenizer = f"/Users/riccardomusmeci/Developer/data/github/mlx_llm/weights/{model_name}/tokenizer.model"
      
    llm = LLM.build(
      model_name=model_name,
      weights_path=weights,
      tokenizer_path=tokenizer,
      personality=personalities[args.personality]["personality"],
      examples=personalities[args.personality]["examples"],
    )
    
    llm.chat(max_tokens=200)
