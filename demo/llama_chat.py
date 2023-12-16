import argparse
from mlx_llm.model import LLaMA
  
def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="LLaMA chat arguments")
    
    parser.add_argument(
      "--model", 
      default="llama-7B",
      required=False,
      type=str, 
      help="Model name."
    )

    # Add arguments
    parser.add_argument(
      "--ckpt", 
      default=None,
      required=False,
      type=str, 
      help="If specified, first converts the model weights from the checkpoint path."
    )
    
    parser.add_argument(
      "--npz", 
      required=True,
      type=str, 
      help="LLaMA 2 weights path (npz file)."
    )
    
    parser.add_argument(
      "--tokenizer", 
      required=True,
      type=str, 
      help="LLaMA 2 tokenizer path (model file)."
    )
    
    parser.add_argument(
      "--max_tokens", 
      default=500,
      type=int, 
      help="Max tokens for the chat."
    )

    # Parse the arguments
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    
    args = parse_args()
    
    if args.ckpt is not None:
      print(f"> Converting {args.ckpt} to {args.npz}")
      LLaMA.convert(args.ckpt, args.npz)
      
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
        },
        {
            "user": "What is your favorite food?",
            "model": "Beets",
        }
    ]
      
    llama = LLaMA.build(
      model_name=args.model,
      weights_path=args.npz,
      tokenizer_path=args.tokenizer,
      personality=personality,
      examples=examples,
    )
    
    llama.chat(max_tokens=200)
     
    
    
    
    
