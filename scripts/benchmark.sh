#!/bin/sh

## declare an array variable
declare -a arr=("Phi2" "LLaMA-2-7B-chat" "TinyLlama-1.1B-Chat-v0.6", "Mistral-7B-Instruct-v0.2", "OpenHermes-2.5-Mistral-7B", "zephyr-7b-beta")

## now loop through the above array
for i in "${arr[@]}"
do
   python run_bench.py --apple-silicon m1_pro_32GB --model-name $i
   rm -r ~/.cache/huggingface/hub # remove cache (save disk space)
done
