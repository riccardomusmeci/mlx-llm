#!/bin/sh

## declare an array variable
declare -a arr=("LLaMA-2-7B-chat" "Mistral-7B-Instruct-v0.2" "OpenHermes-2.5-Mistral-7B" "Phi2" "TinyLlama-1.1B-Chat-v0.6")

## now loop through the above array
for i in "${arr[@]}"
do
   # set verbose to false otherwise print statements will slow down the benchmark
   python run_bench.py --apple-silicon m1_pro_32GB --model-name $i --verbose false
   rm -r ~/.cache/huggingface/hub # remove cache (save disk space)
done
