#!/bin/bash

# stop on error
set -e

# Step 1: Execute StableLM with NoMAD-Attention
echo "Executing StableLM with NoMAD-Attention..."
taskset -c 0-23 ./app/bin/main -m models/stablelm-3b-4e1t.Q4_0.gguf -n 16384 -pi assets/stablelm-3b-dsub1 2> stablelm_nomad.log

# Step 2: Compare Performance with Original Attention
echo "Comparing performance with original dot-product attention..."
taskset -c 0-23 ./app/bin/main -m models/stablelm-3b-4e1t.Q4_0.gguf -n 16384 2> stablelm_attn.log

# Step 3: Perform Model Inference on a Learning Corpus for StableLM
echo "Performing model inference on a learning corpus for StableLM..."
# make directory for storing attention keys
mkdir -p assets/stablelm-3b-wikitext2-valid-keys
taskset -c 0-23 ./app/bin/perplexity -m models/stablelm-3b-4e1t.Q4_0.gguf -c 512 -f data/wikitext-2-raw/wiki.valid.raw -psi assets/stablelm-3b-wikitext2-valid-keys

# Step 4: Learn Codebooks via k-means Clustering for StableLM
echo "Learning codebooks via k-means clustering for StableLM..."
# taskset -c 0-23 python learn_codebooks.py --paths assets/stablelm-3b-wikitext2-valid-keys --save_path assets/stablelm-3b-wikitext2-valid-codebooks --range 0 1024 --d_sub 1 --niter 100 --dim 80
taskset -c 0-23 python learn_codebooks.py --paths assets/stablelm-3b-wikitext2-valid-keys --save_path assets/stablelm-3b-wikitext2-valid-codebooks --range 0 1024 --d_sub 1 --niter 100

# Step 5: Test StableLM with NoMAD-Attention
echo "Testing StableLM with NoMAD-Attention..."
taskset -c 0-23 ./app/bin/main -m models/stablelm-3b-4e1t.Q4_0.gguf -n 1024 -pi assets/stablelm-3b-wikitext2-valid-codebooks -p "What does the const keyword mean in C++? Answer: " 2> /dev/null

echo "All steps completed."
