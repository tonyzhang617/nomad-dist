# NoMAD-Attention: Efficient LLM Inference on CPUs Through Multiply-add-free Attention

Large language model (LLM) inference on Central Processing Units (CPU) is challenging due to the vast quantities of expensive Multiply-add (MAD) matrix operations in the attention computations. Therefore, we leverage in-register shuffles, a unique capability of CPUs, to propose NoMAD-Attention, an efficient attention algorithm that replaces MAD operations with lookups. Through hardware-aware algorithmic designs, NoMAD-Attention achieves the computation of attention scores using repeated fast accesses to SIMD registers despite their highly limited sizes. Moreover, NoMAD-Attention works with pre-trained attention-based LLMs without model finetuning. Empirical evaluations demonstrate that NoMAD-Attention maintains the quality of the original LLMs well, and speeds up the 4-bit quantized LLaMA-7B-based model by up to 2x at 16k context length. 

**Note** This repository only contains the executable binaries for NoMAD-Attention, with no source code. We are in the process of patent application for NoMAD-Attention. Thank you for your understanding.

## System Requirements

1. Linux

2. A CPU that supports Advanced Vector Extensions 2 (AVX2). To check whether your CPU supports AVX2, run `lscpu | grep "avx2"`, which will output one or more matched entries if the CPU supports AVX2, and none otherwise.

3. OpenBLAS. On Ubuntu, run `sudo apt-get install libopenblas-dev` to install it.

## Quick Start

This section provides an example of running CodeLLaMA-7B with NoMAD-Attention. The `assets` folder contains the codebooks of some models that have been learned for NoMAD-Attention, including CodeLLaMA-7B, LLaMA-2-7B, and StableLM-3B-4E1T. First, we download the CodeLLaMA-7B model with
```bash
cd models
bash download_codellama.sh
cd ..
```

Use the following command to run CodeLLaMA-7B with NoMAD-Attention ($d_\mathrm{sub}=1$) to generate 16384 tokens.
```bash
./app/bin/main -m models/codellama-7b.Q4_0.gguf -n 16384 -pi assets/codellama-7b-dsub1 2> codellama_nomad.log
```

The decoding latency for each token is written to the log file `codellama_nomad.log`. Let's compare with the speed of the original dot-product attention by running the following command, which writes to the log file `codellama_attn.log`.

```bash
./app/bin/main -m models/codellama-7b.Q4_0.gguf -n 16384 2> codellama_attn.log
```

## Datasets and Models

Install the required packages:
```bash
pip install -r requirements.txt
```

Download the datasets `WikiText-2` and `PTB` for perplexity testing:
```bash
cd data
python download.py
cd ..
```

Download the models LLaMA-2-7B and StableLM-3B-4E1T:
```bash
cd models
bash download_llama2.sh
bash download_stablelm.sh
```

To download LLaMA-2, you need to obtain [an API key](https://llama.meta.com/llama-downloads) from META first. 

## Result Reproduction

To reproduce the perplexity results from our paper, use the following commands.

Perplexity of StableLM-3B-4E1T (q8_0 quantized) with NoMAD-Attention ($d_\mathrm{sub}=1$) on `WikiText-2` and `PTB`:
```bash
./app/bin/perplexity -m models/codellama-7b.Q8_0.gguf -pi assets/stablelm-3b-dsub1 -f data/wikitext-2-raw/wiki.test.raw -c 512
./app/bin/perplexity -m models/codellama-7b.Q8_0.gguf -pi assets/stablelm-3b-dsub1 -f data/ptb/test.txt -c 512
```

Perplexity of StableLM-3B-4E1T (q8_0 quantized) with the original Attention on `WikiText-2` and `PTB`:
```bash
./app/bin/perplexity -m models/codellama-7b.Q8_0.gguf -f data/wikitext-2-raw/wiki.test.raw -c 512
./app/bin/perplexity -m models/codellama-7b.Q8_0.gguf -f data/ptb/test.txt -c 512
```

## Using Your LLM with NoMAD-Attention

Adapting your LLM to work with NoMAD-Attention requires no model training. But new codebooks need to be learned first for a new model to work with NoMAD-Attention. To learn codebooks, we first perform model inference on a learning corpus and save its attention key embeddings for each layer and each head separately. Then we perform k-means clustering on the saved attention key embeddings to determine the centroids in each sub-quantizer. This section provides instructions on how your own model can be adapted to NoMAD-Attention by learning codebooks. Currently, NoMAD-Attention are supported for LLaMA-based and StableLM-based LLMs with no multi-query attention.

First, we perform model inference on a corpus and save the attention keys. The corpus does not need to be large in size to ensure good model quality. For example, the following command saves the attention keys of the model `codellama-7b` on the validation set of `WikiText-2` to the directory `assets/codellama-7b-wikitext2-valid-keys`. Warning: this will create a large number of files under the directory. 

```bash
./app/bin/perplexity -m models/codellama-7b.Q4_0.gguf -c 512 -f data/wikitext-2-raw/wiki.valid.raw -psi assets/codellama-7b-wikitext2-valid-keys
```
where `-m` is the model path, `-c` is the context length, `-f` is the text file containing the learning corpus, and `-psi` or `--path-save-index` is the directory in which to save the attention key embeddings (has to be created first).

Next, we perform k-means on the saved attention keys to learn codebooks using the following command, which reads from the directory `assets/codellama-7b-wikitext2-valid-keys` and saves the codebooks to the directory `assets/codellama-7b-wikitext2-valid-codebooks`.

```bash
python learn_codebooks.py --paths assets/codellama-7b-wikitext2-valid-keys --save_path assets/codellama-7b-wikitext2-valid-codebooks --range 0 1024 --d_sub 1 --niter 100 --dim 128
```
where `--paths` is a list of paths containing attention keys, `--save_path` is the directory under which to save the codebooks, `--range` is a pair of numbers (0, num_of_layers * num_of_attn_heads) (codellama-7b has 32 layers and 32 attention heads, hence 1024), `--d_sub` is the dimension in each sub-quantizer (1 preserves model quality well), `--niter` is the number of iterations to run k-means for, and `--dim` is the dimensionality of the attention key embeddings.

Now, the new model is ready with NoMAD-Attention. To try it out, run the following.
```bash
./app/bin/main -m models/codellama-7b.Q4_0.gguf -n 1024 -pi assets/codellama-7b-wikitext2-valid-codebooks -p "What does the const keyword mean in C++? Answer: " 2> /dev/null
```
where `-p` is the prompt to start generation with.