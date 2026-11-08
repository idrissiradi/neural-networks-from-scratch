# 09 — Let's Reproduce GPT-2 (124M)

This project follows Andrej Karpathy's video ["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU). It builds a GPT-2 style language model in PyTorch and a training loop with gradient accumulation and optional multi-GPU distributed training.

## What I Built

- A GPT-2 Transformer with causal self-attention, MLP blocks, learned token and position embeddings, and tied input/output embeddings.
- A lightweight data loader that encodes `input.txt` with the GPT-2 tokenizer and creates next-token prediction batches.
- A training loop with AdamW, weight decay, gradient clipping, mixed precision, and a warmup plus cosine learning-rate schedule.
- Optional PyTorch DistributedDataParallel (DDP) training. Each process reads its own portion of the token stream, and gradients synchronize across processes.
- Gradient accumulation to reach a target batch size of 524,288 tokens.

## Key Concepts

- **Next-token prediction:** train the model to predict each token from the preceding context.
- **Causal self-attention:** each position can attend only to earlier positions, preserving the autoregressive setup.
- **Training at scale:** gradient accumulation and DDP make a large effective batch possible across devices.
- **GPT-2 implementation details:** tied embeddings, GELU activations, layer normalization, and carefully initialized residual projections.


## Stack

Python · PyTorch · tiktoken
