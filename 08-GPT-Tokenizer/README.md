# 08 — GPT Tokenizer

## What This Is Based On
Karpathy's "Let's build the GPT Tokenizer"

---

## What I Built
This project implements a GPT-style tokenizer using byte-pair encoding (BPE), following the approach used in modern language models. I built the core tokenization pipeline for encoding text into integer IDs, decoding IDs back to text, and training a custom vocabulary from a corpus of text data.

The work highlights how tokenizers turn raw strings into model-ready inputs, and how vocabulary construction affects efficiency, compression, and downstream model performance.

## Key Insights
- **tokenization is a model input design choice, not just preprocessing**
- **BPE learns frequent character sequences and merges them into tokens**
- **vocabulary size affects compression and representation quality**
- **encoding/decoding must preserve text exactly for reliable training and generation**
- **small corpora can still demonstrate the core tokenizer mechanics used in large language models**

## Stack
Python · PyTorch