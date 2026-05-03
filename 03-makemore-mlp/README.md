# 03 — Makemore MLP: Multi-Layer Perceptron Language Model from Scratch

## What This Is Based On
Karpathy's "makemore" series, Part 2: Building a character-level language model using a Multi-Layer Perceptron (MLP).

---

## What I Built

I built a character-level language model that takes 3 previous characters as context to predict the next one.Unlike the bigram model (1 character → next character), this MLP uses richer context.
**The architecture is: embeddings → hidden layer (tanh) → output (softmax).**

I implemented the full training cycle: **forward pass**, **loss**, **backward pass**, **parameter update**. 
tuning initialization and added L2 regularization to reduce overfitting.

---

## Key Concepts

### Embeddings
Instead of one-hot encoding; which treats every character as equally distant from every other; I use learned embeddings to represent each character as a dense vector. This allows the model to capture similarity between characters during training.

### Training Loop
I started by splitting the dataset into train / dev / test sets. The training runs for 200,000 iterations. Each iteration starts with a forward pass, computes cross-entropy loss with L2 regularization, then runs a backward pass, and ends with updating the parameters. Performance is evaluated by comparing loss on the train and dev sets.

### Initialization
Proper initialization matters. A bad initialization (random b2, unscaled W2) gave a starting loss of 18.03, far worse than random guessing. 
After tuning (W2 * 0.01, b2 = zeros), the starting loss dropped to ~3.45, close to the theoretical uniform baseline of 3.296. This helped training converge faster and more reliably.

### Generation
Instead of using 1 previous character like the bigram model, the MLP uses the last 3 characters as context to predict the next one. At each generation step, we sample from the softmax probabilities of the output.

---

## Hyperparameters

| Hyperparameter              | Value      |
| --------------------------- | ---------- |
| Vocabulary size             | 27         |
| Context window (block_size) | 3          |
| Embedding dim               | 10         |
| Hidden size                 | 200        |
| Iterations                  | 200,000    |
| Batch size                  | 32         |
| Learning rate               | 0.1 → 0.01 |
| L2 lambda                   | 0.001      |

---
## Results

| Run                         | Train Loss | Dev Loss |
| --------------------------- | ---------- | -------- |
| Baseline (Karpathy best)    | —          | 2.2000   |
| My implementation           | 2.0743     | 2.1305   |
| After init fix (W2 * 0.01)  | 2.0684     | 2.1360   |
| W2 * 0.01, b2 = zeros       | 2.0662     | 2.1287   |
| + L2 regularization (0.001) | 2.0706     | 2.1271   |

---

## What I Extended

- **Initialization tuning**: Fixed W2 and b2 initialization. 
  Bad init gave starting loss of 18.03. After W2 * 0.01 
  and b2 = zeros, starting loss dropped to ~3.45, close 
  to the theoretical uniform baseline of 3.296.

- **L2 Regularization**: Added Ridge regularization from 
  Bengio et al. 2003. Improved dev loss from 2.1287 → 2.1220 
  and reduced the train/dev gap from 0.0625 → 0.0512.

---

## Stack
Python · PyTorch