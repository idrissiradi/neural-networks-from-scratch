# 03 — Makemore MLP: Multi-Layer Perceptron Language Model from Scratch

## What This Is Based On
Karpathy's "makemore" series, Part 2: Building a character-level language model using a Multi-Layer Perceptron (MLP).


---

## Key Concepts

### Embeddings
Instead of one-hot encoding, we use learned embeddings to represent characters in a continuous vector space. This allows the model to capture relationships between characters.

### Multi-Layer Perceptron (MLP)
A neural network with multiple layers: input (embeddings), hidden layers (with tanh activation), and output (logits for next character prediction).

### Training
- **Forward Pass**: Embed characters, pass through MLP to get predictions.
- **Loss**: Cross-entropy loss between predicted and actual next characters.
- **Backward Pass**: Use backpropagation (via PyTorch's autograd) to compute gradients.
- **Optimization**: Gradient descent to update weights and embeddings.

### Generation
Sample from the softmax probabilities at each step to generate sequences, similar to the bigram model but with richer context.

---

## Stack
Python · NumPy · PyTorch