# 06 — makemore Part 5: WaveNet

## What This Is Based On
Karpathy's "Building makemore Part 5: WaveNet"

---

## What I Built
A hierarchical character-level language model inspired by DeepMind's WaveNet.
Instead of flattening all context into one big hidden layer, the model combines
characters progressively as a binary tree, pairs first, then pairs of pairs,
then everything together.

Architecture (context length = 8):
  Embedding (24-dim)
  FlattenConsecutive(2) → Linear(48→128) → BatchNorm → Tanh   [level 1]
  FlattenConsecutive(2) → Linear(256→128) → BatchNorm → Tanh  [level 2]
  FlattenConsecutive(2) → Linear(256→128) → BatchNorm → Tanh  [level 3]
  Linear(128→27)

## Why Hierarchical Scales Better
Flat MLP: doubling context length doubles the first hidden layer; expensive.
WaveNet: doubling context length adds one level to the tree; just one more
FlattenConsecutive + Linear. Context length 8 → 3 levels. Context 16 → 4 levels.
Cost grows as log2(n) not linearly.

## Key Insight — Convolution Preview
FlattenConsecutive + Linear applied at every position is equivalent to a 1D
convolution with kernel size 2. Same weights, slid across all positions.
Andrej previews this at the end, a convolution is just an efficient for loop
over positions. GPT replaces the fixed sliding window with attention, dynamic
weights that depend on content, not position.

## Sample Generated Names
hendrix, jaylene, aubreana, marianah, jayce, rocklei

## Stack
Python · PyTorch