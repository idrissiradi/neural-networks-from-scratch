# 04 — makemore Part 3: Activations & Gradients, BatchNorm

## What This Is Based On
Karpathy's "Building makemore Part 3: Activations & Gradients, BatchNorm"  

---

## What I Built
I built a character-level language model that takes 3 previous characters as context to predict the next one. The architecture is: embeddings → hidden layer (tanh) → output (softmax). I implemented the full training cycle: forward pass, loss, backward pass, parameter update. I also added BatchNorm to the hidden layer and compared it with the non-BatchNorm version.

---

## Key Concepts

**Kaiming Initialization**: initializing W1 with `* (5/3) / sqrt(fan_in)` kept 
activations at std≈1 through the hidden layer. Without it activations explode 
before BatchNorm can fix them, starting training at a much higher loss.

**BatchNorm**: normalizes hidden layer inputs to mean=0 std=1 per batch during 
training, then uses running mean/std at inference. Keeps activation distributions 
healthy across layers and gradient flow stable. The gain and bias parameters 
(bngain, bnbias) let the model learn to scale and shift after normalization.

**Activation & Gradient Statistics**: plotted std of activations and gradients 
across layers to verify healthy flow. Target: consistent std throughout, no 
saturation at tanh (bimodal spike at ±1 = bad). Update-to-data ratio should 
stay around 1e-3.

---

## Stack
Python · PyTorch