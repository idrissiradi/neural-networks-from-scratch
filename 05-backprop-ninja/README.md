# 04 — makemore Part 4: Becoming a Backprop Ninja

## What This Is Based On
Karpathy's "Building makemore Part 4: Becoming a Backprop Ninja"  

---

## What I Built
Manually implemented the full backward pass of the MLP from Video 4 —
every gradient derived by hand and verified against PyTorch autograd.
Final exercise: replaced loss.backward() entirely with manual gradients.
Model trained identically to the PyTorch version.

---

## Key Concepts

### Key Derivatives

**Cross-entropy + softmax (compressed):**
dlogits = softmax(logits)
dlogits[range(n), Yb] -= 1
dlogits /= n

**Linear layer:**
dx = dout @ W.T
dW = x.T @ dout
db = dout.sum(0)

**Tanh:**
dx = (1 - out**2) * dout

**BatchNorm (compressed):**
dhprebn = bngain*bnvar_inv/n * (n*dbnraw - dbnraw.sum(0) - bnraw*(dbnraw*bnraw).sum(0))

---

### Key Insight
Approximate gradient match (maxdiff < 1e-9) is correct.
floating point operations are not commutative so exact match
is not expected. PyTorch autograd and manual gradients take
different computation paths.

---

## Stack
Python · PyTorch