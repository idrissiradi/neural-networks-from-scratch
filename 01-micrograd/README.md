# 01 — Micrograd: Autograd Engine from Scratch

## What This Is Based On
Karpathy's "The spelled-out intro to neural networks and 
backpropagation: building micrograd"  

---

## Key Concepts

### Backpropagation
The training loop has two passes:
- **Forward pass**: data flows through the network, 
  we get a prediction and compute the loss
- **Backward pass**: we compute the gradient of the loss 
  with respect to every parameter (weights and biases) 
  using the chain rule
  
The gradients tell us in which direction each parameter 
is pushing the loss. The optimizer (gradient descent) then 
uses those gradients to update the parameters and reduce the loss.

Backprop doesn't give you the final values of w and b directly.

it gives you the gradients. Gradient descent does the actual update.

### Computational Graph
A directed acyclic graph (DAG) where:
- Each **node** is a value (scalar)
- Each **edge** represents an operation (+, *, tanh, etc.)
- **Forward pass** builds the graph left to right
- **Backward pass** traverses it right to left, 
  computing gradients at each node via the chain rule

It makes the full cycle visible: 
input → operations → loss → gradients → back to inputs.

---

## Stack
Python · NumPy · PyTorch (comparison only)