# 02 — makemore bigram:

## What This Is Based On
Karpathy's "The spelled-out intro to language modeling: building makemore"  

---

## What I Built
A bigram and trigram build a character-level bigram language model from scratch using PyTorch. trained on a dataset of names. Built using PyTorch tensors. covering the full language modeling framework: splitting data, training a model with gradient descent, sampling new names, and evaluating quality using negative log-likelihood loss for classification(also using cross_entropy). Also extended the bigram to a trigram neural net with a concatenated one-hot input and compared both models.

---

The included `names.txt` dataset, as an example, has the most common 32K names takes from [ssa.gov](https://www.ssa.gov/oact/babynames/) for the year 2018. It looks like:

```
emma
olivia
ava
isabella
sophia
charlotte
...
```
---

## Key Concept
**The bigram**: samples from dataset based on probabilities, it takes one character as input to predict next character based on the current character only

**NLL**: negative log likelihood is negative of log of probabilities assigned to the correct character, it measures how well the model is doing in predicting the next character. The lower the NLL, the better the model's predictions are.

**Log probabilities**: to avoid numerical underflow, we use Log to converts multiplication into addition, and also makes the loss function more stable and easier to optimize.

---

## Stack
Python · PyTorch