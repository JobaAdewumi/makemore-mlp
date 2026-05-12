# makemore-mlp

This repository contains a Jupyter notebook implementation of a simple character-level neural language model trained on a list of names, together with the original reference paper used as inspiration.

## Contents

- `makemore-mlp.ipynb` — Jupyter notebook implementing the model and training loop.
- `names.txt` — dataset of names used to build the character language model.
- `bengio03a.pdf` — paper: "A Neural Probabilistic Language Model" by Yoshua Bengio et al. (2003).

## Project Overview

The notebook builds a neural model that learns to predict the next character in a name given a fixed-length context of previous characters. It is a lightweight demonstration of the following concepts:

- character-level language modeling
- embedding discrete symbols as continuous vectors
- using a feed-forward neural network for sequence prediction
- training with cross-entropy loss and minibatch SGD
- splitting data into training, validation, and test sets

## Dataset

The dataset `names.txt` contains one name per line. The notebook:

1. reads all names
2. builds a vocabulary of characters plus the end-of-name token `.`
3. maps each character to an integer index
4. creates examples where each training target is the next character following a 3-character context

The data is shuffled and split into:

- 80% training
- 10% development/validation
- 10% test

## Model Architecture

The notebook implements a small neural network with the following steps:

1. learnable character embeddings for each vocabulary symbol
2. flatten the embedding vectors for the context window
3. apply a hidden layer with `tanh` activation
4. compute logits for the next character prediction
5. softmax the logits to obtain a probability distribution

The learned parameters include:

- embedding matrix `C`
- hidden weights `W1` and bias `b1`
- output weights `W2` and bias `b2`

## Training

The model is trained using a basic manual SGD loop in the notebook:

- minibatch size: 32
- total iterations: 200,000
- learning rate: 0.1 for the first 100,000 iterations, then 0.01
- loss: `torch.nn.functional.cross_entropy`

Training statistics are tracked and visualized with a loss curve.

## Evaluation

After training, the notebook computes cross-entropy loss on both:

- the training set
- the development/validation set

This gives a quick measure of model fit and generalization.

## Paper Inspiration

The included paper `bengio03a.pdf` is the original work by Bengio et al. on neural probabilistic language modeling. Key ideas from the paper that influenced this notebook:

- use of distributed vector representations for discrete symbols
- modeling conditional probabilities with a neural network
- fighting the curse of dimensionality through learned embeddings
- estimating a probability function for sequences as a product of conditional probabilities

This notebook adapts those ideas to a small character-level setting instead of the original word-level model.

## Requirements

- Python 3
- `torch`
- `matplotlib`

## Usage

Open `makemore-mlp.ipynb` in Jupyter or VS Code and run the cells sequentially.

## Notes

- The notebook uses a fixed context length of 3 characters.
- The end-of-name token `.` is used to mark the end of a name.
- The project is primarily educational and meant to illustrate the basic mechanics of a neural language model.

## References

- Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin, "A Neural Probabilistic Language Model", Journal of Machine Learning Research, 2003.
