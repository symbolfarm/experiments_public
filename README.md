# Symbol Farm - Public Machine Learning Experiments

These experiments will make use of coding agents (e.g., Claude Code) to rapidly prototype machine learning experiments.
Experiments will be able to run locally on consumer hardware, i.e., 24GB VRAM or less.

# Experiments

## Cascade-Correlation Transformer

`./01_cascade_correlation`

Cascade-Correlation is a constructive algorithm for artificial neural networks that was developed by Fahlman & Lebiere in 1989.
In the original algorithm construction would produce a single new neuron when construction conditions were met.
In this version of the algorithm a construction event produces a transformer block (attention and MLP).

