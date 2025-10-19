# Cascade-Correlation Transformer

A minimal implementation of a hybrid cascade-correlation neural network using transformer decoder blocks, designed for language modeling on the TinyStories dataset.

## Overview

This project combines the dynamic network construction principles of Cascade-Correlation (Fahlman & Lebiere, 1990) with modern transformer architecture. Instead of adding individual neurons, we dynamically add entire transformer decoder blocks (attention + MLP) during training.

## Key Features

- **Hybrid Architecture**: Combines cascade-correlation with transformer decoder blocks
- **Dynamic Growth**: Network starts small and grows by adding transformer blocks
- **Efficient Training**: Designed to stay under 100M parameters total
- **TinyStories Dataset**: Trained on the simplified story generation task
- **RTX 4090 Optimized**: Configured for single GPU training

## Quick Start

```bash
# Install dependencies
pip install torch datasets transformers matplotlib

# Download and prepare TinyStories dataset
python data.py

# Train the model
python train.py

# Monitor training progress
# The model will automatically add new transformer blocks when improvement plateaus
```

## Architecture

The network consists of:
- **Input embedding layer** with positional encoding
- **Dynamically added transformer blocks** (attention + MLP)
- **Output projection layer** for token prediction

Each transformer block is small (e.g., 4-8 attention heads, 256-512 hidden dimensions) to keep the total parameter count manageable while allowing for growth.

## Training Process

1. **Initialize**: Start with embedding layers and output projection
2. **Train**: Train the current network until performance plateaus
3. **Grow**: Add a new transformer block and freeze previous weights
4. **Repeat**: Continue until convergence or parameter budget reached

## Files

- `model.py`: Cascade-correlation transformer implementation
- `train.py`: Training loop with dynamic block addition
- `config.py`: Hyperparameter configuration
- `data.py`: TinyStories dataset loading and preprocessing
- `DESIGN.md`: Detailed technical design document

## Configuration

Key hyperparameters in `config.py`:
- `max_blocks`: Maximum number of transformer blocks (default: 12)
- `d_model`: Model dimension per block (default: 256)
- `n_heads`: Number of attention heads per block (default: 4)
- `patience`: Epochs to wait before adding new block (default: 5)
- `growth_threshold`: Performance improvement threshold for growth

## Performance Monitoring

The training script includes:
- Loss tracking per block addition
- Parameter count monitoring
- Visualization of network growth
- Evaluation metrics on validation set

## Research Context

This implementation explores whether the constructive learning principles of cascade-correlation can benefit modern transformer architectures. The hypothesis is that starting small and growing the network based on learning progress might lead to more efficient parameter usage and better generalization.

## References

- Fahlman, S. E., & Lebiere, C. (1990). The cascade-correlation learning architecture. NIPS.
- Eldan, R., & Li, Y. (2023). TinyStories: How small can language models be and still speak coherent English?