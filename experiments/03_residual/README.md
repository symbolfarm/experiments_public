# Residual Transformer Experiment

This experiment implements and evaluates a novel transformer architecture with **fixed token embeddings** and an evolving **residual state** that acts as working memory.

## Architecture Overview

### Core Innovation
- **Fixed token embeddings**: Token embeddings remain constant throughout all processing blocks
- **Evolving residual state**: A fixed-length state vector that gets updated between blocks
- **Dual attention pattern**: Processing blocks vs final readout block work differently

### Processing Blocks (Blocks 1 to N-1)
1. Token embeddings (fixed) → Query via layer-specific W_q
2. Residual state → Key, Value 
3. Attention output used to update residual state via cross-attention
4. Token embeddings unchanged

### Final Readout Block
1. Residual state → Query
2. Token embeddings (fixed) → Key, Value
3. Attention output updates token embeddings for the first time
4. FFN processes updated token embeddings
5. Output projection to vocabulary

## Key Benefits
- **Linear complexity**: O(n) instead of O(n²) for long sequences
- **Information bottleneck**: Forces efficient information compression
- **Theoretical elegance**: Clear separation between working memory and input representation

## Files

- `config.py`: Configuration dataclass with all hyperparameters
- `model.py`: ResidualTransformer implementation
- `baseline_model.py`: Standard transformer for comparison
- `data.py`: TinyStories dataset loading and preprocessing
- `train.py`: Training script for both models
- `evaluate.py`: Comprehensive model comparison
- `test_models.py`: Quick functionality test

## Usage

### Quick Test
```bash
python test_models.py
```

### Training
```bash
# Train both models
python train.py --model_type both --batch_size 32 --max_epochs 20

# Train only residual model
python train.py --model_type residual --fixed_kv_length 64 --n_processing_blocks 8

# Train only baseline
python train.py --model_type baseline
```

### Evaluation
```bash
python evaluate.py \
  --residual_checkpoint checkpoints/residual_best_model.pt \
  --baseline_checkpoint checkpoints/baseline_best_model.pt \
  --save_dir evaluation_results
```

## Key Hyperparameters

- `fixed_kv_length`: Size of residual state (32, 64, 128, 256)
- `n_processing_blocks`: Number of processing blocks before readout
- `residual_init_strategy`: How to initialize residual state ("random", "pooled", "learned")
- `d_model`: Model dimension
- `n_heads`: Attention heads

## Experimental Questions

1. **Does the architecture learn anything useful?** Can it match baseline performance?
2. **Information bottleneck**: How does `fixed_kv_length` affect performance?
3. **Scaling behavior**: Does efficiency improve with longer sequences?
4. **Initialization sensitivity**: Which residual initialization works best?
5. **Working memory**: Does the residual state capture meaningful information?

## Expected Results

- **Efficiency**: Should see memory/compute advantages for longer sequences
- **Performance**: May underperform baseline initially but could be competitive
- **Compression**: Residual state should learn to compress sequence information
- **Scaling**: Benefits should increase with sequence length

## Current Status

✅ Architecture implemented  
✅ Training script ready  
✅ Evaluation framework ready  
🔄 Ready for experimentation  

## Next Steps

1. Run small-scale experiments to verify learning
2. Sweep hyperparameters (especially `fixed_kv_length`)
3. Test on longer sequences to see efficiency gains
4. Analyze what the residual state learns
5. Compare with other efficient attention mechanisms