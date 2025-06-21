# Growing BPE Tokenizer Experiments

This directory contains pre-configured experiments for testing different growth strategies and parameters for the growing BPE tokenizer.

## Available Experiments

### 1. Minimal Test (`minimal_test.py`)
**Purpose**: Quick functionality validation and development testing
- **Dataset**: Very small (5K tokens, 50 samples)
- **Vocabulary**: 50 → 200 tokens
- **Growth**: Frequent, simple frequency-based
- **Runtime**: < 1 minute
- **Use case**: Development, debugging, CI/CD

### 2. Aggressive Growth (`aggressive_growth.py`)
**Purpose**: Explore rapid vocabulary expansion limits
- **Dataset**: Moderate (500K tokens)
- **Vocabulary**: 64 → 8,000 tokens
- **Growth**: Multiple triggers, very sensitive thresholds
- **Features**: Pattern mining, frequent growth events
- **Use case**: Understanding maximum growth potential

### 3. Conservative Growth (`conservative_growth.py`)
**Purpose**: Stable, efficient vocabulary development
- **Dataset**: Moderate (500K tokens)
- **Vocabulary**: 1,000 → 2,500 tokens
- **Growth**: Single trigger, high thresholds, patient timing
- **Features**: Token pruning, stability focus
- **Use case**: Production-like scenarios requiring reliability

### 4. Hybrid Strategy (`hybrid_strategy.py`)
**Purpose**: Balanced exploration and exploitation
- **Dataset**: Large (750K tokens)
- **Vocabulary**: 500 → 4,000 tokens
- **Growth**: Multiple triggers, adaptive merging
- **Features**: Moderate settings, linguistic metrics
- **Use case**: General-purpose tokenizer development

### 5. Pattern Exploration (`pattern_exploration.py`)
**Purpose**: Discover linguistically meaningful subword patterns
- **Dataset**: Large (1M tokens)
- **Vocabulary**: 750 → 3,500 tokens
- **Growth**: Pattern mining, information-theoretic selection
- **Features**: Linguistic analysis, case preservation
- **Use case**: Research into subword structure

### 6. Comparison Baseline (`comparison_baseline.py`)
**Purpose**: Reference point for comparative analysis
- **Dataset**: Moderate (600K tokens)
- **Vocabulary**: 512 → 3,000 tokens
- **Growth**: Standard BPE, moderate settings
- **Features**: Balanced, no advanced features
- **Use case**: Baseline for comparing other experiments

## Running Experiments

### Basic Usage
```bash
# Run a specific experiment
python train.py --config experiments/minimal_test.py

# Run with custom parameters
python train.py --config aggressive --max-tokens 100000

# Run built-in configurations
python train.py --config conservative
python train.py --config hybrid
```

### Evaluation
```bash
# Evaluate a trained tokenizer
python evaluate.py --tokenizer checkpoints/minimal_test_experiment/final_tokenizer.json

# Generate visualizations
python evaluate.py --tokenizer checkpoints/aggressive_growth_experiment/final_tokenizer.json --visualize
```

### Batch Experiments
```bash
# Run multiple experiments in sequence
for config in minimal_test aggressive_growth conservative_growth; do
    python train.py --config experiments/${config}.py
    python evaluate.py --tokenizer checkpoints/${config}_experiment/final_tokenizer.json --visualize
done
```

## Experiment Comparison

| Experiment | Runtime | Vocab Growth | Focus | Best For |
|------------|---------|--------------|-------|----------|
| Minimal Test | < 1 min | 50 → 200 | Testing | Development |
| Aggressive | ~10 min | 64 → 8K | Speed | Exploration |
| Conservative | ~15 min | 1K → 2.5K | Stability | Production |
| Hybrid | ~20 min | 500 → 4K | Balance | General use |
| Pattern Exploration | ~25 min | 750 → 3.5K | Quality | Research |
| Baseline | ~15 min | 512 → 3K | Reference | Comparison |

## Creating Custom Experiments

To create a new experiment:

1. Copy an existing experiment file
2. Modify the configuration parameters
3. Update the `run_name` to be unique
4. Adjust comments and documentation

Example:
```python
# experiments/my_experiment.py
from config import GrowingBPEConfig

config = GrowingBPEConfig(
    run_name="my_custom_experiment",
    # ... your custom parameters
)
```

## Key Configuration Parameters

### Growth Triggers
- `compression`: Trigger when compression ratio is poor
- `frequency`: Trigger when frequent patterns aren't tokenized
- `coverage`: Trigger when encountering high OOV rates
- `pattern`: Trigger based on pattern significance

### Growth Mechanisms
- `standard_bpe`: Traditional BPE extension
- `pattern_mining`: Discover new subword patterns
- `adaptive_merging`: Smart merging based on efficiency

### Initial Vocabulary Strategies
- `character_only`: Start with characters only
- `minimal_subwords`: Include common English patterns
- `frequency_based`: Use frequent n-grams
- `linguistic`: Morphologically motivated units

## Results Analysis

After running experiments, you'll find:

- `checkpoints/`: Saved tokenizers and training states
- `plots/`: Visualization of growth patterns
- `final_evaluation.json`: Comprehensive metrics
- `training.log`: Detailed training logs

## Tips for Experimentation

1. **Start Small**: Use `minimal_test` to verify functionality
2. **Compare Systematically**: Run `comparison_baseline` first
3. **Resource Management**: Monitor memory usage for large experiments
4. **Reproducibility**: Keep `deterministic=True` and set `random_seed`
5. **Documentation**: Update experiment descriptions when modifying

## Performance Expectations

| Hardware | Experiment Type | Expected Runtime |
|----------|----------------|------------------|
| CPU Only | Minimal | 30s - 2min |
| CPU Only | Standard | 10min - 30min |
| CPU Only | Large | 30min - 2hr |
| GPU | Any | 2-5x faster |

## Troubleshooting

- **Out of Memory**: Reduce `batch_size` or `max_tokens_to_process`
- **Slow Growth**: Lower `growth_patience` or `frequency_threshold`
- **No Growth**: Check trigger thresholds aren't too restrictive
- **Poor Quality**: Increase `min_pattern_frequency` or enable pruning