# Growing BPE Tokenizer

A dynamic Byte-Pair Encoding (BPE) tokenizer that grows its vocabulary during operation, implementing constructive learning principles similar to cascade-correlation but for tokenization.

## Overview

This experiment explores adaptive tokenization where the vocabulary expands dynamically based on the characteristics of text being processed. Unlike traditional static tokenizers trained once on a corpus, this growing tokenizer continuously learns new subword patterns and extends its vocabulary.

## Key Features

- **Dynamic Vocabulary Growth**: Starts with minimal vocabulary and adds tokens based on configurable criteria
- **Multiple Growth Strategies**: Configurable triggers and mechanisms for vocabulary expansion
- **From-Scratch Implementation**: Educational BPE implementation without external tokenizer libraries
- **Comprehensive Evaluation**: Multiple metrics to assess tokenization quality and growth effectiveness
- **TinyStories Integration**: Designed for the simplified story generation domain
- **Experimental Framework**: Supports A/B testing of different growth configurations

## Quick Start

```bash
# Install dependencies
pip install torch datasets matplotlib

# Download and prepare TinyStories dataset
python data.py

# Train a growing tokenizer with default settings
python train.py

# Experiment with different growth strategies
python train.py --config experiments/aggressive_growth.py
python train.py --config experiments/conservative_growth.py

# Evaluate tokenizer performance
python evaluate.py --tokenizer checkpoints/growing_bpe_latest/
```

## Growth Strategies

### Growth Triggers
- **Compression-Based**: Add tokens when compression ratio drops below threshold
- **Frequency-Based**: Add frequently occurring character n-grams
- **Coverage-Based**: Add tokens when encountering high OOV rates
- **Hybrid**: Combine multiple trigger conditions

### Growth Mechanisms
- **Standard BPE**: Extend merge rules using traditional BPE algorithm
- **Pattern Mining**: Discover new subword patterns in recent text
- **Token Synthesis**: Create new tokens by combining/modifying existing ones
- **Adaptive Merging**: Dynamically adjust merge priorities

### Initial Vocabularies
- **Character-Only**: Start with basic characters and punctuation
- **Minimal Subwords**: Include common English prefixes/suffixes
- **Frequency-Based**: Use most common n-grams from sample text
- **Linguistic**: Include morphologically motivated units

## Architecture

The system consists of:
- **Core BPE Engine**: Custom implementation of byte-pair encoding
- **Growth Monitor**: Tracks metrics and triggers vocabulary expansion
- **Pattern Analyzer**: Identifies candidates for new tokens
- **Evaluation Suite**: Comprehensive tokenization quality assessment

## Files

- `tokenizer.py`: Core growing BPE tokenizer implementation
- `train.py`: Training loop with dynamic vocabulary expansion
- `config.py`: Configuration options for all growth strategies
- `data.py`: TinyStories dataset loading and text preprocessing
- `evaluate.py`: Evaluation metrics and analysis tools
- `experiments/`: Pre-configured growth strategy experiments
- `DESIGN.md`: Detailed technical design document

## Configuration

Key parameters in `config.py`:
- `initial_vocab_strategy`: How to build starting vocabulary
- `growth_trigger`: When to expand vocabulary
- `growth_mechanism`: How to add new tokens
- `max_vocab_size`: Upper limit on vocabulary size
- `growth_patience`: How long to wait before considering growth
- `evaluation_metrics`: Which metrics to track

## Evaluation Metrics

### Compression Efficiency
- **Compression Ratio**: Average tokens per character
- **Vocabulary Utilization**: Percentage of vocabulary actively used
- **Token Frequency Distribution**: Analysis of token usage patterns

### Quality Measures
- **Subword Coherence**: Semantic meaningfulness of learned subwords
- **Morphological Alignment**: Correspondence with linguistic units
- **Domain Adaptation**: Performance on different text types

### Growth Dynamics
- **Growth Timeline**: When and why vocabulary expanded
- **Stability**: How often tokens become unused after addition
- **Efficiency**: Parameter overhead vs. performance gain

## Experimental Hypotheses

### Primary Hypothesis
Dynamic vocabulary growth will achieve better compression and more meaningful subword units compared to static tokenizers of equivalent size.

### Secondary Hypotheses
1. Different growth strategies will excel in different scenarios
2. Larger initial vocabularies will require less subsequent growth
3. Growth patterns will reflect linguistic structure of the domain
4. Optimal growth timing will correlate with text complexity changes

## Research Context

This work extends BPE tokenization with constructive learning principles:
- **Static Limitation**: Traditional tokenizers can't adapt to new domains
- **Constructive Learning**: Build complexity incrementally based on need
- **Dynamic Systems**: Allow tokenizers to evolve with their input
- **Efficiency Focus**: Add complexity only when beneficial

## Performance Monitoring

The training includes:
- Real-time compression ratio tracking
- Vocabulary growth visualization
- Token usage frequency analysis
- Comparative evaluation against static baselines

## Future Extensions

1. **Multi-Domain**: Train on multiple text types with domain-specific growth
2. **Integration**: Connect with growing neural language models
3. **Pruning**: Remove underutilized tokens to maintain efficiency
4. **Transfer**: Pre-trained growing tokenizers for new domains
5. **Hierarchical**: Multi-level tokenization with different granularities

## References

- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. ACL.
- Gage, P. (1994). A New Algorithm for Data Compression. C Users Journal.
- Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer. EMNLP.