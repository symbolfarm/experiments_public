"""
Aggressive Growth Experiment Configuration

This configuration explores rapid vocabulary expansion with multiple growth triggers
and sensitive thresholds. Designed to test the limits of dynamic growth.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import GrowingBPEConfig

# Aggressive growth configuration
config = GrowingBPEConfig(
    # Basic settings
    run_name="aggressive_growth_experiment",
    
    # Initial vocabulary - start very small
    initial_vocab_strategy="character_only",
    initial_vocab_size=64,  # Very minimal start
    
    # Multiple growth triggers with sensitive thresholds
    growth_triggers=["compression", "frequency", "coverage"],
    compression_threshold=0.5,    # More sensitive to poor compression
    frequency_threshold=3,        # Very low frequency requirement
    oov_threshold=0.02,          # Sensitive to out-of-vocabulary
    ngram_range=(2, 6),          # Wider n-gram range
    
    # Aggressive growth mechanism
    growth_mechanism="pattern_mining",  # More exploratory
    max_candidates_per_growth=25,       # Add many tokens at once
    min_pattern_frequency=2,            # Very low requirement
    max_pattern_length=10,              # Allow longer patterns
    
    # Rapid growth timing
    growth_patience=200,           # Quick to grow
    min_growth_interval=100,       # Short intervals between growth
    max_growth_events=150,         # Allow many growth events
    growth_cooldown=50,            # Short cooldown
    
    # Large vocabulary budget
    max_vocab_size=8000,
    
    # Training parameters
    max_tokens_to_process=500000,  # Moderate dataset size
    batch_size=500,                # Smaller batches for more frequent decisions
    
    # More frequent logging and checkpointing
    log_interval=500,
    checkpoint_interval=5000,
    validation_interval=2500,
    
    # Text preprocessing - more aggressive
    lowercase=True,
    handle_punctuation="separate",  # Separate punctuation for more tokens
    
    # Evaluation settings
    evaluation_window=2000,
    track_token_usage=True,
    track_compression_ratio=True,
    track_vocabulary_utilization=True,
    track_growth_events=True,
    
    # Performance settings
    cache_tokenization=True,
    cache_size=5000,  # Smaller cache due to frequent changes
    
    # Experimental features
    enable_token_pruning=False,  # No pruning in aggressive mode
    
    # Visualization
    plot_growth=True,
    save_vocabulary_evolution=True
)

if __name__ == "__main__":
    print("=== Aggressive Growth Experiment Configuration ===")
    config.print_config()
    
    print(f"\nMemory estimate: {config.estimate_memory_usage()['total'] / 1024 / 1024:.1f} MB")
    print(f"Expected growth events: High frequency")
    print(f"Target vocabulary size: {config.max_vocab_size}")
    print("Focus: Rapid exploration of vocabulary space")