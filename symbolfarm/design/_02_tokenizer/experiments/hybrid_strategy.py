"""
Hybrid Strategy Experiment Configuration

This configuration balances aggressive and conservative approaches, using multiple
triggers with moderate thresholds and adaptive mechanisms.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import GrowingBPEConfig

# Hybrid strategy configuration
config = GrowingBPEConfig(
    # Basic settings
    run_name="hybrid_strategy_experiment",
    
    # Initial vocabulary - balanced start
    initial_vocab_strategy="frequency_based",
    initial_vocab_size=500,  # Medium starting vocabulary
    
    # Multiple triggers with moderate thresholds
    growth_triggers=["compression", "frequency"],
    compression_threshold=0.65,   # Moderate sensitivity
    frequency_threshold=8,        # Moderate frequency requirement
    compression_window_size=1500,
    ngram_range=(2, 5),
    
    # Adaptive growth mechanism
    growth_mechanism="adaptive_merging",  # Smart merging strategy
    max_candidates_per_growth=10,         # Moderate growth rate
    min_pattern_frequency=5,              # Balanced requirement
    max_pattern_length=8,                 # Moderate pattern length
    candidate_selection_strategy="frequency",
    
    # Balanced growth timing
    growth_patience=1000,        # Moderate patience
    min_growth_interval=500,     # Moderate intervals
    max_growth_events=75,        # Moderate number of events
    growth_cooldown=200,         # Moderate cooldown
    
    # Moderate vocabulary budget
    max_vocab_size=4000,
    
    # Training parameters
    max_tokens_to_process=750000,  # Larger dataset
    batch_size=750,                # Medium batch size
    
    # Moderate logging frequency
    log_interval=1000,
    checkpoint_interval=10000,
    validation_interval=5000,
    
    # Text preprocessing - balanced
    lowercase=True,
    handle_punctuation="separate",
    normalize_unicode=True,
    
    # Evaluation settings
    evaluation_window=3000,
    track_token_usage=True,
    track_compression_ratio=True,
    track_vocabulary_utilization=True,
    track_growth_events=True,
    compute_linguistic_metrics=True,  # Enable detailed metrics
    
    # Performance settings
    cache_tokenization=True,
    cache_size=7500,
    
    # Balanced features
    enable_token_pruning=True,
    pruning_threshold=5,           # Moderate pruning threshold
    enable_domain_adaptation=False,
    
    # Advanced evaluation
    compatible_with_transformers=True,
    save_tokenizer_json=True,
    
    # Visualization
    plot_growth=True,
    save_vocabulary_evolution=True
)

if __name__ == "__main__":
    print("=== Hybrid Strategy Experiment Configuration ===")
    config.print_config()
    
    print(f"\nMemory estimate: {config.estimate_memory_usage()['total'] / 1024 / 1024:.1f} MB")
    print(f"Expected growth events: Moderate frequency, adaptive quality")
    print(f"Target vocabulary size: {config.max_vocab_size}")
    print("Focus: Balanced exploration and exploitation")