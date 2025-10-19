"""
Conservative Growth Experiment Configuration

This configuration explores careful, deliberate vocabulary expansion with high
thresholds and longer patience periods. Designed for stable, efficient growth.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import GrowingBPEConfig

# Conservative growth configuration
config = GrowingBPEConfig(
    # Basic settings
    run_name="conservative_growth_experiment",
    
    # Initial vocabulary - start with good foundation
    initial_vocab_strategy="minimal_subwords",
    initial_vocab_size=1000,  # Substantial starting vocabulary
    
    # Single, reliable growth trigger
    growth_triggers=["compression"],
    compression_threshold=0.85,   # Only grow when compression is quite poor
    compression_window_size=2000, # Larger window for stable decisions
    
    # Conservative growth mechanism
    growth_mechanism="standard_bpe",  # Well-tested approach
    max_candidates_per_growth=3,      # Few tokens at a time
    min_pattern_frequency=15,         # High frequency requirement
    max_pattern_length=6,             # Shorter patterns
    candidate_selection_strategy="frequency",
    
    # Patient growth timing
    growth_patience=3000,        # Long patience period
    min_growth_interval=2000,    # Long intervals between growth
    max_growth_events=25,        # Limited growth events
    growth_cooldown=500,         # Long cooldown
    
    # Moderate vocabulary budget
    max_vocab_size=2500,
    
    # Training parameters
    max_tokens_to_process=500000,
    batch_size=1000,             # Larger batches for stability
    
    # Less frequent logging
    log_interval=2000,
    checkpoint_interval=15000,
    validation_interval=10000,
    
    # Text preprocessing - standard
    lowercase=True,
    handle_punctuation="keep",   # Keep punctuation attached
    
    # Evaluation settings
    evaluation_window=5000,      # Larger evaluation window
    track_token_usage=True,
    track_compression_ratio=True,
    track_vocabulary_utilization=True,
    track_growth_events=True,
    
    # Performance settings
    cache_tokenization=True,
    cache_size=10000,            # Larger cache for stability
    
    # Conservative features
    enable_token_pruning=True,   # Enable pruning of unused tokens
    pruning_threshold=10,        # Higher threshold for keeping tokens
    
    # Visualization
    plot_growth=True,
    save_vocabulary_evolution=True
)

if __name__ == "__main__":
    print("=== Conservative Growth Experiment Configuration ===")
    config.print_config()
    
    print(f"\nMemory estimate: {config.estimate_memory_usage()['total'] / 1024 / 1024:.1f} MB")
    print(f"Expected growth events: Low frequency, high quality")
    print(f"Target vocabulary size: {config.max_vocab_size}")
    print("Focus: Stable, efficient vocabulary development")