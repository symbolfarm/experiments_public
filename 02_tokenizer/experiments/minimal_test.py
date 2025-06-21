"""
Minimal Test Experiment Configuration

This configuration is designed for quick testing and development. Small dataset,
fast execution, basic functionality validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import GrowingBPEConfig

# Minimal test configuration
config = GrowingBPEConfig(
    # Basic settings
    run_name="minimal_test_experiment",
    
    # Initial vocabulary - very small for testing
    initial_vocab_strategy="character_only",
    initial_vocab_size=50,
    
    # Simple growth trigger
    growth_triggers=["frequency"],
    frequency_threshold=3,
    ngram_range=(2, 4),
    
    # Simple growth mechanism
    growth_mechanism="standard_bpe",
    max_candidates_per_growth=3,
    min_pattern_frequency=2,
    max_pattern_length=6,
    
    # Fast growth for testing
    growth_patience=50,
    min_growth_interval=25,
    max_growth_events=10,      # Limit for quick testing
    growth_cooldown=10,
    
    # Small vocabulary for testing
    max_vocab_size=200,
    
    # Minimal training parameters
    max_tokens_to_process=5000,   # Very small for quick testing
    batch_size=100,
    max_samples=50,               # Limit dataset samples
    
    # Frequent logging for testing
    log_interval=50,
    checkpoint_interval=1000,
    validation_interval=500,
    
    # Simple preprocessing
    lowercase=True,
    handle_punctuation="keep",
    
    # Basic evaluation
    evaluation_window=500,
    track_token_usage=True,
    track_compression_ratio=True,
    track_vocabulary_utilization=False,  # Skip expensive metrics
    track_growth_events=True,
    compute_linguistic_metrics=False,
    
    # Minimal performance settings
    cache_tokenization=False,    # Disable for simplicity
    use_multiprocessing=False,
    
    # No advanced features
    enable_token_pruning=False,
    enable_hierarchical_growth=False,
    enable_domain_adaptation=False,
    
    # Basic visualization
    plot_growth=True,
    save_vocabulary_evolution=False,
    save_attention_maps=False,
    
    # Development settings
    log_level="DEBUG",
    deterministic=True,
    random_seed=42
)

if __name__ == "__main__":
    print("=== Minimal Test Experiment Configuration ===")
    config.print_config()
    
    print(f"\nMemory estimate: {config.estimate_memory_usage()['total'] / 1024:.1f} KB")
    print(f"Expected runtime: < 1 minute")
    print(f"Target vocabulary size: {config.max_vocab_size}")
    print("Focus: Quick validation of core functionality")