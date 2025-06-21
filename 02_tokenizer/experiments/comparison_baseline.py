"""
Comparison Baseline Experiment Configuration

This configuration creates a baseline for comparison with other approaches.
Moderate settings across all parameters to serve as a reference point.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import GrowingBPEConfig

# Baseline comparison configuration
config = GrowingBPEConfig(
    # Basic settings
    run_name="comparison_baseline_experiment",
    
    # Standard initial vocabulary
    initial_vocab_strategy="frequency_based",
    initial_vocab_size=512,      # Power of 2 for comparison
    
    # Standard growth triggers
    growth_triggers=["compression", "frequency"],
    compression_threshold=0.7,   # Middle ground
    frequency_threshold=10,      # Standard threshold
    compression_window_size=1000,
    ngram_range=(2, 4),         # Standard range
    
    # Standard BPE mechanism
    growth_mechanism="standard_bpe",
    max_candidates_per_growth=8,
    min_pattern_frequency=6,
    max_pattern_length=6,
    candidate_selection_strategy="frequency",
    
    # Moderate growth timing
    growth_patience=1500,
    min_growth_interval=750,
    max_growth_events=50,
    growth_cooldown=250,
    
    # Standard vocabulary size
    max_vocab_size=3000,
    
    # Standard training parameters
    max_tokens_to_process=600000,
    batch_size=600,
    
    # Standard logging
    log_interval=1200,
    checkpoint_interval=12000,
    validation_interval=6000,
    
    # Standard preprocessing
    lowercase=True,
    handle_punctuation="separate",
    normalize_unicode=True,
    
    # Standard evaluation
    evaluation_window=3000,
    track_token_usage=True,
    track_compression_ratio=True,
    track_vocabulary_utilization=True,
    track_growth_events=True,
    compute_linguistic_metrics=False,
    
    # Standard performance settings
    cache_tokenization=True,
    cache_size=6000,
    
    # Standard features
    enable_token_pruning=False,
    enable_domain_adaptation=False,
    
    # Standard compatibility
    compatible_with_transformers=True,
    save_tokenizer_json=True,
    
    # Standard visualization
    plot_growth=True,
    save_vocabulary_evolution=True,
    
    # Standard settings
    log_level="INFO",
    deterministic=True,
    random_seed=42
)

if __name__ == "__main__":
    print("=== Comparison Baseline Experiment Configuration ===")
    config.print_config()
    
    print(f"\nMemory estimate: {config.estimate_memory_usage()['total'] / 1024 / 1024:.1f} MB")
    print(f"Expected growth events: Moderate frequency and quality")
    print(f"Target vocabulary size: {config.max_vocab_size}")
    print("Focus: Balanced baseline for comparative analysis")