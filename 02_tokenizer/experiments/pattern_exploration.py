"""
Pattern Exploration Experiment Configuration

This configuration focuses on discovering interesting subword patterns through
advanced pattern mining and analysis. Emphasizes pattern quality over speed.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import GrowingBPEConfig

# Pattern exploration configuration
config = GrowingBPEConfig(
    # Basic settings
    run_name="pattern_exploration_experiment",
    
    # Initial vocabulary - linguistic foundation
    initial_vocab_strategy="linguistic",
    initial_vocab_size=750,
    
    # Pattern-focused triggers
    growth_triggers=["frequency", "pattern"],
    frequency_threshold=12,
    pattern_threshold=0.03,      # Moderate pattern significance
    ngram_range=(2, 8),          # Wide range for pattern discovery
    
    # Advanced pattern mining
    growth_mechanism="pattern_mining",
    max_candidates_per_growth=15,
    min_pattern_frequency=8,
    max_pattern_length=12,       # Allow longer patterns
    candidate_selection_strategy="mutual_info",  # Information-theoretic selection
    
    # Patient growth for quality patterns
    growth_patience=2000,
    min_growth_interval=1000,
    max_growth_events=60,
    growth_cooldown=300,
    
    # Medium vocabulary for quality
    max_vocab_size=3500,
    
    # Training parameters
    max_tokens_to_process=1000000,  # Large dataset for pattern discovery
    batch_size=800,
    
    # Moderate logging
    log_interval=1500,
    checkpoint_interval=12000,
    validation_interval=6000,
    
    # Text preprocessing for patterns
    lowercase=False,             # Preserve case for pattern analysis
    handle_punctuation="separate",
    normalize_unicode=True,
    
    # Comprehensive evaluation
    evaluation_window=4000,
    track_token_usage=True,
    track_compression_ratio=True,
    track_vocabulary_utilization=True,
    track_growth_events=True,
    compute_linguistic_metrics=True,  # Enable linguistic analysis
    
    # Performance settings
    cache_tokenization=True,
    cache_size=8000,
    
    # Advanced features for pattern analysis
    enable_token_pruning=True,
    pruning_threshold=3,
    enable_domain_adaptation=True,   # Adapt to text patterns
    
    # Enhanced compatibility
    compatible_with_transformers=True,
    save_tokenizer_json=True,
    
    # Detailed visualization
    plot_growth=True,
    save_vocabulary_evolution=True,
    save_attention_maps=True,
    
    # Research settings
    log_level="INFO",
    deterministic=True,
    random_seed=123
)

if __name__ == "__main__":
    print("=== Pattern Exploration Experiment Configuration ===")
    config.print_config()
    
    print(f"\nMemory estimate: {config.estimate_memory_usage()['total'] / 1024 / 1024:.1f} MB")
    print(f"Expected growth events: Moderate frequency, high pattern quality")
    print(f"Target vocabulary size: {config.max_vocab_size}")
    print("Focus: Discovery of linguistically meaningful subword patterns")