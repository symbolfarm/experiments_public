from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import datetime
import os

@dataclass
class GrowingBPEConfig:
    # Initial vocabulary configuration
    initial_vocab_strategy: str = "character_only"  # character_only, minimal_subwords, frequency_based, linguistic
    initial_vocab_size: int = 256
    include_special_tokens: bool = True
    special_tokens: List[str] = field(default_factory=lambda: ['<pad>', '<unk>', '<bos>', '<eos>'])
    
    # Growth trigger configuration
    growth_triggers: List[str] = field(default_factory=lambda: ["compression", "frequency"])
    # Compression-based trigger
    compression_threshold: float = 0.6  # tokens per character - higher means worse compression
    compression_window_size: int = 1000
    # Frequency-based trigger  
    frequency_threshold: int = 10
    ngram_range: tuple = (2, 4)
    # Coverage-based trigger
    oov_threshold: float = 0.1
    # Pattern-based trigger
    pattern_threshold: float = 0.05  # minimum pattern significance
    
    # Growth mechanism configuration
    growth_mechanism: str = "standard_bpe"  # standard_bpe, pattern_mining, adaptive_merging, token_synthesis
    max_candidates_per_growth: int = 10
    min_pattern_frequency: int = 5
    max_pattern_length: int = 8
    candidate_selection_strategy: str = "frequency"  # frequency, mutual_info, entropy
    
    # Growth control
    max_vocab_size: int = 10000
    growth_patience: int = 1000  # tokens to process before considering growth
    min_growth_interval: int = 500  # minimum tokens between growth events
    max_growth_events: int = 100  # maximum number of growth events
    growth_cooldown: int = 100  # tokens to wait after growth before next consideration
    
    # BPE algorithm parameters
    max_merge_iterations: int = 1000  # for initial BPE training
    min_pair_frequency: int = 2  # minimum frequency for BPE merge
    merge_threshold: float = 0.0001  # stop merging when improvement is small
    
    # Text preprocessing
    lowercase: bool = True
    normalize_unicode: bool = True
    handle_punctuation: str = "separate"  # separate, keep, remove
    word_boundary_token: str = "▁"  # SentencePiece-style
    
    # Evaluation configuration
    evaluation_window: int = 10000  # tokens for computing metrics
    track_token_usage: bool = True
    track_compression_ratio: bool = True
    track_vocabulary_utilization: bool = True
    track_growth_events: bool = True
    compute_linguistic_metrics: bool = False  # expensive metrics
    
    # Data configuration
    dataset_name: str = "roneneldan/TinyStories"
    dataset_split: str = "train"
    validation_split: str = "validation" 
    max_sequence_length: int = 512
    batch_size: int = 1000  # for batch processing
    streaming: bool = True  # use streaming for large datasets
    max_samples: Optional[int] = None  # limit samples for testing
    
    # Training configuration
    max_tokens_to_process: int = 1000000  # total tokens to process
    checkpoint_interval: int = 10000  # tokens between checkpoints
    log_interval: int = 1000  # tokens between log messages
    validation_interval: int = 5000  # tokens between validation runs
    
    # Logging and visualization
    save_dir: str = "./checkpoints"
    run_name: Optional[str] = None
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    plot_growth: bool = True
    save_attention_maps: bool = False  # if integrating with transformer
    save_vocabulary_evolution: bool = True
    
    # Performance optimization
    use_multiprocessing: bool = False  # for parallel processing
    num_workers: int = 1
    cache_tokenization: bool = True
    cache_size: int = 10000  # number of cached tokenizations
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    
    # Experimental features
    enable_token_pruning: bool = False  # remove unused tokens
    pruning_threshold: int = 5  # minimum usage to keep token
    enable_hierarchical_growth: bool = False  # multi-level tokenization
    enable_domain_adaptation: bool = False  # domain-specific growth
    
    # Integration settings
    compatible_with_transformers: bool = True  # HuggingFace compatibility
    save_tokenizer_json: bool = True  # save in tokenizers format
    
    def __post_init__(self):
        """Validate configuration and set derived values."""
        # Validate strategy choices
        valid_vocab_strategies = ["character_only", "minimal_subwords", "frequency_based", "linguistic"]
        if self.initial_vocab_strategy not in valid_vocab_strategies:
            raise ValueError(f"initial_vocab_strategy must be one of {valid_vocab_strategies}")
            
        valid_triggers = ["compression", "frequency", "coverage", "pattern"]
        for trigger in self.growth_triggers:
            if trigger not in valid_triggers:
                raise ValueError(f"Growth trigger '{trigger}' not in {valid_triggers}")
                
        valid_mechanisms = ["standard_bpe", "pattern_mining", "adaptive_merging", "token_synthesis"]
        if self.growth_mechanism not in valid_mechanisms:
            raise ValueError(f"growth_mechanism must be one of {valid_mechanisms}")
        
        # Validate thresholds
        assert 0 < self.compression_threshold <= 2.0, "compression_threshold should be between 0 and 2"
        assert self.frequency_threshold > 0, "frequency_threshold must be positive"
        assert 0 <= self.oov_threshold <= 1, "oov_threshold must be between 0 and 1"
        assert self.initial_vocab_size > 0, "initial_vocab_size must be positive"
        assert self.max_vocab_size >= self.initial_vocab_size, "max_vocab_size must be >= initial_vocab_size"
        assert self.growth_patience > 0, "growth_patience must be positive"
        assert self.min_growth_interval > 0, "min_growth_interval must be positive"
        
        # Set run name if not provided
        if self.run_name is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.run_name = f"growing_bpe_{timestamp}"
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        if self.deterministic:
            import random
            import numpy as np
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
    
    def get_save_path(self) -> str:
        """Get the full save path for this run."""
        return os.path.join(self.save_dir, self.run_name)
    
    def estimate_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage for the configuration."""
        # Rough estimates in bytes
        vocab_memory = self.max_vocab_size * 50  # average token length * overhead
        cache_memory = self.cache_size * 100 if self.cache_tokenization else 0
        stats_memory = self.evaluation_window * 10  # for tracking statistics
        
        return {
            'vocabulary': vocab_memory,
            'cache': cache_memory, 
            'statistics': stats_memory,
            'total': vocab_memory + cache_memory + stats_memory
        }
    
    def print_config(self):
        """Print configuration summary."""
        print("=== Growing BPE Tokenizer Configuration ===")
        print(f"Initial vocab: {self.initial_vocab_strategy} ({self.initial_vocab_size} tokens)")
        print(f"Growth: {' + '.join(self.growth_triggers)} triggers, {self.growth_mechanism} mechanism")
        print(f"Limits: max_vocab={self.max_vocab_size}, max_tokens={self.max_tokens_to_process}")
        print(f"Dataset: {self.dataset_name} ({self.dataset_split})")
        print(f"Run: {self.run_name}")
        
        memory = self.estimate_memory_usage()
        print(f"Estimated memory: {memory['total'] / 1024 / 1024:.1f} MB")
        print("=" * 45)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GrowingBPEConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

# Pre-configured experiment settings
class ExperimentConfigs:
    @staticmethod
    def aggressive_growth() -> GrowingBPEConfig:
        """Aggressive growth configuration for rapid vocabulary expansion."""
        return GrowingBPEConfig(
            initial_vocab_strategy="character_only",
            initial_vocab_size=128,
            growth_triggers=["compression", "frequency", "coverage"],
            compression_threshold=0.5,  # More sensitive
            frequency_threshold=5,      # Lower threshold
            oov_threshold=0.05,        # More sensitive to OOV
            growth_patience=500,        # Grow more frequently
            min_growth_interval=200,
            max_candidates_per_growth=20,
            max_vocab_size=5000,
            growth_mechanism="pattern_mining",
            run_name="aggressive_growth"
        )
    
    @staticmethod
    def conservative_growth() -> GrowingBPEConfig:
        """Conservative growth configuration for careful vocabulary expansion."""
        return GrowingBPEConfig(
            initial_vocab_strategy="frequency_based",
            initial_vocab_size=1000,
            growth_triggers=["compression"],
            compression_threshold=0.8,  # Less sensitive
            frequency_threshold=20,     # Higher threshold
            growth_patience=2000,       # Grow less frequently
            min_growth_interval=1000,
            max_candidates_per_growth=5,
            max_vocab_size=2000,
            growth_mechanism="standard_bpe",
            run_name="conservative_growth"
        )
    
    @staticmethod
    def hybrid_strategy() -> GrowingBPEConfig:
        """Balanced approach with multiple strategies."""
        return GrowingBPEConfig(
            initial_vocab_strategy="minimal_subwords",
            initial_vocab_size=500,
            growth_triggers=["compression", "frequency"],
            compression_threshold=0.65,
            frequency_threshold=10,
            growth_patience=1000,
            min_growth_interval=500,
            max_candidates_per_growth=10,
            max_vocab_size=4000,
            growth_mechanism="adaptive_merging",
            run_name="hybrid_strategy"
        )
    
    @staticmethod
    def minimal_experiment() -> GrowingBPEConfig:
        """Minimal configuration for quick testing."""
        return GrowingBPEConfig(
            initial_vocab_strategy="character_only",
            initial_vocab_size=100,
            growth_triggers=["frequency"],
            frequency_threshold=5,
            growth_patience=100,
            max_candidates_per_growth=5,
            max_vocab_size=500,
            max_tokens_to_process=10000,
            log_interval=100,
            checkpoint_interval=1000,
            run_name="minimal_test"
        )

# Default configuration
default_config = GrowingBPEConfig()

if __name__ == "__main__":
    # Test configurations
    print("=== Default Configuration ===")
    default_config.print_config()
    
    print("\n=== Aggressive Growth ===")
    aggressive = ExperimentConfigs.aggressive_growth()
    aggressive.print_config()
    
    print("\n=== Conservative Growth ===") 
    conservative = ExperimentConfigs.conservative_growth()
    conservative.print_config()
    
    print("\n=== Memory Usage Estimates ===")
    for name, config in [
        ("Default", default_config),
        ("Aggressive", aggressive), 
        ("Conservative", conservative)
    ]:
        memory = config.estimate_memory_usage()
        print(f"{name}: {memory['total'] / 1024 / 1024:.1f} MB")