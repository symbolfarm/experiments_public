from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class CascadeConfig:
    # Model architecture
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    d_model: int = 64       # Model dimension per block (smaller for more blocks)
    n_heads: int = 4         # Number of attention heads per block
    d_ff: int = 256          # FFN expansion (4x d_model)
    max_len: int = 256       # Maximum sequence length (reduced for better memory usage)
    dropout: float = 0.1     # Dropout probability
    
    # Cascade-correlation specific
    max_blocks: int = 20     # Maximum transformer blocks
    initial_blocks: int = 1  # Start with 1 block
    patience: int = 4        # Epochs to wait before growth (more aggressive)
    growth_threshold: float = 0.5  # Min improvement to continue
    freeze_previous: bool = True     # Freeze previous blocks when growing
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 3e-4  # Higher LR for faster training with smaller model
    weight_decay: float = 0.01
    max_epochs: int = 50         # Fewer epochs, focus on cascade growth
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    
    # Data
    dataset_name: str = "roneneldan/TinyStories"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: Optional[str] = None  # TinyStories doesn't have test split
    
    # Optimization
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # linear, cosine, constant
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Regularization
    label_smoothing: float = 0.0
    new_block_l2: float = 0.01  # L2 regularization for new blocks only
    
    # Hardware and performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # Use automatic mixed precision
    compile_model: bool = True    # Use torch.compile for speed
    num_workers: int = 0          # DataLoader workers (set to 0 to avoid multiprocessing issues)
    pin_memory: bool = True       # Pin memory for faster GPU transfer
    
    # Logging and evaluation
    log_interval: int = 100       # Log every N steps
    eval_interval: int = 1000     # Evaluate every N steps  
    save_interval: int = 5000     # Save checkpoint every N steps
    generate_samples: int = 5     # Number of samples to generate for evaluation
    max_generate_length: int = 100  # Max length for sample generation
    
    # Visualization and monitoring
    plot_growth: bool = True      # Plot network growth over time
    save_attention_maps: bool = False  # Save attention visualizations
    track_block_gradients: bool = True  # Monitor gradient flow per block
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    run_name: Optional[str] = None  # Unique run identifier
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 20  # Epochs without improvement
    early_stopping_min_delta: float = 0.001  # Minimum change to qualify as improvement
    
    # Parameter budget management
    max_parameters: int = 100_000_000  # 100M parameter budget
    warn_at_parameters: int = 80_000_000   # Warn when approaching limit
    
    def __post_init__(self):
        """Validate configuration and set derived values."""
        # Validate basic constraints
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.initial_blocks >= 1, "Must start with at least 1 block"
        assert self.max_blocks >= self.initial_blocks, "max_blocks must be >= initial_blocks"
        assert self.patience > 0, "Patience must be positive"
        assert 0 <= self.dropout <= 1, "Dropout must be between 0 and 1"
        
        # Set run name if not provided
        if self.run_name is None:
            import datetime
            self.run_name = f"cascade_transformer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Adjust batch size for available GPU memory (rough heuristic)
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb < 12:  # Adjust for smaller GPUs
                self.batch_size = min(self.batch_size, 16)
                print(f"Adjusted batch_size to {self.batch_size} for GPU with {gpu_memory_gb:.1f}GB memory")
    
    def get_parameter_estimate(self) -> int:
        """Estimate total parameters for max configuration."""
        # Embedding parameters
        embedding_params = self.vocab_size * self.d_model  # Token embedding
        pos_embedding_params = self.max_len * self.d_model  # Positional embedding
        
        # Per-block parameters
        # Attention: Q, K, V, O projections + layer norm
        attn_params = 4 * self.d_model * self.d_model + self.d_model
        # FFN: two linear layers + layer norm  
        ffn_params = self.d_model * self.d_ff + self.d_ff * self.d_model + self.d_model
        # Layer norms
        ln_params = 2 * self.d_model
        
        block_params = attn_params + ffn_params + ln_params
        
        # Output layer
        output_params = self.d_model * self.vocab_size
        
        # Final layer norm
        final_ln_params = self.d_model
        
        total_params = (embedding_params + pos_embedding_params + 
                       self.max_blocks * block_params + 
                       output_params + final_ln_params)
        
        return total_params
    
    def print_config(self):
        """Print configuration summary."""
        print("=== Cascade-Correlation Transformer Configuration ===")
        print(f"Model: d_model={self.d_model}, n_heads={self.n_heads}, d_ff={self.d_ff}")
        print(f"Cascade: {self.initial_blocks} -> {self.max_blocks} blocks, patience={self.patience}")
        print(f"Training: lr={self.learning_rate}, batch_size={self.batch_size}, max_epochs={self.max_epochs}")
        print(f"Data: {self.dataset_name}, max_len={self.max_len}")
        print(f"Device: {self.device}, mixed_precision={self.mixed_precision}")
        print(f"Estimated max parameters: {self.get_parameter_estimate():,}")
        print("=" * 55)

# Default configuration instance
default_config = CascadeConfig()

if __name__ == "__main__":
    # Test configuration
    config = CascadeConfig()
    config.print_config()
    
    # Test parameter estimation
    params = config.get_parameter_estimate()
    print(f"\nParameter budget check:")
    print(f"Estimated parameters: {params:,}")
    print(f"Budget limit: {config.max_parameters:,}")
    print(f"Within budget: {params <= config.max_parameters}")