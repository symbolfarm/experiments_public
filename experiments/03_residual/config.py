from dataclasses import dataclass
from typing import Optional, Literal
import torch

@dataclass
class ResidualTransformerConfig:
    # Model architecture
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    d_model: int = 64        # Model dimension (token embedding size)
    n_heads: int = 4         # Number of attention heads
    d_ff: int = 256          # FFN expansion (4x d_model)
    max_len: int = 256       # Maximum sequence length
    dropout: float = 0.1     # Dropout probability
    
    # Residual-specific parameters
    fixed_kv_length: int = 32        # Fixed length of residual state (start small)
    n_processing_blocks: int = 6     # Number of processing blocks (before readout)
    residual_init_strategy: Literal["random", "pooled", "learned"] = "random"
    residual_init_scale: float = 0.02  # Scale for random initialization
    
    # Training mode
    model_type: Literal["residual", "baseline", "both"] = "both"  # Which models to train
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 50
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    
    # Data
    dataset_name: str = "roneneldan/TinyStories"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: Optional[str] = None
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Hardware and performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    compile_model: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    
    # Logging and evaluation
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    generate_samples: int = 5
    max_generate_length: int = 100
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    run_name: Optional[str] = None
    resume_from: Optional[str] = None
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    
    def __post_init__(self):
        """Validate configuration and set derived values."""
        # Validate basic constraints
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.fixed_kv_length > 0, "fixed_kv_length must be positive"
        assert self.n_processing_blocks >= 1, "Must have at least 1 processing block"
        assert 0 <= self.dropout <= 1, "Dropout must be between 0 and 1"
        assert self.residual_init_strategy in ["random", "pooled", "learned"]
        
        # Set run name if not provided
        if self.run_name is None:
            import datetime
            self.run_name = f"residual_transformer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Adjust batch size for available GPU memory
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb < 12:
                self.batch_size = min(self.batch_size, 16)
                print(f"Adjusted batch_size to {self.batch_size} for GPU with {gpu_memory_gb:.1f}GB memory")
    
    def get_residual_parameter_estimate(self) -> int:
        """Estimate parameters for residual model."""
        # Token embedding parameters (fixed throughout)
        embedding_params = self.vocab_size * self.d_model
        pos_embedding_params = self.max_len * self.d_model
        
        # Initial residual state (if learned)
        residual_init_params = self.fixed_kv_length * self.d_model if self.residual_init_strategy == "learned" else 0
        
        # Processing blocks: only Q projection + residual updater + layer norms
        per_processing_block = (
            self.d_model * self.d_model +  # W_q projection
            3 * self.d_model * self.d_model +  # Residual updater (Q, K, V for cross-attention)
            self.d_model * self.d_model +  # Residual updater output projection
            2 * self.d_model  # Layer norms
        )
        processing_params = self.n_processing_blocks * per_processing_block
        
        # Final readout block: attention + FFN + layer norms
        readout_attention = 4 * self.d_model * self.d_model  # Q, K, V, O projections
        readout_ffn = self.d_model * self.d_ff + self.d_ff * self.d_model  # Two linear layers
        readout_ln = 2 * self.d_model  # Layer norms
        readout_params = readout_attention + readout_ffn + readout_ln
        
        # Output projection
        output_params = self.d_model * self.vocab_size
        
        total_params = (embedding_params + pos_embedding_params + residual_init_params + 
                       processing_params + readout_params + output_params)
        
        return total_params
    
    def get_baseline_parameter_estimate(self) -> int:
        """Estimate parameters for standard transformer baseline."""
        # Embedding parameters
        embedding_params = self.vocab_size * self.d_model
        pos_embedding_params = self.max_len * self.d_model
        
        # Per-block parameters (attention + FFN + layer norms)
        attn_params = 4 * self.d_model * self.d_model + self.d_model
        ffn_params = self.d_model * self.d_ff + self.d_ff * self.d_model + self.d_model
        ln_params = 2 * self.d_model
        block_params = attn_params + ffn_params + ln_params
        
        # Use same total number of blocks for fair comparison
        total_blocks = self.n_processing_blocks + 1  # +1 for readout block equivalent
        
        # Output layer + final layer norm
        output_params = self.d_model * self.vocab_size + self.d_model
        
        total_params = (embedding_params + pos_embedding_params + 
                       total_blocks * block_params + output_params)
        
        return total_params
    
    def print_config(self):
        """Print configuration summary."""
        print("=== Residual Transformer Configuration ===")
        print(f"Architecture: d_model={self.d_model}, n_heads={self.n_heads}, d_ff={self.d_ff}")
        print(f"Residual: fixed_kv_length={self.fixed_kv_length}, n_processing_blocks={self.n_processing_blocks}")
        print(f"Residual init: {self.residual_init_strategy} (scale={self.residual_init_scale})")
        print(f"Training: {self.model_type}, lr={self.learning_rate}, batch_size={self.batch_size}")
        print(f"Data: {self.dataset_name}, max_len={self.max_len}")
        print(f"Device: {self.device}, mixed_precision={self.mixed_precision}")
        
        if self.model_type in ["residual", "both"]:
            residual_params = self.get_residual_parameter_estimate()
            print(f"Residual model parameters: {residual_params:,}")
        
        if self.model_type in ["baseline", "both"]:
            baseline_params = self.get_baseline_parameter_estimate()
            print(f"Baseline model parameters: {baseline_params:,}")
        
        print("=" * 45)

# Default configuration instance
default_config = ResidualTransformerConfig()

if __name__ == "__main__":
    # Test configuration
    config = ResidualTransformerConfig()
    config.print_config()
    
    # Test different configurations
    print("\n=== Small Model ===")
    small_config = ResidualTransformerConfig(
        d_model=32, n_heads=2, fixed_kv_length=16, n_processing_blocks=3
    )
    small_config.print_config()
    
    print("\n=== Large Model ===")  
    large_config = ResidualTransformerConfig(
        d_model=128, n_heads=8, fixed_kv_length=64, n_processing_blocks=12
    )
    large_config.print_config()