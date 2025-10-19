import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

from config import CascadeConfig

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Register causal mask
        self.register_buffer('causal_mask', None)
    
    def _get_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create or retrieve causal mask."""
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            # Create lower triangular mask
            mask = torch.tril(torch.ones(seq_len, seq_len))
            mask = mask.masked_fill(mask == 0, float('-inf'))
            mask = mask.masked_fill(mask == 1, 0.0)
            self.register_buffer('causal_mask', mask)
        
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply causal mask
        causal_mask = self._get_causal_mask(seq_len).to(scores.device)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        
        return out

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Use GELU like modern transformers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single transformer decoder block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block with residual connections."""
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x

class CascadeTransformer(nn.Module):
    """Cascade-Correlation Transformer that grows dynamically."""
    
    def __init__(self, config: CascadeConfig):
        super().__init__()
        self.config = config
        
        # Core components (always present)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_len)
        self.dropout = nn.Dropout(config.dropout)
        
        # Dynamically grown transformer blocks
        self.blocks = nn.ModuleList()
        
        # Output components
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Track growth history (initialize before add_block calls)
        self.growth_history = []
        
        # Initialize with specified number of blocks
        for _ in range(config.initial_blocks):
            self.add_block()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using Xavier/Kaiming initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def add_block(self) -> bool:
        """Add a new transformer block to the network."""
        if len(self.blocks) >= self.config.max_blocks:
            return False
        
        # Create new block
        new_block = TransformerBlock(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            dropout=self.config.dropout
        )
        
        # Initialize weights for new block
        new_block.apply(self._init_weights)
        
        # Move new block to the same device as the model
        device = next(self.parameters()).device
        new_block = new_block.to(device)
        
        # Add to module list
        self.blocks.append(new_block)
        
        # Record growth
        block_num = len(self.blocks)
        self.growth_history.append({
            'block_num': block_num,
            'total_params': self.count_parameters()
        })
        
        print(f"Added block {block_num}, total parameters: {self.count_parameters():,}")
        return True
    
    def freeze_blocks(self, except_last: bool = True):
        """Freeze parameters of existing blocks."""
        if not self.config.freeze_previous:
            return
        
        for i, block in enumerate(self.blocks):
            # Skip the last block if except_last is True
            if except_last and i == len(self.blocks) - 1:
                continue
            
            # Freeze all parameters in this block
            for param in block.parameters():
                param.requires_grad = False
    
    def unfreeze_all_blocks(self):
        """Unfreeze all transformer blocks."""
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = True
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count total parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def get_block_parameters(self) -> List[int]:
        """Get parameter count for each block."""
        return [sum(p.numel() for p in block.parameters()) for block in self.blocks]
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the cascade transformer.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask (not used in this implementation)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Embedding and positional encoding
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and output projection
        x = self.layer_norm(x)
        logits = self.output_proj(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs of shape (batch_size, seq_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Generated token IDs of shape (batch_size, max_length)
        """
        self.eval()
        batch_size, start_len = input_ids.shape
        
        # Pad input to max_length if needed
        if start_len >= max_length:
            return input_ids[:, :max_length]
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - start_len):
                # Get logits for current sequence
                logits = self.forward(generated)
                
                # Get logits for last position
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we exceed max sequence length
                if generated.size(1) >= self.config.max_len:
                    break
        
        return generated
    
    def get_info(self) -> dict:
        """Get model information."""
        return {
            'num_blocks': len(self.blocks),
            'total_parameters': self.count_parameters(),
            'trainable_parameters': self.count_parameters(trainable_only=True),
            'growth_history': self.growth_history,
            'block_parameters': self.get_block_parameters(),
            'config': self.config
        }

# Loss function for language modeling
class LanguageModelingLoss(nn.Module):
    """Cross-entropy loss for language modeling with optional label smoothing."""
    
    def __init__(self, vocab_size: int, label_smoothing: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute language modeling loss.
        
        Args:
            logits: Model predictions of shape (batch_size, seq_len, vocab_size)
            targets: Target token IDs of shape (batch_size, seq_len)
            
        Returns:
            Loss tensor
        """
        # Shift targets for autoregressive prediction
        # Predict next token, so targets are shifted by 1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()
        
        # Flatten for cross-entropy computation
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_targets = shift_targets.view(-1)
        
        return self.criterion(shift_logits, shift_targets)

if __name__ == "__main__":
    # Test the model
    from config import CascadeConfig
    
    config = CascadeConfig()
    config.vocab_size = 1000  # Smaller vocab for testing
    config.max_len = 64
    config.d_model = 128
    config.n_heads = 4
    config.d_ff = 512
    config.initial_blocks = 2
    
    # Create model
    model = CascadeTransformer(config)
    print(f"Initial model: {model.count_parameters():,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test adding blocks
    model.add_block()
    print(f"After adding block: {model.count_parameters():,} parameters")
    
    # Test loss
    loss_fn = LanguageModelingLoss(config.vocab_size)
    loss = loss_fn(logits, input_ids)
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    generated = model.generate(input_ids[:1, :10], max_length=20)
    print(f"Generated shape: {generated.shape}")
    
    print("\nModel info:")
    info = model.get_info()
    for key, value in info.items():
        if key != 'config':
            print(f"{key}: {value}")