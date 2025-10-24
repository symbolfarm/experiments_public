from typing import Optional

import torch
from torch import nn
from torch import functional as F

from symbolfarm.module.model import (
    PositionalEncoding,
    TransformerBlock,
    LanguageModelingLoss
)

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
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
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