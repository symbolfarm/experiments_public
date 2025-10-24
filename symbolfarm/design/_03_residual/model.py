import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from config import ResidualTransformerConfig

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

class ProcessingAttention(nn.Module):
    """Processing block attention: fixed tokens query residual state."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Only Q projection for token embeddings
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # K, V projections for residual state
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, token_embeddings: torch.Tensor, residual_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: (batch_size, seq_len, d_model) - fixed throughout
            residual_state: (batch_size, fixed_kv_length, d_model)
        Returns:
            attention_output: (batch_size, seq_len, d_model) - for residual update only
        """
        batch_size, seq_len, _ = token_embeddings.shape
        _, fixed_kv_len, _ = residual_state.shape
        
        # Project Q, K, V
        q = self.q_proj(token_embeddings)  # (batch, seq_len, d_model)
        k = self.k_proj(residual_state)    # (batch, fixed_kv_len, d_model)
        v = self.v_proj(residual_state)    # (batch, fixed_kv_len, d_model)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, fixed_kv_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, fixed_kv_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = attention_weights @ v
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        return attention_output

class ResidualUpdater(nn.Module):
    """Cross-attention to update residual state from token representations."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Projections for cross-attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)  # From residual
        self.k_proj = nn.Linear(d_model, d_model, bias=False)  # From tokens
        self.v_proj = nn.Linear(d_model, d_model, bias=False)  # From tokens
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, residual_state: torch.Tensor, token_representations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual_state: (batch_size, fixed_kv_length, d_model)
            token_representations: (batch_size, seq_len, d_model)
        Returns:
            updated_residual: (batch_size, fixed_kv_length, d_model)
        """
        batch_size, fixed_kv_len, _ = residual_state.shape
        _, seq_len, _ = token_representations.shape
        
        # Project Q, K, V
        q = self.q_proj(residual_state)           # (batch, fixed_kv_len, d_model)
        k = self.k_proj(token_representations)    # (batch, seq_len, d_model)
        v = self.v_proj(token_representations)    # (batch, seq_len, d_model)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, fixed_kv_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = attention_weights @ v
        
        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, fixed_kv_len, self.d_model)
        
        return self.out_proj(attention_output)

class ProcessingBlock(nn.Module):
    """Processing block: tokens query residual, then residual gets updated."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = ProcessingAttention(d_model, n_heads, dropout)
        self.residual_updater = ResidualUpdater(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, token_embeddings: torch.Tensor, residual_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: (batch_size, seq_len, d_model) - fixed, not modified
            residual_state: (batch_size, fixed_kv_length, d_model)
        Returns:
            new_residual_state: (batch_size, fixed_kv_length, d_model)
        """
        # Token embeddings query residual state
        token_attention_output = self.attention(token_embeddings, residual_state)
        
        # Use attention output to update residual state
        residual_update = self.residual_updater(residual_state, token_attention_output)
        
        # Apply residual connection and layer norm
        new_residual_state = self.norm(residual_state + self.dropout(residual_update))
        
        return new_residual_state

class ReadoutAttention(nn.Module):
    """Final readout attention: residual queries tokens to update token embeddings."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Q from residual, K,V from tokens
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, residual_state: torch.Tensor, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual_state: (batch_size, fixed_kv_length, d_model)
            token_embeddings: (batch_size, seq_len, d_model)
        Returns:
            updated_tokens: (batch_size, seq_len, d_model)
        """
        batch_size, fixed_kv_len, _ = residual_state.shape
        _, seq_len, _ = token_embeddings.shape
        
        # Project Q, K, V
        q = self.q_proj(residual_state)      # (batch, fixed_kv_len, d_model)
        k = self.k_proj(token_embeddings)    # (batch, seq_len, d_model)
        v = self.v_proj(token_embeddings)    # (batch, seq_len, d_model)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, fixed_kv_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = attention_weights @ v  # (batch, heads, fixed_kv_len, d_k)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, fixed_kv_len, self.d_model)
        
        # Project back to token space - this is the key insight!
        # We need to somehow map from fixed_kv_len back to seq_len
        # For now, use a simple linear interpolation or learned projection
        updated_tokens = self.out_proj(attention_output)
        
        # TODO: This is a mismatch - we have fixed_kv_len but need seq_len
        # For now, let's use a different approach: broadcast or pool
        if fixed_kv_len != seq_len:
            # Simple approach: use another attention to map back
            # For now, let's just repeat the first fixed_kv_len positions
            if seq_len <= fixed_kv_len:
                updated_tokens = updated_tokens[:, :seq_len, :]
            else:
                # Pad with zeros or repeat
                padding = torch.zeros(batch_size, seq_len - fixed_kv_len, self.d_model, 
                                    device=updated_tokens.device, dtype=updated_tokens.dtype)
                updated_tokens = torch.cat([updated_tokens, padding], dim=1)
        
        return updated_tokens

class ReadoutBlock(nn.Module):
    """Final readout block: residual queries tokens, then FFN on updated tokens."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = ReadoutAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, residual_state: torch.Tensor, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual_state: (batch_size, fixed_kv_length, d_model)
            token_embeddings: (batch_size, seq_len, d_model)
        Returns:
            final_tokens: (batch_size, seq_len, d_model)
        """
        # Residual queries tokens to update them
        updated_tokens = self.attention(residual_state, token_embeddings)
        updated_tokens = self.norm1(token_embeddings + self.dropout(updated_tokens))
        
        # Apply FFN
        ffn_output = self.ffn(updated_tokens)
        final_tokens = self.norm2(updated_tokens + self.dropout(ffn_output))
        
        return final_tokens

class ResidualTransformer(nn.Module):
    """Residual Transformer with fixed token embeddings and evolving residual state."""
    
    def __init__(self, config: ResidualTransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (fixed throughout forward pass)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_len)
        
        # Initial residual state
        self.residual_init_strategy = config.residual_init_strategy
        if config.residual_init_strategy == "learned":
            # Learnable parameter
            self.initial_residual = nn.Parameter(
                torch.randn(1, config.fixed_kv_length, config.d_model) * config.residual_init_scale
            )
        elif config.residual_init_strategy == "pooled":
            # Will be computed from token embeddings
            self.residual_projector = nn.Linear(config.d_model, config.d_model)
        else:  # "random"
            # Will be generated randomly each forward pass
            pass
        
        # Processing blocks
        self.processing_blocks = nn.ModuleList([
            ProcessingBlock(config.d_model, config.n_heads, config.dropout)
            for _ in range(config.n_processing_blocks)
        ])
        
        # Final readout block
        self.readout_block = ReadoutBlock(
            config.d_model, config.n_heads, config.d_ff, config.dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        # Layer norm before output
        self.final_norm = nn.LayerNorm(config.d_model)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def _initialize_residual_state(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Initialize residual state based on strategy."""
        batch_size, seq_len, d_model = token_embeddings.shape
        device = token_embeddings.device
        
        if self.residual_init_strategy == "learned":
            # Expand learned parameter to batch size
            return self.initial_residual.expand(batch_size, -1, -1)
        
        elif self.residual_init_strategy == "pooled":
            # Pool token embeddings to fixed length
            if seq_len <= self.config.fixed_kv_length:
                # Pad with zeros
                padding = torch.zeros(
                    batch_size, self.config.fixed_kv_length - seq_len, d_model, 
                    device=device, dtype=token_embeddings.dtype
                )
                pooled = torch.cat([token_embeddings, padding], dim=1)
            else:
                # Use adaptive average pooling
                pooled = F.adaptive_avg_pool1d(
                    token_embeddings.transpose(1, 2), 
                    self.config.fixed_kv_length
                ).transpose(1, 2)
            
            return self.residual_projector(pooled)
        
        else:  # "random"
            # Random initialization
            return torch.randn(
                batch_size, self.config.fixed_kv_length, d_model,
                device=device, dtype=token_embeddings.dtype
            ) * self.config.residual_init_scale
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            targets: (batch_size, seq_len) for training
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: Optional loss if targets provided
        """
        # Get fixed token embeddings (these never change)
        token_embeddings = self.token_embedding(input_ids)
        token_embeddings = self.pos_encoding(token_embeddings)
        
        # Initialize residual state
        residual_state = self._initialize_residual_state(token_embeddings)
        
        # Process through processing blocks
        for block in self.processing_blocks:
            residual_state = block(token_embeddings, residual_state)
        
        # Final readout: residual queries tokens to update them
        final_tokens = self.readout_block(residual_state, token_embeddings)
        
        # Apply final norm and output projection
        final_tokens = self.final_norm(final_tokens)
        logits = self.output_projection(final_tokens)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Shift targets for next-token prediction
            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_targets = targets[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_length: int = 100) -> torch.Tensor:
        """Simple greedy generation."""
        self.eval()
        
        for _ in range(max_length):
            # Forward pass
            logits, _ = self.forward(input_ids)
            
            # Get next token
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to input
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop at max length
            if input_ids.size(1) >= max_length:
                break
        
        return input_ids