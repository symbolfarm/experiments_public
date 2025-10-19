# DESIGN.md: Residual Transformer with Fixed Token Embeddings

## 1. Overview and Motivation

This document outlines the design for a novel Transformer architecture that achieves O(n) complexity while maintaining the expressiveness needed for language modeling. The key innovation is the separation of token representation and information processing through **fixed token embeddings** and an evolving **residual state**.

The core problem with vanilla Transformers is O(n²) complexity due to full token-to-token attention. Our architecture solves this by:
1. Keeping token embeddings **fixed** throughout processing
2. Using a **fixed-length residual state** as working memory
3. Having different attention patterns for processing vs readout

## 2. Revolutionary Architectural Concept

### Key Innovation: Fixed Token Embeddings
Unlike standard transformers where token representations evolve through each block, our model keeps the original token embeddings **completely unchanged** during processing. All dynamic computation happens in a separate residual state.

### Two-Phase Processing
1. **Processing Phase** (Blocks 1 to N-1): Token embeddings query residual state, only residual state evolves
2. **Readout Phase** (Final Block): Residual state queries token embeddings to generate final predictions

This creates a clean separation:
- **Token embeddings**: Static positional context
- **Residual state**: Dynamic working memory
- **Final readout**: Integration for prediction

### Complexity Analysis
- **Processing blocks**: O(n · m · d_model) where m is fixed residual length
- **Readout block**: O(n · m · d_model) 
- **Total complexity**: O(n) since m is a hyperparameter (m << n for long sequences)

This residual state acts as a **learned compression** of sequence information.

## 3. Detailed Component Breakdown

### 3.1. Fixed Token Embeddings

Token embeddings are computed **once** at the beginning and **never modified** during processing:

- **Token Embedding**: Standard `nn.Embedding(vocab_size, d_model)`
- **Positional Encoding**: Sinusoidal encoding added to token embeddings
- **Immutability**: These embeddings remain constant through all processing blocks
- **Shape**: `(batch_size, seq_len, d_model)` - fixed throughout forward pass

### 3.2. Residual State Initialization

The residual state acts as working memory and must be initialized before processing:

- **Shape**: `(batch_size, fixed_kv_length, d_model)`
- **Initialization Strategies**:
  - `"random"`: Random Gaussian initialization
  - `"learned"`: Learnable `nn.Parameter` expanded to batch size
  - `"pooled"`: Adaptive pooling from token embeddings + projection

### 3.3. Processing Blocks (Phase 1)

Each processing block follows this pattern:

#### 3.3.1. Processing Attention
- **Query**: Token embeddings (fixed) → `Q = TokenEmbeddings @ W_q_i`
- **Key**: Residual state → `K = ResidualState @ W_k_i`  
- **Value**: Residual state → `V = ResidualState @ W_v_i`
- **Output**: Attention result used ONLY for residual update (not to modify tokens)

#### 3.3.2. Residual State Update
Cross-attention to evolve the working memory:
- **Query**: Current residual state → `Res_Q = ResidualState @ W_res_q`
- **Key**: Attention output → `Res_K = AttentionOutput @ W_res_k`
- **Value**: Attention output → `Res_V = AttentionOutput @ W_res_v`
- **Update**: `NewResidualState = LayerNorm(ResidualState + CrossAttention(Res_Q, Res_K, Res_V))`

**Critical**: Token embeddings are **never modified** in processing blocks.

### 3.4. Readout Block (Phase 2)

The final block reverses the attention pattern to generate outputs:

#### 3.4.1. Readout Attention
- **Query**: Final residual state → `Q = FinalResidualState @ W_q_final`
- **Key**: Token embeddings (fixed) → `K = TokenEmbeddings @ W_k_final`
- **Value**: Token embeddings (fixed) → `V = TokenEmbeddings @ W_v_final`
- **Output**: Updates token embeddings for the **first and only time**

#### 3.4.2. Feed-Forward Network
- **Input**: Updated token embeddings from readout attention
- **FFN**: Standard feed-forward network 
- **Output**: Final token representations for language modeling

#### 3.4.3. Output Projection
- **Final Layer Norm**: Applied to FFN output
- **Vocabulary Projection**: `FinalTokens @ W_out → logits`
- **Shape**: `(batch_size, seq_len, vocab_size)`

## 4. PyTorch Implementation Guidance

### `ResidualAttention` Module

```python
import torch
import torch.nn as nn
import math

class ResidualAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads

        # Projections for Q from input, and K, V from residual
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x_input, residual_state):
        # x_input shape: (batch_size, seq_len, d_model)
        # residual_state shape: (batch_size, fixed_kv_len, d_model)
        batch_size, seq_len, _ = x_input.shape
        _, fixed_kv_len, _ = residual_state.shape

        # 1. Project Q, K, V
        q = self.w_q(x_input)
        k = self.w_k(residual_state)
        v = self.w_v(residual_state)

        # 2. Reshape for multi-head attention
        # (batch, len, heads, d_head) -> (batch, heads, len, d_head)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, fixed_kv_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, fixed_kv_len, self.n_heads, self.d_head).transpose(1, 2)

        # 3. Compute attention scores
        # (batch, heads, seq_len, d_head) @ (batch, heads, d_head, fixed_kv_len) -> (batch, heads, seq_len, fixed_kv_len)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attention_weights = torch.softmax(scores, dim=-1)

        # 4. Apply attention to values
        # (batch, heads, seq_len, fixed_kv_len) @ (batch, heads, fixed_kv_len, d_head) -> (batch, heads, seq_len, d_head)
        attention_output = attention_weights @ v

        # 5. Concatenate heads and apply final linear layer
        # (batch, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attention_output)
```

### `TransformerBlock` Module

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attention = ResidualAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # This part is for updating the residual state
        self.residual_updater = ResidualAttention(d_model, n_heads) # Could be a different, smaller attention
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual_state):
        # x shape: (batch, seq_len, d_model)
        # residual_state shape: (batch, fixed_kv_len, d_model)

        # 1. Main attention mechanism
        attn_out = self.attention(x, residual_state)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Feed-Forward Network
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        # 3. Update the residual state
        # Here, the residual queries the output of the FFN
        # Other update mechanisms are possible and should be explored.
        # Note the inverted argument order: residual_state is the query source, x is the key/value source.
        residual_update = self.residual_updater(x_input=residual_state, residual_state=x)
        new_residual_state = self.norm3(residual_state + self.dropout(residual_update))

        return x, new_residual_state
```

## 6. Critical Hyperparameters

- **`fixed_kv_length`**: Size of residual state - **THE** key hyperparameter (16, 32, 64, 128, 256)
- **`n_processing_blocks`**: Number of processing blocks before readout (4, 6, 8, 12)
- **`residual_init_strategy`**: How to initialize working memory ("random", "pooled", "learned")
- **`d_model`**: Model dimension (64, 128, 256, 512)
- **`n_heads`**: Number of attention heads (4, 8, 12)
- **`d_ff`**: FFN dimension (typically 4x d_model)

## 7. Expected Benefits vs Challenges

### Benefits
1. **Linear Complexity**: O(n) vs O(n²) for long sequences
2. **Memory Efficiency**: Fixed working memory regardless of sequence length
3. **Information Bottleneck**: Forces efficient representation learning
4. **Conceptual Clarity**: Clean separation of roles (storage vs processing)

### Challenges  
1. **Information Loss**: Fixed bottleneck may lose important details
2. **Training Difficulty**: Complex information flow patterns
3. **Readout Complexity**: Mapping from fixed_kv_length back to seq_len
4. **Hyperparameter Sensitivity**: Critical dependence on fixed_kv_length

## 8. Experimental Priorities

### Phase 1: Proof of Concept
1. **Basic Learning**: Can the model learn anything on TinyStories?
2. **Baseline Comparison**: How does it compare to standard transformer?
3. **Parameter Efficiency**: Better performance per parameter?

### Phase 2: Hyperparameter Exploration
1. **Residual Size Sweep**: Impact of fixed_kv_length (16, 32, 64, 128)
2. **Architecture Depth**: Optimal n_processing_blocks
3. **Initialization Strategy**: Which residual init works best?

### Phase 3: Scaling Analysis  
1. **Sequence Length**: Benefits for longer sequences (512, 1024, 2048)
2. **Memory Profiling**: Actual memory usage vs sequence length
3. **Speed Benchmarking**: Real-world inference speed improvements

### Phase 4: Advanced Analysis
1. **Residual State Analysis**: What information does it capture?
2. **Attention Pattern Visualization**: How does information flow?
3. **Contextual Memory**: Can residual state adapt to different contexts?

## 9. Success Metrics

- **Learning Capability**: Can achieve reasonable perplexity on language modeling
- **Efficiency**: Memory/compute advantages for sequences > 512 tokens  
- **Scalability**: Performance maintains or improves with longer sequences
- **Information Preservation**: Maintains important long-range dependencies