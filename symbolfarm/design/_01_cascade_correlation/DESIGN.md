# Cascade-Correlation Transformer: Technical Design

## Architecture Overview

### Core Concept

This implementation adapts cascade-correlation principles to transformer architectures by treating entire transformer decoder blocks as the "units" that are dynamically added during training. This differs from traditional cascade-correlation which adds individual neurons.

### Network Structure

```
Input Tokens
     ↓
Token Embedding + Positional Encoding
     ↓
[Transformer Block 1] ← Added at initialization
     ↓
[Transformer Block 2] ← Added when Block 1 plateaus
     ↓
[Transformer Block N] ← Added dynamically
     ↓
Layer Normalization
     ↓
Output Projection (vocab_size)
     ↓
Softmax (for generation)
```

### Transformer Block Details

Each dynamically added block consists of:
1. **Multi-Head Self-Attention**
   - Causal masking for autoregressive generation
   - Scaled dot-product attention
   - Residual connection + Layer norm

2. **Feed-Forward Network (MLP)**
   - Two linear layers with ReLU/GELU activation
   - Expansion ratio of 4x (configurable)
   - Residual connection + Layer norm

## Cascade-Correlation Adaptation

### Traditional vs. Transformer Cascade-Correlation

| Aspect | Traditional CC | Transformer CC |
|--------|----------------|----------------|
| Unit | Single neuron | Transformer block |
| Input | All previous units | Sequential block output |
| Training | Freeze previous weights | Freeze previous blocks |
| Growth | Add neuron when plateau | Add block when plateau |

### Growth Strategy

1. **Initialization Phase**
   - Start with token embeddings, positional encoding, and output projection
   - No transformer blocks initially (or start with 1 minimal block)

2. **Training Phase**
   - Train current network until performance plateaus
   - Monitor validation loss for `patience` epochs
   - If improvement < `growth_threshold`, trigger growth

3. **Growth Phase**
   - Freeze all existing transformer blocks
   - Add new transformer block at the end of the sequence
   - Initialize new block weights (Xavier/Kaiming initialization)
   - Resume training with new block trainable

4. **Termination**
   - Stop when max_blocks reached OR
   - No improvement after adding block OR
   - Parameter budget (100M) exceeded

## Implementation Details

### Model Architecture (`model.py`)

```python
class CascadeTransformer(nn.Module):
    def __init__(self, config):
        # Core components always present
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList()  # Dynamically grown
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def add_block(self):
        # Add new transformer block
        new_block = TransformerBlock(self.config)
        self.blocks.append(new_block)
        
    def freeze_blocks(self, except_last=True):
        # Freeze all blocks except the newest one
        for i, block in enumerate(self.blocks):
            if except_last and i == len(self.blocks) - 1:
                continue
            for param in block.parameters():
                param.requires_grad = False
```

### Training Loop (`train.py`)

```python
class CascadeTrainer:
    def __init__(self, model, config):
        self.model = model
        self.growth_monitor = GrowthMonitor(config.patience, config.growth_threshold)
        
    def train_epoch(self):
        # Standard training epoch
        for batch in dataloader:
            loss = self.model(batch)
            loss.backward()
            optimizer.step()
            
    def should_grow(self, val_loss):
        return self.growth_monitor.check_plateau(val_loss)
        
    def grow_network(self):
        # Add new block and adjust optimizer
        self.model.add_block()
        self.model.freeze_blocks(except_last=True)
        self.optimizer = self.create_optimizer()  # Only train new block
```

### Configuration (`config.py`)

```python
@dataclass
class CascadeConfig:
    # Model architecture
    vocab_size: int = 50257  # GPT-2 tokenizer
    d_model: int = 256       # Small model dimension
    n_heads: int = 4         # Few attention heads
    d_ff: int = 1024         # FFN expansion (4x d_model)
    max_len: int = 512       # Maximum sequence length
    
    # Cascade-correlation specific
    max_blocks: int = 12     # Maximum transformer blocks
    initial_blocks: int = 1  # Start with 1 block
    patience: int = 5        # Epochs to wait before growth
    growth_threshold: float = 0.01  # Min improvement to continue
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 100
    warmup_steps: int = 1000
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

## Parameter Budget Management

### Estimation

For each transformer block:
- Attention: `4 * d_model^2 + 4 * d_model` parameters
- FFN: `2 * d_model * d_ff + d_model + d_ff` parameters
- Layer norms: `4 * d_model` parameters

With `d_model=256, d_ff=1024`:
- Per block: ~1.3M parameters
- 12 blocks: ~15.6M parameters
- Embeddings + output: ~25.6M parameters
- **Total: ~41.2M parameters** (well under 100M budget)

### Scaling Strategy

If approaching 100M limit:
- Reduce `d_model` for new blocks
- Use fewer attention heads
- Implement progressive shrinking of new blocks

## Data Pipeline (`data.py`)

### TinyStories Processing

1. **Download**: Use HuggingFace datasets library
2. **Tokenization**: GPT-2 tokenizer (BPE)
3. **Chunking**: Split into fixed-length sequences (512 tokens)
4. **Batching**: Dynamic batching with padding
5. **Caching**: Cache processed data for faster loading

```python
def create_dataloader(config):
    dataset = load_dataset("roneneldan/TinyStories")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Process and chunk
    processed = dataset.map(
        lambda x: tokenize_and_chunk(x, tokenizer, config.max_len),
        batched=True
    )
    
    return DataLoader(processed, batch_size=config.batch_size)
```

## Training Monitoring and Visualization

### Metrics Tracked

1. **Training Loss**: Per epoch and per block addition
2. **Validation Loss**: Plateau detection for growth decisions
3. **Parameter Count**: Monitor approach to 100M limit
4. **Block Addition Timeline**: When each block was added
5. **Generation Quality**: Perplexity and sample outputs

### Visualization

1. **Loss Curves**: Show loss before/after each block addition
2. **Network Growth**: Timeline of block additions
3. **Parameter Evolution**: Parameter count over time
4. **Attention Patterns**: Visualize attention weights per block

## Experimental Hypotheses

### Primary Hypothesis
Dynamic network construction will lead to more efficient parameter usage compared to training a fixed-size transformer of equivalent capacity.

### Secondary Hypotheses
1. Earlier blocks will learn basic language patterns
2. Later blocks will capture more complex dependencies
3. Growth timing will correlate with linguistic complexity milestones

### Evaluation Metrics
- **Efficiency**: Perplexity per parameter
- **Sample Quality**: Human evaluation of generated stories
- **Learning Dynamics**: Convergence speed vs. fixed architecture
- **Interpretability**: Block specialization analysis

## Potential Challenges and Solutions

### Challenge 1: Gradient Flow
**Problem**: Deep networks may suffer from vanishing gradients
**Solution**: Use proper initialization, residual connections, and gradient clipping

### Challenge 2: Overfitting New Blocks
**Problem**: New blocks might overfit quickly
**Solution**: L2 regularization on new blocks only, dropout

### Challenge 3: Growth Decision
**Problem**: When exactly to add new blocks
**Solution**: Multiple criteria (plateau detection + validation metrics)

### Challenge 4: Memory Efficiency
**Problem**: Growing network consumes more GPU memory
**Solution**: Gradient checkpointing, mixed precision training

## Future Extensions

1. **Block Specialization**: Add different types of blocks (attention-only, FFN-only)
2. **Pruning**: Remove underutilized blocks
3. **Multi-Task**: Train on multiple tasks and add task-specific blocks
4. **Transfer Learning**: Pre-train base blocks, fine-tune with growth