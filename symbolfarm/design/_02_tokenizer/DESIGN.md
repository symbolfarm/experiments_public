# Growing BPE Tokenizer: Technical Design

## Architecture Overview

### Core Concept

This implementation adapts constructive learning principles to tokenization by dynamically growing the BPE vocabulary during operation. Unlike traditional static tokenizers, the vocabulary starts minimal and expands based on the characteristics of text being processed.

### System Components

```
Text Stream
     ↓
Text Preprocessor
     ↓
Growing BPE Tokenizer
     ↓
[Initial Vocab] ← Character-level tokens
     ↓
Growth Monitor ← Tracks metrics and triggers
     ↓
[Extended Vocab] ← Dynamically added subwords
     ↓
Pattern Analyzer ← Identifies growth candidates
     ↓
Tokenized Output
```

## Growing BPE Algorithm

### Traditional BPE vs. Growing BPE

| Aspect | Traditional BPE | Growing BPE |
|--------|-----------------|-------------|
| Training | Offline, fixed corpus | Online, streaming |
| Vocabulary | Static after training | Dynamic, expandable |
| Adaptation | None | Continuous |
| Merge Rules | Fixed set | Continuously extended |

### Growth Process

1. **Initialization Phase**
   - Build minimal initial vocabulary
   - Set up growth monitoring
   - Initialize merge rule tracking

2. **Processing Phase**
   - Tokenize incoming text with current vocabulary
   - Monitor performance metrics
   - Collect statistics for growth decisions

3. **Growth Decision Phase**
   - Evaluate growth triggers
   - Identify candidate tokens
   - Decide whether to expand vocabulary

4. **Growth Execution Phase**
   - Add new tokens to vocabulary
   - Update merge rules
   - Retokenize recent text if beneficial

## Implementation Details

### Core Classes (`tokenizer.py`)

```python
class GrowingBPETokenizer:
    def __init__(self, config):
        self.vocab = {}              # token_id -> token_string
        self.token_to_id = {}        # token_string -> token_id
        self.merge_rules = []        # List of (token1, token2) -> new_token
        self.growth_monitor = GrowthMonitor(config)
        self.pattern_analyzer = PatternAnalyzer(config)
        
    def tokenize(self, text: str) -> List[int]:
        # Apply current BPE rules
        tokens = self._apply_bpe(text)
        
        # Update growth monitoring
        self.growth_monitor.update(text, tokens)
        
        # Check if growth is needed
        if self.growth_monitor.should_grow():
            self._grow_vocabulary(text)
            
        return tokens
        
    def _grow_vocabulary(self, recent_text: str):
        # Find growth candidates
        candidates = self.pattern_analyzer.find_candidates(
            recent_text, self.vocab
        )
        
        # Select best candidates
        new_tokens = self._select_growth_tokens(candidates)
        
        # Add to vocabulary
        for token in new_tokens:
            self._add_token(token)
```

### Growth Strategies

#### 1. Growth Triggers

**Compression-Based Triggers**
```python
class CompressionTrigger:
    def __init__(self, threshold=0.6, window_size=1000):
        self.threshold = threshold
        self.window_size = window_size
        self.recent_ratios = deque(maxlen=window_size)
        
    def should_trigger(self, text: str, tokens: List[int]) -> bool:
        ratio = len(tokens) / len(text)  # tokens per character
        self.recent_ratios.append(ratio)
        
        if len(self.recent_ratios) < self.window_size:
            return False
            
        avg_ratio = sum(self.recent_ratios) / len(self.recent_ratios)
        return avg_ratio > self.threshold  # Poor compression
```

**Frequency-Based Triggers**
```python
class FrequencyTrigger:
    def __init__(self, min_frequency=10, ngram_range=(2, 4)):
        self.min_frequency = min_frequency
        self.ngram_range = ngram_range
        self.char_ngram_counts = defaultdict(int)
        
    def should_trigger(self, text: str, tokens: List[int]) -> bool:
        # Count character n-grams
        for n in range(*self.ngram_range):
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                self.char_ngram_counts[ngram] += 1
                
        # Check if any frequent n-grams aren't well tokenized
        return any(count >= self.min_frequency 
                  for count in self.char_ngram_counts.values())
```

**Coverage-Based Triggers**
```python
class CoverageTrigger:
    def __init__(self, oov_threshold=0.1):
        self.oov_threshold = oov_threshold
        self.oov_chars = set()
        
    def should_trigger(self, text: str, tokens: List[int]) -> bool:
        # Track characters that require multiple tokens
        retokenized = self.detokenize(tokens)
        
        if len(retokenized) != len(text):
            oov_rate = abs(len(retokenized) - len(text)) / len(text)
            return oov_rate > self.oov_threshold
            
        return False
```

#### 2. Growth Mechanisms

**Standard BPE Extension**
```python
class StandardBPEGrowth:
    def find_candidates(self, text: str, vocab: Dict) -> List[str]:
        # Build character pair frequency table
        pair_counts = defaultdict(int)
        words = text.split()
        
        for word in words:
            chars = list(word)
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i+1])
                pair_counts[pair] += 1
                
        # Return most frequent pairs not in vocab
        candidates = []
        for (c1, c2), count in sorted(pair_counts.items(), 
                                     key=lambda x: x[1], reverse=True):
            merged = c1 + c2
            if merged not in vocab and count >= self.min_count:
                candidates.append(merged)
                
        return candidates[:self.max_candidates]
```

**Pattern Mining Growth**
```python
class PatternMiningGrowth:
    def find_candidates(self, text: str, vocab: Dict) -> List[str]:
        candidates = []
        
        # Find repeated substrings
        for length in range(2, self.max_pattern_length):
            patterns = defaultdict(int)
            for i in range(len(text) - length + 1):
                pattern = text[i:i+length]
                patterns[pattern] += 1
                
            for pattern, count in patterns.items():
                if (count >= self.min_frequency and 
                    pattern not in vocab and
                    self._is_meaningful_pattern(pattern)):
                    candidates.append(pattern)
                    
        return candidates
        
    def _is_meaningful_pattern(self, pattern: str) -> bool:
        # Heuristics for meaningful subwords
        if pattern.isspace() or not pattern.strip():
            return False
        if len(set(pattern)) == 1:  # Repeated character
            return False
        if pattern.startswith(' ') and pattern.endswith(' '):
            return False
        return True
```

**Adaptive Merging Growth**
```python
class AdaptiveMergingGrowth:
    def find_candidates(self, text: str, vocab: Dict) -> List[str]:
        # Analyze current tokenization efficiency
        current_tokens = self._tokenize_with_current_vocab(text)
        
        # Find positions where tokenization is inefficient
        inefficient_spans = self._find_inefficient_spans(text, current_tokens)
        
        candidates = []
        for span in inefficient_spans:
            # Try different merge strategies for this span
            merge_candidates = self._generate_merge_candidates(span)
            candidates.extend(merge_candidates)
            
        return candidates
```

#### 3. Initial Vocabulary Strategies

**Character-Only Initialization**
```python
def build_character_vocab(text_sample: str) -> Dict[str, int]:
    chars = set(text_sample)
    vocab = {char: i for i, char in enumerate(sorted(chars))}
    
    # Add special tokens
    vocab['<pad>'] = len(vocab)
    vocab['<unk>'] = len(vocab)
    vocab['<bos>'] = len(vocab)
    vocab['<eos>'] = len(vocab)
    
    return vocab
```

**Minimal Subwords Initialization**
```python
def build_minimal_subword_vocab(text_sample: str) -> Dict[str, int]:
    vocab = build_character_vocab(text_sample)
    
    # Add common English patterns
    common_patterns = [
        'the', 'and', 'ing', 'ed', 'er', 'ly', 
        'un', 're', 'in', 'on', 'at', 'to'
    ]
    
    for pattern in common_patterns:
        if pattern not in vocab:
            vocab[pattern] = len(vocab)
            
    return vocab
```

**Frequency-Based Initialization**
```python
def build_frequency_vocab(text_sample: str, vocab_size: int = 1000) -> Dict[str, int]:
    # Count all n-grams up to length 6
    ngram_counts = defaultdict(int)
    
    for n in range(1, 7):
        for i in range(len(text_sample) - n + 1):
            ngram = text_sample[i:i+n]
            ngram_counts[ngram] += 1
    
    # Select most frequent n-grams
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
    
    vocab = {}
    for ngram, count in sorted_ngrams[:vocab_size]:
        if ngram not in vocab:
            vocab[ngram] = len(vocab)
            
    return vocab
```

## Configuration System (`config.py`)

```python
@dataclass
class GrowingBPEConfig:
    # Initial vocabulary
    initial_vocab_strategy: str = "character_only"  # character_only, minimal_subwords, frequency_based
    initial_vocab_size: int = 256
    
    # Growth triggers
    growth_triggers: List[str] = field(default_factory=lambda: ["compression", "frequency"])
    compression_threshold: float = 0.6
    frequency_threshold: int = 10
    oov_threshold: float = 0.1
    
    # Growth mechanisms  
    growth_mechanism: str = "standard_bpe"  # standard_bpe, pattern_mining, adaptive_merging
    max_candidates_per_growth: int = 10
    min_pattern_frequency: int = 5
    max_pattern_length: int = 8
    
    # Growth control
    max_vocab_size: int = 10000
    growth_patience: int = 1000  # tokens to process before considering growth
    min_growth_interval: int = 500  # minimum tokens between growth events
    
    # Evaluation
    evaluation_window: int = 10000  # tokens for computing metrics
    track_token_usage: bool = True
    track_compression_ratio: bool = True
    track_vocabulary_utilization: bool = True
    
    # Data processing
    dataset_name: str = "roneneldan/TinyStories"
    max_sequence_length: int = 512
    batch_size: int = 1000  # for batch processing
    
    # Logging and visualization
    log_interval: int = 1000
    save_interval: int = 5000
    plot_growth: bool = True
    save_dir: str = "./checkpoints"
```

## Training Process (`train.py`)

### Main Training Loop

```python
def train_growing_tokenizer(config: GrowingBPEConfig):
    # Initialize tokenizer
    tokenizer = GrowingBPETokenizer(config)
    
    # Load dataset
    dataset = load_dataset(config.dataset_name)
    text_stream = create_text_stream(dataset, config)
    
    # Training metrics
    metrics = GrowthMetrics()
    
    processed_tokens = 0
    for batch_text in text_stream:
        # Tokenize batch
        batch_tokens = []
        for text in batch_text:
            tokens = tokenizer.tokenize(text)
            batch_tokens.extend(tokens)
            
        # Update metrics
        metrics.update(batch_text, batch_tokens, tokenizer.vocab_size)
        processed_tokens += len(batch_tokens)
        
        # Logging and checkpointing
        if processed_tokens % config.log_interval == 0:
            metrics.log_current_state()
            
        if processed_tokens % config.save_interval == 0:
            save_checkpoint(tokenizer, metrics, config)
            
    return tokenizer, metrics
```

### Growth Monitoring

```python
class GrowthMetrics:
    def __init__(self):
        self.compression_history = []
        self.vocab_size_history = []
        self.growth_events = []
        self.token_usage_counts = defaultdict(int)
        
    def update(self, texts: List[str], tokens: List[List[int]], vocab_size: int):
        # Compression ratio
        total_chars = sum(len(text) for text in texts)
        total_tokens = sum(len(token_list) for token_list in tokens)
        compression_ratio = total_tokens / total_chars if total_chars > 0 else 0
        
        self.compression_history.append(compression_ratio)
        self.vocab_size_history.append(vocab_size)
        
        # Token usage
        for token_list in tokens:
            for token in token_list:
                self.token_usage_counts[token] += 1
                
    def record_growth_event(self, trigger_reason: str, tokens_added: List[str]):
        self.growth_events.append({
            'step': len(self.compression_history),
            'reason': trigger_reason,
            'tokens_added': tokens_added,
            'vocab_size_before': self.vocab_size_history[-2] if len(self.vocab_size_history) > 1 else 0,
            'vocab_size_after': self.vocab_size_history[-1]
        })
```

## Evaluation Framework (`evaluate.py`)

### Evaluation Metrics

```python
class TokenizerEvaluator:
    def __init__(self, tokenizer: GrowingBPETokenizer):
        self.tokenizer = tokenizer
        
    def evaluate_compression(self, texts: List[str]) -> Dict[str, float]:
        total_chars = sum(len(text) for text in texts)
        total_tokens = sum(len(self.tokenizer.tokenize(text)) for text in texts)
        
        return {
            'compression_ratio': total_tokens / total_chars,
            'avg_tokens_per_text': total_tokens / len(texts),
            'avg_chars_per_token': total_chars / total_tokens
        }
        
    def evaluate_vocabulary_utilization(self, texts: List[str]) -> Dict[str, float]:
        token_counts = defaultdict(int)
        total_tokens = 0
        
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            total_tokens += len(tokens)
            for token in tokens:
                token_counts[token] += 1
                
        active_vocab = len([t for t, c in token_counts.items() if c > 0])
        total_vocab = len(self.tokenizer.vocab)
        
        return {
            'vocabulary_utilization': active_vocab / total_vocab,
            'active_vocabulary_size': active_vocab,
            'total_vocabulary_size': total_vocab,
            'singleton_tokens': len([t for t, c in token_counts.items() if c == 1])
        }
        
    def evaluate_subword_quality(self, texts: List[str]) -> Dict[str, float]:
        # Heuristic measures of subword meaningfulness
        vocab_tokens = list(self.tokenizer.vocab.keys())
        
        # Average token length
        avg_token_length = sum(len(token) for token in vocab_tokens) / len(vocab_tokens)
        
        # Linguistic pattern matching (simple heuristics)
        morphological_tokens = 0
        for token in vocab_tokens:
            if (token.endswith('ing') or token.endswith('ed') or 
                token.endswith('er') or token.startswith('un') or
                token.startswith('re')):
                morphological_tokens += 1
                
        return {
            'avg_token_length': avg_token_length,
            'morphological_alignment': morphological_tokens / len(vocab_tokens),
            'character_diversity': len(set(''.join(vocab_tokens))) / 256  # Assuming 256 possible chars
        }
```

## Experimental Framework

### Pre-configured Experiments (`experiments/`)

**Aggressive Growth Configuration**
```python
# experiments/aggressive_growth.py
from config import GrowingBPEConfig

config = GrowingBPEConfig(
    initial_vocab_strategy="character_only",
    initial_vocab_size=128,
    growth_triggers=["compression", "frequency", "coverage"],
    compression_threshold=0.5,  # More sensitive
    frequency_threshold=5,      # Lower threshold
    growth_patience=500,        # Grow more frequently
    max_candidates_per_growth=20,
    max_vocab_size=5000
)
```

**Conservative Growth Configuration**
```python
# experiments/conservative_growth.py
from config import GrowingBPEConfig

config = GrowingBPEConfig(
    initial_vocab_strategy="frequency_based",
    initial_vocab_size=1000,
    growth_triggers=["compression"],
    compression_threshold=0.8,  # Less sensitive
    frequency_threshold=20,     # Higher threshold
    growth_patience=2000,       # Grow less frequently
    max_candidates_per_growth=5,
    max_vocab_size=2000
)
```

## Performance Optimizations

### Efficient Data Structures
- Use tries for fast prefix matching
- Implement incremental merge rule updates
- Cache tokenization results for repeated text

### Memory Management
- Periodic cleanup of unused tokens
- Sliding window for growth statistics
- Lazy evaluation of growth candidates

### Computational Efficiency
- Batch processing for growth decisions
- Parallel candidate evaluation
- Early stopping for growth trigger evaluation

## Future Research Directions

1. **Hierarchical Growth**: Multi-level tokenization with different granularities
2. **Domain Adaptation**: Specialized growth for different text types
3. **Neural Integration**: Use neural networks to guide growth decisions
4. **Pruning Mechanisms**: Remove underutilized tokens to maintain efficiency
5. **Multi-Modal**: Extend to images, audio, or other modalities
6. **Distributed Growth**: Coordinate growth across multiple tokenizer instances

## Validation and Testing

### Unit Tests
- Individual component functionality
- Growth trigger correctness
- Vocabulary consistency checks

### Integration Tests
- End-to-end tokenization pipeline
- Growth mechanism coordination
- Metric computation accuracy

### Benchmark Comparisons
- Static BPE tokenizers of equivalent size
- Other subword tokenization methods
- Domain adaptation performance