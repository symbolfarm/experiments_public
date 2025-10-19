import re
import json
import pickle
from collections import defaultdict, Counter, deque
from typing import List, Dict, Tuple, Set, Optional, Any
import unicodedata
import logging
from dataclasses import dataclass
from config import GrowingBPEConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GrowthEvent:
    """Record of a vocabulary growth event."""
    step: int
    tokens_processed: int
    trigger_reason: str
    tokens_added: List[str]
    vocab_size_before: int
    vocab_size_after: int
    compression_ratio_before: float
    compression_ratio_after: float

class GrowthTrigger:
    """Base class for growth triggers."""
    
    def __init__(self, config: GrowingBPEConfig):
        self.config = config
        
    def should_trigger(self, text: str, tokens: List[int], tokenizer: 'GrowingBPETokenizer') -> bool:
        """Check if growth should be triggered."""
        raise NotImplementedError

class CompressionTrigger(GrowthTrigger):
    """Trigger growth when compression ratio is poor."""
    
    def __init__(self, config: GrowingBPEConfig):
        super().__init__(config)
        self.recent_ratios = deque(maxlen=config.compression_window_size)
        
    def should_trigger(self, text: str, tokens: List[int], tokenizer: 'GrowingBPETokenizer') -> bool:
        if len(text) == 0:
            return False
            
        ratio = len(tokens) / len(text)  # tokens per character
        self.recent_ratios.append(ratio)
        
        if len(self.recent_ratios) < min(100, self.config.compression_window_size):
            return False
            
        avg_ratio = sum(self.recent_ratios) / len(self.recent_ratios)
        return avg_ratio > self.config.compression_threshold

class FrequencyTrigger(GrowthTrigger):
    """Trigger growth when frequent patterns aren't well tokenized."""
    
    def __init__(self, config: GrowingBPEConfig):
        super().__init__(config)
        self.char_ngram_counts = defaultdict(int)
        self.total_chars = 0
        
    def should_trigger(self, text: str, tokens: List[int], tokenizer: 'GrowingBPETokenizer') -> bool:
        # Count character n-grams
        for n in range(self.config.ngram_range[0], self.config.ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                if ngram.strip() and not ngram.isspace():  # Skip whitespace-only ngrams
                    self.char_ngram_counts[ngram] += 1
                    
        self.total_chars += len(text)
        
        # Check if any frequent n-grams aren't in vocabulary
        vocab_strings = set(tokenizer.id_to_token.values())
        for ngram, count in self.char_ngram_counts.items():
            if count >= self.config.frequency_threshold and ngram not in vocab_strings:
                return True
                
        return False

class CoverageTrigger(GrowthTrigger):
    """Trigger growth when encountering high OOV rates."""
    
    def __init__(self, config: GrowingBPEConfig):
        super().__init__(config)
        self.char_coverage = set()
        
    def should_trigger(self, text: str, tokens: List[int], tokenizer: 'GrowingBPETokenizer') -> bool:
        # Track character coverage
        text_chars = set(text)
        vocab_chars = set(''.join(tokenizer.id_to_token.values()))
        
        uncovered_chars = text_chars - vocab_chars
        if len(text_chars) == 0:
            return False
            
        oov_rate = len(uncovered_chars) / len(text_chars)
        return oov_rate > self.config.oov_threshold

class GrowthMechanism:
    """Base class for growth mechanisms."""
    
    def __init__(self, config: GrowingBPEConfig):
        self.config = config
        
    def find_candidates(self, text: str, tokenizer: 'GrowingBPETokenizer') -> List[str]:
        """Find candidate tokens for growth."""
        raise NotImplementedError

class StandardBPEGrowth(GrowthMechanism):
    """Standard BPE extension mechanism."""
    
    def find_candidates(self, text: str, tokenizer: 'GrowingBPETokenizer') -> List[str]:
        # Tokenize text with current vocabulary
        tokens = tokenizer._tokenize_text(text)
        
        # Count adjacent token pairs
        pair_counts = defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += 1
            
        # Find most frequent pairs
        candidates = []
        for (token1, token2), count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= self.config.min_pattern_frequency:
                # Merge the tokens
                str1 = tokenizer.id_to_token[token1]
                str2 = tokenizer.id_to_token[token2]
                merged = str1 + str2
                
                # Check if it's a reasonable merge
                if self._is_valid_merge(merged):
                    candidates.append(merged)
                    
                if len(candidates) >= self.config.max_candidates_per_growth:
                    break
                    
        return candidates
    
    def _is_valid_merge(self, token: str) -> bool:
        """Check if a merged token is valid."""
        if len(token) < 2 or len(token) > self.config.max_pattern_length:
            return False
        if token.isspace():
            return False
        if len(set(token)) == 1:  # Repeated character
            return False
        return True

class PatternMiningGrowth(GrowthMechanism):
    """Pattern mining growth mechanism."""
    
    def find_candidates(self, text: str, tokenizer: 'GrowingBPETokenizer') -> List[str]:
        candidates = []
        vocab_strings = set(tokenizer.id_to_token.values())
        
        # Find repeated substrings
        for length in range(2, min(self.config.max_pattern_length + 1, len(text))):
            patterns = defaultdict(int)
            for i in range(len(text) - length + 1):
                pattern = text[i:i+length]
                if self._is_meaningful_pattern(pattern):
                    patterns[pattern] += 1
                    
            for pattern, count in patterns.items():
                if (count >= self.config.min_pattern_frequency and 
                    pattern not in vocab_strings and
                    len(candidates) < self.config.max_candidates_per_growth):
                    candidates.append(pattern)
                    
        return candidates
    
    def _is_meaningful_pattern(self, pattern: str) -> bool:
        """Check if a pattern is meaningful for tokenization."""
        if not pattern.strip() or pattern.isspace():
            return False
        if len(set(pattern)) == 1:  # Repeated character
            return False
        if pattern.startswith(' ') and pattern.endswith(' '):
            return False
        return True

class AdaptiveMergingGrowth(GrowthMechanism):
    """Adaptive merging based on tokenization efficiency."""
    
    def find_candidates(self, text: str, tokenizer: 'GrowingBPETokenizer') -> List[str]:
        # Analyze current tokenization
        tokens = tokenizer._tokenize_text(text)
        token_strings = [tokenizer.id_to_token[t] for t in tokens]
        
        # Find inefficient spans (where many short tokens are used)
        candidates = []
        i = 0
        while i < len(token_strings) - 1 and len(candidates) < self.config.max_candidates_per_growth:
            # Look for sequences of short tokens
            if len(token_strings[i]) <= 2 and len(token_strings[i + 1]) <= 2:
                # Try merging this sequence
                merged = token_strings[i] + token_strings[i + 1]
                if self._is_valid_merge(merged):
                    candidates.append(merged)
                i += 2
            else:
                i += 1
                
        return candidates
    
    def _is_valid_merge(self, token: str) -> bool:
        """Check if a merged token is valid."""
        if len(token) < 2 or len(token) > self.config.max_pattern_length:
            return False
        if token.isspace():
            return False
        return True

class GrowingBPETokenizer:
    """A BPE tokenizer that grows its vocabulary dynamically."""
    
    def __init__(self, config: GrowingBPEConfig):
        self.config = config
        self.token_to_id = {}
        self.id_to_token = {}
        self.merge_rules = []  # List of (token1, token2) -> merged_token
        self.vocab_size = 0
        
        # Growth components
        self.growth_triggers = self._create_triggers()
        self.growth_mechanism = self._create_mechanism()
        self.growth_history = []
        
        # Statistics tracking
        self.tokens_processed = 0
        self.last_growth_step = 0
        self.token_usage_counts = defaultdict(int)
        
        # Caching
        if config.cache_tokenization:
            self.tokenization_cache = {}
        
        # Initialize vocabulary
        self._initialize_vocabulary()
        
        logger.info(f"Initialized tokenizer with {self.vocab_size} tokens")
    
    def _create_triggers(self) -> List[GrowthTrigger]:
        """Create growth triggers based on configuration."""
        triggers = []
        for trigger_name in self.config.growth_triggers:
            if trigger_name == "compression":
                triggers.append(CompressionTrigger(self.config))
            elif trigger_name == "frequency":
                triggers.append(FrequencyTrigger(self.config))
            elif trigger_name == "coverage":
                triggers.append(CoverageTrigger(self.config))
            else:
                logger.warning(f"Unknown trigger: {trigger_name}")
        return triggers
    
    def _create_mechanism(self) -> GrowthMechanism:
        """Create growth mechanism based on configuration."""
        if self.config.growth_mechanism == "standard_bpe":
            return StandardBPEGrowth(self.config)
        elif self.config.growth_mechanism == "pattern_mining":
            return PatternMiningGrowth(self.config)
        elif self.config.growth_mechanism == "adaptive_merging":
            return AdaptiveMergingGrowth(self.config)
        else:
            logger.warning(f"Unknown mechanism: {self.config.growth_mechanism}, using standard_bpe")
            return StandardBPEGrowth(self.config)
    
    def _initialize_vocabulary(self):
        """Initialize the vocabulary based on configuration."""
        if self.config.initial_vocab_strategy == "character_only":
            self._build_character_vocab()
        elif self.config.initial_vocab_strategy == "minimal_subwords":
            self._build_minimal_subword_vocab()
        elif self.config.initial_vocab_strategy == "frequency_based":
            self._build_frequency_vocab()
        else:
            logger.warning(f"Unknown vocab strategy: {self.config.initial_vocab_strategy}, using character_only")
            self._build_character_vocab()
    
    def _build_character_vocab(self):
        """Build character-only vocabulary."""
        # Basic ASCII characters
        chars = set()
        for i in range(32, 127):  # Printable ASCII
            chars.add(chr(i))
        
        # Add special tokens
        if self.config.include_special_tokens:
            for token in self.config.special_tokens:
                chars.add(token)
        
        # Sort for deterministic ordering
        for i, char in enumerate(sorted(chars)):
            self.token_to_id[char] = i
            self.id_to_token[i] = char
            
        self.vocab_size = len(self.token_to_id)
    
    def _build_minimal_subword_vocab(self):
        """Build vocabulary with characters + common subwords."""
        self._build_character_vocab()
        
        # Add common English patterns
        common_patterns = [
            'the', 'and', 'ing', 'ed', 'er', 'ly', 'tion', 'ness',
            'un', 're', 'in', 'on', 'at', 'to', 'of', 'for'
        ]
        
        for pattern in common_patterns:
            if pattern not in self.token_to_id:
                self.token_to_id[pattern] = self.vocab_size
                self.id_to_token[self.vocab_size] = pattern
                self.vocab_size += 1
    
    def _build_frequency_vocab(self):
        """Build vocabulary based on character frequency."""
        # Start with character vocab
        self._build_character_vocab()
        
        # This would typically analyze a sample of the dataset
        # For now, just use the minimal subword approach
        # In a real implementation, you'd analyze text statistics
        logger.info("Frequency-based vocab building not fully implemented, using minimal subwords")
        self._build_minimal_subword_vocab()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text according to configuration."""
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.handle_punctuation == "separate":
            # Add spaces around punctuation
            text = re.sub(r'([.!?,:;])', r' \1 ', text)
        elif self.config.handle_punctuation == "remove":
            text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text with current vocabulary (without growth)."""
        # Apply BPE algorithm
        tokens = []
        
        # Start with character-level tokenization
        chars = list(text)
        
        # Apply merge rules
        for rule in self.merge_rules:
            token1, token2, merged = rule
            # Replace occurrences of token1 + token2 with merged
            i = 0
            while i < len(chars) - 1:
                if (chars[i] == self.id_to_token.get(token1, '') and 
                    chars[i + 1] == self.id_to_token.get(token2, '')):
                    chars[i:i+2] = [merged]
                    i += 1
                else:
                    i += 1
        
        # Convert to token IDs
        for char in chars:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                # Use UNK token
                unk_id = self.token_to_id.get('<unk>', 0)
                tokens.append(unk_id)
        
        return tokens
    
    def _should_grow(self, text: str, tokens: List[int]) -> bool:
        """Check if vocabulary should grow."""
        # Check if we've processed enough tokens since last growth
        if self.tokens_processed - self.last_growth_step < self.config.growth_patience:
            return False
        
        # Check if we're within growth cooldown
        if self.tokens_processed - self.last_growth_step < self.config.growth_cooldown:
            return False
        
        # Check vocabulary size limit
        if self.vocab_size >= self.config.max_vocab_size:
            return False
        
        # Check maximum growth events
        if len(self.growth_history) >= self.config.max_growth_events:
            return False
        
        # Check triggers
        for trigger in self.growth_triggers:
            if trigger.should_trigger(text, tokens, self):
                return True
        
        return False
    
    def _grow_vocabulary(self, text: str, tokens: List[int]):
        """Grow the vocabulary by adding new tokens."""
        # Find candidates
        candidates = self.growth_mechanism.find_candidates(text, self)
        
        if not candidates:
            logger.debug("No growth candidates found")
            return
        
        # Record state before growth
        compression_before = len(tokens) / len(text) if len(text) > 0 else 0
        vocab_size_before = self.vocab_size
        
        # Add candidates to vocabulary
        added_tokens = []
        for candidate in candidates[:self.config.max_candidates_per_growth]:
            if candidate not in self.token_to_id:
                self.token_to_id[candidate] = self.vocab_size
                self.id_to_token[self.vocab_size] = candidate
                self.vocab_size += 1
                added_tokens.append(candidate)
        
        # Record growth event
        compression_after = self._compute_compression_ratio(text)
        
        growth_event = GrowthEvent(
            step=len(self.growth_history),
            tokens_processed=self.tokens_processed,
            trigger_reason="multi_trigger",  # Could be more specific
            tokens_added=added_tokens,
            vocab_size_before=vocab_size_before,
            vocab_size_after=self.vocab_size,
            compression_ratio_before=compression_before,
            compression_ratio_after=compression_after
        )
        
        self.growth_history.append(growth_event)
        self.last_growth_step = self.tokens_processed
        
        logger.info(f"Vocabulary grew: {vocab_size_before} -> {self.vocab_size} tokens")
        logger.info(f"Added tokens: {added_tokens[:3]}{'...' if len(added_tokens) > 3 else ''}")
    
    def _compute_compression_ratio(self, text: str) -> float:
        """Compute current compression ratio."""
        if len(text) == 0:
            return 0.0
        tokens = self._tokenize_text(text)
        return len(tokens) / len(text)
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text and potentially grow vocabulary."""
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Check cache
        if self.config.cache_tokenization and text in self.tokenization_cache:
            tokens = self.tokenization_cache[text]
        else:
            # Tokenize with current vocabulary
            tokens = self._tokenize_text(text)
            
            # Cache result
            if self.config.cache_tokenization:
                self.tokenization_cache[text] = tokens
        
        # Update statistics
        self.tokens_processed += len(tokens)
        for token in tokens:
            self.token_usage_counts[token] += 1
        
        # Check if growth is needed
        if self._should_grow(text, tokens):
            self._grow_vocabulary(text, tokens)
            
            # Clear cache since vocabulary changed
            if self.config.cache_tokenization:
                self.tokenization_cache.clear()
            
            # Re-tokenize with new vocabulary
            tokens = self._tokenize_text(text)
        
        return tokens
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.id_to_token.get(token_id, '<unk>') for token_id in token_ids]
        return ''.join(tokens)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the current vocabulary."""
        return self.token_to_id.copy()
    
    def get_vocab_size(self) -> int:
        """Get the current vocabulary size."""
        return self.vocab_size
    
    def get_growth_history(self) -> List[GrowthEvent]:
        """Get the history of growth events."""
        return self.growth_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tokenizer statistics."""
        return {
            'vocab_size': self.vocab_size,
            'tokens_processed': self.tokens_processed,
            'growth_events': len(self.growth_history),
            'last_growth_step': self.last_growth_step,
            'token_usage_distribution': dict(self.token_usage_counts),
            'cache_size': len(self.tokenization_cache) if self.config.cache_tokenization else 0
        }
    
    def save(self, filepath: str):
        """Save tokenizer to file."""
        data = {
            'config': self.config.to_dict(),
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'merge_rules': self.merge_rules,
            'vocab_size': self.vocab_size,
            'growth_history': [
                {
                    'step': event.step,
                    'tokens_processed': event.tokens_processed,
                    'trigger_reason': event.trigger_reason,
                    'tokens_added': event.tokens_added,
                    'vocab_size_before': event.vocab_size_before,
                    'vocab_size_after': event.vocab_size_after,
                    'compression_ratio_before': event.compression_ratio_before,
                    'compression_ratio_after': event.compression_ratio_after
                }
                for event in self.growth_history
            ],
            'tokens_processed': self.tokens_processed,
            'last_growth_step': self.last_growth_step,
            'token_usage_counts': dict(self.token_usage_counts)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'GrowingBPETokenizer':
        """Load tokenizer from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = GrowingBPEConfig.from_dict(data['config'])
        tokenizer = cls(config)
        
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        tokenizer.merge_rules = data['merge_rules']
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.tokens_processed = data['tokens_processed']
        tokenizer.last_growth_step = data['last_growth_step']
        tokenizer.token_usage_counts = defaultdict(int, data['token_usage_counts'])
        
        # Reconstruct growth history
        tokenizer.growth_history = [
            GrowthEvent(**event_data) for event_data in data['growth_history']
        ]
        
        return tokenizer

if __name__ == "__main__":
    # Test the tokenizer
    from config import ExperimentConfigs
    
    config = ExperimentConfigs.minimal_experiment()
    tokenizer = GrowingBPETokenizer(config)
    
    # Test tokenization
    test_texts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating and powerful.",
        "Natural language processing enables computers to understand text."
    ]
    
    print("=== Testing Growing BPE Tokenizer ===")
    print(f"Initial vocabulary size: {tokenizer.get_vocab_size()}")
    
    for i, text in enumerate(test_texts):
        print(f"\nText {i+1}: {text}")
        tokens = tokenizer.tokenize(text)
        print(f"Tokens: {tokens}")
        print(f"Detokenized: {tokenizer.detokenize(tokens)}")
        print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    print(f"\nFinal statistics:")
    stats = tokenizer.get_statistics()
    for key, value in stats.items():
        if key != 'token_usage_distribution':  # Skip large dict
            print(f"{key}: {value}")
    
    print(f"\nGrowth events: {len(tokenizer.get_growth_history())}")
    for event in tokenizer.get_growth_history():
        print(f"  Step {event.step}: Added {len(event.tokens_added)} tokens, vocab {event.vocab_size_before} -> {event.vocab_size_after}")