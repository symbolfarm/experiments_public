import os
import json
import pickle
from typing import Iterator, List, Optional, Dict, Any
from datasets import load_dataset, Dataset
import logging
from pathlib import Path
from config import GrowingBPEConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TinyStoriesDataLoader:
    """Data loader for TinyStories dataset with streaming and caching support."""
    
    def __init__(self, config: GrowingBPEConfig):
        self.config = config
        self.cache_dir = Path(config.save_dir) / "data_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset state
        self.dataset = None
        self.validation_dataset = None
        self.text_stats = {}
        
    def load_dataset(self, force_reload: bool = False) -> None:
        """Load the TinyStories dataset."""
        cache_file = self.cache_dir / f"dataset_cache_{self.config.dataset_split}.pkl"
        
        if not force_reload and cache_file.exists():
            logger.info("Loading dataset from cache...")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.dataset = cached_data['dataset']
                    self.validation_dataset = cached_data.get('validation_dataset')
                    self.text_stats = cached_data.get('text_stats', {})
                logger.info(f"Loaded cached dataset with {len(self.dataset)} samples")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Reloading dataset...")
        
        logger.info(f"Loading {self.config.dataset_name} dataset...")
        
        try:
            # Load main dataset
            if self.config.streaming:
                # Use streaming for large datasets
                dataset = load_dataset(
                    self.config.dataset_name,
                    split=self.config.dataset_split,
                    streaming=True
                )
                
                # Convert to list for easier handling (for smaller experiments)
                samples = []
                for i, sample in enumerate(dataset):
                    if self.config.max_samples and i >= self.config.max_samples:
                        break
                    samples.append(sample)
                    
                self.dataset = samples
            else:
                # Load full dataset into memory
                dataset = load_dataset(
                    self.config.dataset_name,
                    split=self.config.dataset_split
                )
                
                if self.config.max_samples:
                    dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
                
                self.dataset = dataset
            
            # Load validation dataset if specified
            if self.config.validation_split:
                try:
                    if self.config.streaming:
                        val_dataset = load_dataset(
                            self.config.dataset_name,
                            split=self.config.validation_split,
                            streaming=True
                        )
                        # Take a smaller validation set
                        val_samples = []
                        for i, sample in enumerate(val_dataset):
                            if i >= 1000:  # Limit validation set size
                                break
                            val_samples.append(sample)
                        self.validation_dataset = val_samples
                    else:
                        self.validation_dataset = load_dataset(
                            self.config.dataset_name,
                            split=self.config.validation_split
                        )
                        if len(self.validation_dataset) > 1000:
                            self.validation_dataset = self.validation_dataset.select(range(1000))
                except Exception as e:
                    logger.warning(f"Could not load validation split: {e}")
                    self.validation_dataset = None
            
            # Compute text statistics
            self._compute_text_stats()
            
            # Cache the dataset
            logger.info("Caching dataset...")
            cache_data = {
                'dataset': self.dataset,
                'validation_dataset': self.validation_dataset,
                'text_stats': self.text_stats
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Loaded dataset with {len(self.dataset)} samples")
            if self.validation_dataset:
                logger.info(f"Validation set: {len(self.validation_dataset)} samples")
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _compute_text_stats(self):
        """Compute basic statistics about the text data."""
        logger.info("Computing text statistics...")
        
        total_chars = 0
        total_stories = 0
        char_counts = {}
        story_lengths = []
        
        # Sample for stats (don't process entire dataset)
        sample_size = min(1000, len(self.dataset))
        
        for i, sample in enumerate(self.dataset):
            if i >= sample_size:
                break
                
            text = self._extract_text(sample)
            if text:
                total_chars += len(text)
                total_stories += 1
                story_lengths.append(len(text))
                
                # Count characters
                for char in text:
                    char_counts[char] = char_counts.get(char, 0) + 1
        
        self.text_stats = {
            'total_chars_sampled': total_chars,
            'total_stories_sampled': total_stories,
            'avg_story_length': total_chars / total_stories if total_stories > 0 else 0,
            'min_story_length': min(story_lengths) if story_lengths else 0,
            'max_story_length': max(story_lengths) if story_lengths else 0,
            'unique_chars': len(char_counts),
            'char_frequency': char_counts,
            'sample_size': sample_size
        }
        
        logger.info(f"Text stats: {total_stories} stories, avg length {self.text_stats['avg_story_length']:.1f} chars")
        logger.info(f"Character vocabulary: {len(char_counts)} unique characters")
    
    def _extract_text(self, sample: Dict[str, Any]) -> Optional[str]:
        """Extract text from a dataset sample."""
        # TinyStories has 'text' field
        if isinstance(sample, dict):
            return sample.get('text', '')
        elif hasattr(sample, 'text'):
            return sample.text
        else:
            logger.warning(f"Could not extract text from sample: {type(sample)}")
            return None
    
    def get_text_stream(self, batch_size: Optional[int] = None) -> Iterator[List[str]]:
        """Get a stream of text batches."""
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        batch_size = batch_size or self.config.batch_size
        batch = []
        
        for sample in self.dataset:
            text = self._extract_text(sample)
            if text:
                batch.append(text)
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        
        # Yield remaining batch
        if batch:
            yield batch
    
    def get_validation_texts(self, max_samples: Optional[int] = None) -> List[str]:
        """Get validation texts for evaluation."""
        if not self.validation_dataset:
            logger.warning("No validation dataset available")
            return []
        
        texts = []
        max_samples = max_samples or len(self.validation_dataset)
        
        for i, sample in enumerate(self.validation_dataset):
            if i >= max_samples:
                break
            text = self._extract_text(sample)
            if text:
                texts.append(text)
        
        return texts
    
    def get_sample_texts(self, n_samples: int = 10) -> List[str]:
        """Get a small sample of texts for analysis."""
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        texts = []
        for i, sample in enumerate(self.dataset):
            if i >= n_samples:
                break
            text = self._extract_text(sample)
            if text:
                texts.append(text)
        
        return texts
    
    def get_text_stats(self) -> Dict[str, Any]:
        """Get text statistics."""
        return self.text_stats.copy()
    
    def print_dataset_info(self):
        """Print information about the loaded dataset."""
        if not self.dataset:
            print("No dataset loaded")
            return
        
        print("=== Dataset Information ===")
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Split: {self.config.dataset_split}")
        print(f"Samples: {len(self.dataset)}")
        
        if self.validation_dataset:
            print(f"Validation samples: {len(self.validation_dataset)}")
        
        if self.text_stats:
            stats = self.text_stats
            print(f"Average story length: {stats['avg_story_length']:.1f} characters")
            print(f"Story length range: {stats['min_story_length']} - {stats['max_story_length']}")
            print(f"Unique characters: {stats['unique_chars']}")
            
            # Show most common characters
            char_freq = stats.get('char_frequency', {})
            if char_freq:
                common_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                print("Most common characters:")
                for char, count in common_chars:
                    if char == ' ':
                        char_display = 'SPACE'
                    elif char == '\n':
                        char_display = 'NEWLINE'
                    elif char == '\t':
                        char_display = 'TAB'
                    else:
                        char_display = repr(char)
                    print(f"  {char_display}: {count}")
        
        print("=" * 27)
    
    def save_sample_data(self, filepath: str, n_samples: int = 100):
        """Save a sample of the data for inspection."""
        if not self.dataset:
            raise ValueError("Dataset not loaded")
        
        samples = self.get_sample_texts(n_samples)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, text in enumerate(samples):
                f.write(f"=== Sample {i+1} ===\n")
                f.write(text)
                f.write("\n\n")
        
        logger.info(f"Saved {len(samples)} samples to {filepath}")

class TextProcessor:
    """Utility class for text processing operations."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text for tokenization."""
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove control characters except newlines and tabs
        cleaned = []
        for char in text:
            if ord(char) >= 32 or char in ['\n', '\t']:
                cleaned.append(char)
        
        return ''.join(cleaned)
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def get_text_complexity_score(text: str) -> float:
        """Compute a simple complexity score for text."""
        if not text:
            return 0.0
        
        # Factors: unique characters, average word length, punctuation density
        unique_chars = len(set(text))
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        punct_density = sum(1 for char in text if not char.isalnum() and not char.isspace()) / len(text)
        
        # Normalize and combine
        char_score = unique_chars / 100.0  # Normalize to roughly 0-1
        word_score = avg_word_length / 10.0  # Normalize to roughly 0-1
        punct_score = punct_density * 10.0  # Amplify since it's usually small
        
        return (char_score + word_score + punct_score) / 3.0

def create_data_loader(config: GrowingBPEConfig) -> TinyStoriesDataLoader:
    """Create a data loader with the given configuration."""
    return TinyStoriesDataLoader(config)

if __name__ == "__main__":
    # Test the data loader
    from config import ExperimentConfigs
    
    # Use minimal config for testing
    config = ExperimentConfigs.minimal_experiment()
    config.max_samples = 100  # Limit for testing
    
    print("=== Testing TinyStories Data Loader ===")
    
    # Create and load data
    data_loader = create_data_loader(config)
    data_loader.load_dataset()
    data_loader.print_dataset_info()
    
    # Test text stream
    print("\n=== Testing Text Stream ===")
    batch_count = 0
    total_texts = 0
    
    for batch in data_loader.get_text_stream(batch_size=10):
        batch_count += 1
        total_texts += len(batch)
        
        if batch_count == 1:
            print(f"First batch: {len(batch)} texts")
            print(f"Sample text: {batch[0][:100]}...")
        
        if batch_count >= 3:  # Limit for testing
            break
    
    print(f"Processed {batch_count} batches, {total_texts} texts")
    
    # Test validation data
    print("\n=== Testing Validation Data ===")
    val_texts = data_loader.get_validation_texts(max_samples=5)
    print(f"Validation texts: {len(val_texts)}")
    if val_texts:
        print(f"Sample validation text: {val_texts[0][:100]}...")
    
    # Test text processing
    print("\n=== Testing Text Processing ===")
    sample_text = "  Hello    world!  \n\n  This is a test.  "
    cleaned = TextProcessor.clean_text(sample_text)
    print(f"Original: {repr(sample_text)}")
    print(f"Cleaned: {repr(cleaned)}")
    
    sentences = TextProcessor.split_into_sentences(cleaned)
    print(f"Sentences: {sentences}")
    
    complexity = TextProcessor.get_text_complexity_score(cleaned)
    print(f"Complexity score: {complexity:.3f}")
    
    print("\n=== Test Complete ===")