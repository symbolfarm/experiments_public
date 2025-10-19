import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path

from config import CascadeConfig

class TinyStoriesDataset(Dataset):
    """Dataset class for TinyStories with tokenization and chunking."""
    
    def __init__(self, stories: List[str], tokenizer: GPT2TokenizerFast, 
                 max_length: int = 512, stride: int = 256):
        """
        Initialize dataset with stories and tokenizer.
        
        Args:
            stories: List of story strings
            tokenizer: GPT-2 tokenizer
            max_length: Maximum sequence length
            stride: Stride for overlapping chunks (set to max_length for non-overlapping)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Tokenize all stories and create chunks
        self.chunks = []
        self._process_stories(stories)
        
        print(f"Created {len(self.chunks)} chunks from {len(stories)} stories")
    
    def _process_stories(self, stories: List[str]):
        """Process stories into tokenized chunks."""
        for story in stories:
            if not story or len(story.strip()) == 0:
                continue
            
            # Tokenize the story
            tokens = self.tokenizer.encode(story, add_special_tokens=True)
            
            # Skip very short stories
            if len(tokens) < 10:
                continue
            
            # Create overlapping chunks if story is longer than max_length
            if len(tokens) <= self.max_length:
                # Pad short sequences
                if len(tokens) < self.max_length:
                    tokens.extend([self.tokenizer.eos_token_id] * (self.max_length - len(tokens)))
                self.chunks.append(tokens)
            else:
                # Create overlapping chunks for long stories
                for i in range(0, len(tokens) - self.max_length + 1, self.stride):
                    chunk = tokens[i:i + self.max_length]
                    self.chunks.append(chunk)
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a tokenized chunk."""
        return torch.tensor(self.chunks[idx], dtype=torch.long)

class TinyStoriesDataModule:
    """Data module for loading and processing TinyStories dataset."""
    
    def __init__(self, config: CascadeConfig):
        self.config = config
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Create cache directory
        self.cache_dir = Path("./data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self._setup_tokenizer()
    
    def _setup_tokenizer(self):
        """Initialize and configure tokenizer."""
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Tokenizer vocab size: {len(self.tokenizer)}")
        
        # Update config vocab size if different
        if self.config.vocab_size != len(self.tokenizer):
            print(f"Updating vocab_size from {self.config.vocab_size} to {len(self.tokenizer)}")
            self.config.vocab_size = len(self.tokenizer)
    
    def prepare_data(self):
        """Download and cache the dataset."""
        print("Loading TinyStories dataset...")
        
        try:
            # Load dataset
            dataset = load_dataset(self.config.dataset_name)
            
            # Extract stories (text field)
            train_stories = [item['text'] for item in dataset[self.config.train_split]]
            val_stories = [item['text'] for item in dataset[self.config.val_split]]
            
            # Limit dataset size for faster experimentation (remove in production)
            max_train_stories = 50000  # Adjust as needed
            max_val_stories = 5000
            
            if len(train_stories) > max_train_stories:
                train_stories = train_stories[:max_train_stories]
                print(f"Limited training set to {max_train_stories} stories")
            
            if len(val_stories) > max_val_stories:
                val_stories = val_stories[:max_val_stories]
                print(f"Limited validation set to {max_val_stories} stories")
            
            # Create datasets
            self.train_dataset = TinyStoriesDataset(
                train_stories, self.tokenizer, self.config.max_len, stride=256
            )
            
            self.val_dataset = TinyStoriesDataset(
                val_stories, self.tokenizer, self.config.max_len, stride=self.config.max_len  # No overlap for validation
            )
            
            print(f"Training chunks: {len(self.train_dataset)}")
            print(f"Validation chunks: {len(self.val_dataset)}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating dummy dataset for testing...")
            self._create_dummy_dataset()
    
    def _create_dummy_dataset(self):
        """Create a small dummy dataset for testing."""
        dummy_stories = [
            "Once upon a time, there was a little girl named Lucy. She loved to play in the garden.",
            "Tom was a brave boy who liked to explore. One day, he found a magical forest.",
            "The cat sat on the mat. It was a sunny day and the cat was happy.",
            "Mary had a little lamb. The lamb followed Mary everywhere she went.",
            "Jack and Jill went up the hill to fetch a pail of water. They were good friends."
        ] * 1000  # Repeat to create more data
        
        self.train_dataset = TinyStoriesDataset(
            dummy_stories, self.tokenizer, self.config.max_len
        )
        
        val_stories = dummy_stories[:100]  # Small validation set
        self.val_dataset = TinyStoriesDataset(
            val_stories, self.tokenizer, self.config.max_len
        )
        
        print("Created dummy dataset for testing")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,  # Drop last incomplete batch
            collate_fn=collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
            collate_fn=collate_fn
        )
    
    def get_sample_text(self, num_samples: int = 5) -> List[str]:
        """Get sample text for generation evaluation."""
        if self.val_dataset is None:
            return ["Once upon a time"] * num_samples
        
        samples = []
        for i in range(min(num_samples, len(self.val_dataset))):
            tokens = self.val_dataset[i][:20]  # First 20 tokens
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            samples.append(text)
        
        return samples
    
    def decode_tokens(self, tokens: torch.Tensor) -> str:
        """Decode tokens back to text."""
        if tokens.dim() > 1:
            tokens = tokens[0]  # Take first sequence if batch
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

def collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of tokenized sequences
        
    Returns:
        Dictionary with input_ids and targets
    """
    # Stack all sequences
    input_ids = torch.stack(batch)
    
    return {
        'input_ids': input_ids,
        'targets': input_ids.clone()  # For language modeling, targets = inputs
    }

def create_dataloaders(config: CascadeConfig) -> Tuple[DataLoader, DataLoader, TinyStoriesDataModule]:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader, data_module)
    """
    # Create data module
    data_module = TinyStoriesDataModule(config)
    data_module.prepare_data()
    
    # Create dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    return train_loader, val_loader, data_module

def test_dataloader(config: CascadeConfig):
    """Test the dataloader functionality."""
    print("Testing dataloader...")
    
    # Create dataloaders
    train_loader, val_loader, data_module = create_dataloaders(config)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Vocab size: {data_module.get_vocab_size()}")
    
    # Test a batch
    for batch in train_loader:
        input_ids = batch['input_ids']
        targets = batch['targets']
        
        print(f"Batch input_ids shape: {input_ids.shape}")
        print(f"Batch targets shape: {targets.shape}")
        
        # Decode first sequence
        text = data_module.decode_tokens(input_ids[0])
        print(f"Sample text: {text[:100]}...")
        break
    
    # Test sample generation prompts
    samples = data_module.get_sample_text(3)
    print("\nSample prompts:")
    for i, sample in enumerate(samples):
        print(f"{i+1}: {sample}")

if __name__ == "__main__":
    # Test the data module
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Fix tokenizer warning
    
    from config import CascadeConfig
    
    config = CascadeConfig()
    config.batch_size = 4  # Small batch for testing
    config.max_len = 128   # Shorter sequences for testing
    config.num_workers = 0  # Disable multiprocessing for testing
    
    test_dataloader(config)