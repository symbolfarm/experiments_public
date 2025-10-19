import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from typing import Dict, List, Optional, Tuple
import pickle
import os
from pathlib import Path

class TinyStoriesDataset(Dataset):
    """Dataset wrapper for TinyStories with GPT-2 tokenization."""
    
    def __init__(self, texts: List[str], tokenizer: GPT2Tokenizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # For language modeling, labels = input_ids
        }

def load_and_tokenize_data(
    dataset_name: str = "roneneldan/TinyStories",
    train_split: str = "train",
    val_split: str = "validation", 
    max_length: int = 256,
    max_train_samples: Optional[int] = 50000,  # Limit for faster experimentation
    max_val_samples: Optional[int] = 5000,
    cache_dir: str = "./data_cache"
) -> Tuple[TinyStoriesDataset, TinyStoriesDataset, GPT2Tokenizer]:
    """Load and tokenize TinyStories dataset."""
    
    # Create cache directory
    Path(cache_dir).mkdir(exist_ok=True)
    cache_file = Path(cache_dir) / f"processed_data_{max_train_samples}_{max_val_samples}_{max_length}.pkl"
    
    # Check if cached data exists
    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            return data['train_dataset'], data['val_dataset'], data['tokenizer']
    
    print(f"Loading {dataset_name} dataset...")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    
    # Extract texts
    print("Extracting and processing texts...")
    train_texts = []
    val_texts = []
    
    # Process training data
    train_data = dataset[train_split]
    if max_train_samples:
        train_data = train_data.select(range(min(max_train_samples, len(train_data))))
    
    for item in train_data:
        text = item['text'].strip()
        if len(text) > 10:  # Filter very short texts
            train_texts.append(text)
    
    # Process validation data
    if val_split in dataset:
        val_data = dataset[val_split]
        if max_val_samples:
            val_data = val_data.select(range(min(max_val_samples, len(val_data))))
        
        for item in val_data:
            text = item['text'].strip()
            if len(text) > 10:
                val_texts.append(text)
    else:
        # Split training data if no validation split
        split_idx = int(0.9 * len(train_texts))
        val_texts = train_texts[split_idx:]
        train_texts = train_texts[:split_idx]
    
    print(f"Loaded {len(train_texts)} training texts, {len(val_texts)} validation texts")
    
    # Create datasets
    train_dataset = TinyStoriesDataset(train_texts, tokenizer, max_length)
    val_dataset = TinyStoriesDataset(val_texts, tokenizer, max_length)
    
    # Cache the processed data
    print(f"Caching processed data to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'tokenizer': tokenizer
        }, f)
    
    return train_dataset, val_dataset, tokenizer

def create_data_loaders(
    train_dataset: TinyStoriesDataset,
    val_dataset: TinyStoriesDataset,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation."""
    
    def collate_fn(batch):
        """Custom collate function to handle batching."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True  # Drop last incomplete batch for consistent training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader

def get_data_loaders(
    dataset_name: str = "roneneldan/TinyStories",
    train_split: str = "train",
    val_split: str = "validation",
    max_length: int = 256,
    batch_size: int = 32,
    max_train_samples: Optional[int] = 50000,
    max_val_samples: Optional[int] = 5000,
    num_workers: int = 0,
    pin_memory: bool = True,
    cache_dir: str = "./data_cache"
) -> Tuple[DataLoader, DataLoader, GPT2Tokenizer]:
    """Convenience function to get everything in one call."""
    
    # Load and tokenize data
    train_dataset, val_dataset, tokenizer = load_and_tokenize_data(
        dataset_name=dataset_name,
        train_split=train_split,
        val_split=val_split,
        max_length=max_length,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        cache_dir=cache_dir
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Created data loaders:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    
    return train_loader, val_loader, tokenizer

def decode_batch(batch: Dict[str, torch.Tensor], tokenizer: GPT2Tokenizer, num_samples: int = 3) -> List[str]:
    """Decode a batch of token IDs back to text for inspection."""
    input_ids = batch['input_ids']
    
    decoded_texts = []
    for i in range(min(num_samples, input_ids.size(0))):
        # Remove padding tokens
        tokens = input_ids[i]
        # Find first pad token
        try:
            pad_start = (tokens == tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0].item()
            tokens = tokens[:pad_start]
        except IndexError:
            # No padding found
            pass
        
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        decoded_texts.append(text)
    
    return decoded_texts

if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    
    # Load data
    train_loader, val_loader, tokenizer = get_data_loaders(
        max_train_samples=1000,  # Small test
        max_val_samples=100,
        batch_size=4,
        max_length=128
    )
    
    # Test a batch
    print("\nTesting batch...")
    batch = next(iter(train_loader))
    print(f"Batch shapes:")
    for key, tensor in batch.items():
        print(f"  {key}: {tensor.shape}")
    
    # Decode some examples
    print("\nDecoded examples:")
    decoded = decode_batch(batch, tokenizer, num_samples=2)
    for i, text in enumerate(decoded):
        print(f"Example {i+1}: {text[:100]}...")
    
    print("\nData loading test completed!")