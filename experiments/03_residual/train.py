import os
import time
import json
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import asdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from config import ResidualTransformerConfig
from model import ResidualTransformer
from baseline_model import BaselineTransformer
from data import get_data_loaders, decode_batch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingMetrics:
    """Track training metrics for both models."""
    
    def __init__(self):
        self.metrics_history = {
            'residual': {'train_loss': [], 'val_loss': [], 'perplexity': []},
            'baseline': {'train_loss': [], 'val_loss': [], 'perplexity': []}
        }
        self.start_time = time.time()
        
    def add_metrics(self, model_type: str, train_loss: float, val_loss: float, perplexity: float):
        """Add metrics for a specific model type."""
        self.metrics_history[model_type]['train_loss'].append(train_loss)
        self.metrics_history[model_type]['val_loss'].append(val_loss)
        self.metrics_history[model_type]['perplexity'].append(perplexity)
    
    def get_latest_metrics(self, model_type: str) -> Dict[str, float]:
        """Get the latest metrics for a model type."""
        history = self.metrics_history[model_type]
        return {
            'train_loss': history['train_loss'][-1] if history['train_loss'] else float('inf'),
            'val_loss': history['val_loss'][-1] if history['val_loss'] else float('inf'),
            'perplexity': history['perplexity'][-1] if history['perplexity'] else float('inf')
        }
    
    def save_metrics(self, save_path: str):
        """Save metrics to JSON file."""
        metrics_with_time = {
            'training_duration': time.time() - self.start_time,
            'metrics': self.metrics_history
        }
        with open(save_path, 'w') as f:
            json.dump(metrics_with_time, f, indent=2)

class ModelTrainer:
    """Trainer that can handle both residual and baseline models."""
    
    def __init__(self, config: ResidualTransformerConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models based on config
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        
        if config.model_type in ['residual', 'both']:
            self.models['residual'] = ResidualTransformer(config).to(self.device)
            
        if config.model_type in ['baseline', 'both']:
            self.models['baseline'] = BaselineTransformer(config).to(self.device)
        
        # Setup optimizers for each model
        for name, model in self.models.items():
            self.optimizers[name] = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
                eps=config.eps
            )
            
            # Setup schedulers
            if config.scheduler == 'cosine':
                self.schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizers[name], T_max=config.max_epochs
                )
            else:  # constant
                self.schedulers[name] = None
        
        # Training state
        self.metrics = TrainingMetrics()
        self.best_val_loss = {name: float('inf') for name in self.models.keys()}
        self.patience_counter = {name: 0 for name in self.models.keys()}
        
        # Setup save directory
        self.save_dir = Path(config.save_dir) / config.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        logger.info(f"Initialized trainer with models: {list(self.models.keys())}")
        for name, model in self.models.items():
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"{name} model parameters: {param_count:,}")
    
    def train_epoch(self, model_name: str, model: torch.nn.Module, train_loader: DataLoader) -> float:
        """Train one epoch for a specific model."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        optimizer = self.optimizers[model_name]
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, loss = model(input_ids, labels)
                self.scaler.scale(loss).backward()
                
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits, loss = model(input_ids, labels)
                loss.backward()
                
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.log_interval == 0:
                logger.info(f"{model_name} - Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, model_name: str, model: torch.nn.Module, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on validation set."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        for batch in val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, loss = model(input_ids, labels)
            else:
                logits, loss = model(input_ids, labels)
            
            # Calculate number of non-padding tokens
            mask = (labels != -100).float()
            num_tokens = mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    @torch.no_grad()
    def generate_samples(self, model_name: str, model: torch.nn.Module, tokenizer, num_samples: int = 3) -> List[str]:
        """Generate sample text for evaluation."""
        model.eval()
        
        samples = []
        for _ in range(num_samples):
            # Simple prompt
            prompt = "Once upon a time"
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Generate
            generated = model.generate(input_ids, max_length=self.config.max_generate_length)
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            samples.append(text)
        
        return samples
    
    def save_checkpoint(self, model_name: str, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        model = self.models[model_name]
        optimizer = self.optimizers[model_name]
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': asdict(self.config),
            'best_val_loss': self.best_val_loss[model_name]
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"{model_name}_checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / f"{model_name}_best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best {model_name} model at epoch {epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, tokenizer):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config.max_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.config.max_epochs}")
            
            # Train each model
            for model_name, model in self.models.items():
                logger.info(f"\nTraining {model_name} model...")
                
                # Train epoch
                train_loss = self.train_epoch(model_name, model, train_loader)
                
                # Evaluate
                val_loss, perplexity = self.evaluate(model_name, model, val_loader)
                
                # Update metrics
                self.metrics.add_metrics(model_name, train_loss, val_loss, perplexity)
                
                # Log metrics
                logger.info(f"{model_name} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
                
                # Check for best model
                is_best = val_loss < self.best_val_loss[model_name]
                if is_best:
                    self.best_val_loss[model_name] = val_loss
                    self.patience_counter[model_name] = 0
                else:
                    self.patience_counter[model_name] += 1
                
                # Save checkpoint
                if (epoch + 1) % 5 == 0 or is_best:
                    self.save_checkpoint(model_name, epoch, is_best)
                
                # Generate samples
                if (epoch + 1) % 10 == 0:
                    samples = self.generate_samples(model_name, model, tokenizer)
                    logger.info(f"{model_name} generated samples:")
                    for i, sample in enumerate(samples):
                        logger.info(f"  Sample {i+1}: {sample[:100]}...")
                
                # Update scheduler
                if self.schedulers[model_name] is not None:
                    self.schedulers[model_name].step()
                
                # Early stopping check
                if self.config.early_stopping and self.patience_counter[model_name] >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping {model_name} model at epoch {epoch}")
                    break
        
        # Save final metrics
        self.metrics.save_metrics(self.save_dir / "training_metrics.json")
        
        # Save final models
        for model_name in self.models.keys():
            self.save_checkpoint(model_name, epoch, is_best=False)
        
        logger.info("Training completed!")
    
    def plot_training_curves(self):
        """Plot training curves for comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        for model_name in self.models.keys():
            history = self.metrics.metrics_history[model_name]
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Training loss
            axes[0, 0].plot(epochs, history['train_loss'], label=f'{model_name} train')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            
            # Validation loss
            axes[0, 1].plot(epochs, history['val_loss'], label=f'{model_name} val')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            
            # Perplexity
            axes[1, 0].plot(epochs, history['perplexity'], label=f'{model_name}')
            axes[1, 0].set_title('Perplexity')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Perplexity')
            axes[1, 0].legend()
        
        # Parameter comparison
        if len(self.models) > 1:
            model_names = list(self.models.keys())
            param_counts = [sum(p.numel() for p in model.parameters() if p.requires_grad) 
                          for model in self.models.values()]
            
            axes[1, 1].bar(model_names, param_counts)
            axes[1, 1].set_title('Parameter Count Comparison')
            axes[1, 1].set_ylabel('Parameters')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Residual Transformer')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model_type', type=str, choices=['residual', 'baseline', 'both'], 
                       default='both', help='Which model(s) to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum epochs')
    parser.add_argument('--fixed_kv_length', type=int, default=32, help='Fixed KV length for residual model')
    parser.add_argument('--n_processing_blocks', type=int, default=6, help='Number of processing blocks')
    
    args = parser.parse_args()
    
    # Create config
    config = ResidualTransformerConfig(
        model_type=args.model_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        fixed_kv_length=args.fixed_kv_length,
        n_processing_blocks=args.n_processing_blocks
    )
    
    config.print_config()
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, tokenizer = get_data_loaders(
        dataset_name=config.dataset_name,
        train_split=config.train_split,
        val_split=config.val_split,
        max_length=config.max_len,
        batch_size=config.batch_size,
        max_train_samples=10000,  # Start small for testing
        max_val_samples=1000,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Create trainer and train
    trainer = ModelTrainer(config)
    trainer.train(train_loader, val_loader, tokenizer)
    
    # Plot results
    trainer.plot_training_curves()
    
    # Print final comparison
    if len(trainer.models) > 1:
        logger.info("\nFinal Model Comparison:")
        for model_name in trainer.models.keys():
            metrics = trainer.metrics.get_latest_metrics(model_name)
            param_count = sum(p.numel() for p in trainer.models[model_name].parameters() if p.requires_grad)
            logger.info(f"{model_name}:")
            logger.info(f"  Parameters: {param_count:,}")
            logger.info(f"  Best Val Loss: {trainer.best_val_loss[model_name]:.4f}")
            logger.info(f"  Final Perplexity: {metrics['perplexity']:.2f}")

if __name__ == "__main__":
    main()