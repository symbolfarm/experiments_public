import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Fix tokenizer multiprocessing warnings

import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.amp import autocast, GradScaler

import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List
import logging
from dataclasses import asdict

from symbolfarm.design._01_cascade_correlation.config import ConfigCascadeCorrelation, ConfigTrain
from symbolfarm.design._01_cascade_correlation.model import CascadeTransformer
from symbolfarm.module.model import LanguageModelingLoss
from symbolfarm.monitor.growth import GrowthMonitor
from symbolfarm.realm.tiny_stories import create_dataloaders

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CascadeTrainer:
    """Trainer for cascade-correlation transformer."""
    
    def __init__(self, config_model: ConfigCascadeCorrelation, config_train: ConfigTrain):
        self.config_model = config_model
        self.config_train = config_train
        self.device = torch.device(config_train.device)
        
        # Initialize model and move to device
        self.model = CascadeTransformer(config_model).to(self.device)
        
        # Setup data
        self.train_loader, self.val_loader, self.data_module = create_dataloaders(config_train)
        
        # Setup loss function
        self.criterion = LanguageModelingLoss(
            config_model.vocab_size, config_train.label_smoothing
        ).to(self.device)
        
        # Setup growth monitor
        self.growth_monitor = GrowthMonitor(
            patience=config_train.patience,
            threshold=config_train.growth_threshold
        )
        
        # Setup mixed precision if enabled
        self.scaler = GradScaler('cuda') if config_train.mixed_precision else None
        
        # Setup optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self._setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        self.growth_events = []
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config_train.save_dir) / config_train.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup plotting
        self.plot_dir = self.checkpoint_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True)
        
        logger.info(f"Trainer initialized with {self.model.count_parameters():,} parameters")
        logger.info(f"Training on {len(self.train_loader)} batches, validating on {len(self.val_loader)} batches")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Get trainable parameters (important for frozen blocks)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.config_train.optimizer.lower() == 'adamw':
            self.optimizer = AdamW(
                trainable_params,
                lr=self.config_train.learning_rate,
                weight_decay=self.config_train.weight_decay,
                betas=(self.config_train.beta1, self.config_train.beta2),
                eps=self.config_train.eps
            )
        elif self.config_train.optimizer.lower() == 'sgd':
            self.optimizer = SGD(
                trainable_params,
                lr=self.config_train.learning_rate,
                weight_decay=self.config_train.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config_train.optimizer}")
        
        # Setup scheduler
        if self.config_train.scheduler.lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.config_train.max_epochs
            )
        elif self.config_train.scheduler.lower() == 'linear':
            self.scheduler = LinearLR(
                self.optimizer, start_factor=0.1, total_iters=self.config_train.warmup_steps
            )
    
    def _refresh_optimizer_after_growth(self):
        """Refresh optimizer after adding new blocks."""
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Create new optimizer with current parameters
        self._setup_optimizer()
        
        # Restore learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        logger.info(f"Refreshed optimizer with {len(list(self.model.parameters()))} parameters")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.config_train.mixed_precision):
                logits = self.model(input_ids)
                loss = self.criterion(logits, targets)
                
                # Add L2 regularization for new blocks if configured
                if self.config_train.new_block_l2 > 0 and len(self.model.blocks) > 0:
                    # Only regularize the last (newest) block
                    last_block = self.model.blocks[-1]
                    l2_loss = sum(p.pow(2.0).sum() for p in last_block.parameters())
                    loss = loss + self.config_train.new_block_l2 * l2_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config_train.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config_train.gradient_clip)
                self.optimizer.step()
            
            # Update learning rate (after optimizer step)
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if self.global_step % self.config_train.log_interval == 0:
                avg_loss = total_loss / num_batches
                logger.info(f"Epoch {self.epoch}, Step {self.global_step}: Loss = {avg_loss:.4f}")
            
            # Evaluation
            if self.global_step % self.config_train.eval_interval == 0:
                val_metrics = self.validate()
                logger.info(f"Validation: Loss = {val_metrics['loss']:.4f}, PPL = {val_metrics['perplexity']:.2f}")
            
            # Save checkpoint
            if self.global_step % self.config_train.save_interval == 0:
                self.save_checkpoint()
        
        return {'loss': total_loss / num_batches}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                with autocast('cuda', enabled=self.config_train.mixed_precision):
                    logits = self.model(input_ids)
                    loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity
        }
    
    def generate_samples(self, num_samples: int = 3) -> List[str]:
        """Generate sample text for evaluation."""
        self.model.eval()
        samples = []
        
        # Get sample prompts
        prompts = self.data_module.get_sample_text(num_samples)
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                tokens = self.data_module.tokenizer.encode(prompt, return_tensors='pt')
                tokens = tokens[:, :50].to(self.device)  # Limit prompt length
                
                # Generate
                generated = self.model.generate(
                    tokens, 
                    max_length=self.config_train.max_generate_length,
                    temperature=0.8,
                    top_k=50
                )
                
                # Decode
                text = self.data_module.decode_tokens(generated[0])
                samples.append(text)
        
        return samples
    
    def grow_network(self) -> bool:
        """Add a new transformer block to the network."""
        # Check if we can grow
        if len(self.model.blocks) >= self.config_model.max_blocks:
            logger.info("Maximum blocks reached, cannot grow further")
            return False
        
        # Check parameter budget
        current_params = self.model.count_parameters()
        if current_params > self.config_model.warn_at_parameters:
            logger.warning(f"Approaching parameter limit: {current_params:,}/{self.config_model.max_parameters:,}")
        
        if current_params > self.config_model.max_parameters:
            logger.info("Parameter budget exceeded, cannot grow further")
            return False
        
        # Add new block
        success = self.model.add_block()
        
        if success:
            # Freeze previous blocks if configured
            self.model.freeze_blocks(except_last=True)
            
            # Refresh optimizer for new parameters
            self._refresh_optimizer_after_growth()
            
            # Record growth event
            self.growth_events.append({
                'epoch': self.epoch,
                'global_step': self.global_step,
                'num_blocks': len(self.model.blocks),
                'total_params': self.model.count_parameters(),
                'trainable_params': self.model.count_parameters(trainable_only=True)
            })
            
            logger.info(f"Network grown to {len(self.model.blocks)} blocks")
            logger.info(f"Total parameters: {self.model.count_parameters():,}")
            logger.info(f"Trainable parameters: {self.model.count_parameters(trainable_only=True):,}")
        
        return success
    
    def train(self):
        """Main training loop with cascade growth."""
        logger.info("Starting cascade-correlation training...")
        logger.info(f"Initial model: {len(self.model.blocks)} blocks, {self.model.count_parameters():,} parameters")
        
        for epoch in range(self.config_train.max_epochs):
            self.epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update training history
            history_entry = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_perplexity': val_metrics['perplexity'],
                'num_blocks': len(self.model.blocks),
                'total_params': self.model.count_parameters(),
                'trainable_params': self.model.count_parameters(trainable_only=True),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(history_entry)
            
            # Log epoch summary
            logger.info(f"Epoch {epoch} complete:")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"  Val Perplexity: {val_metrics['perplexity']:.2f}")
            logger.info(f"  Blocks: {len(self.model.blocks)}")
            logger.info(f"  Parameters: {self.model.count_parameters():,}")
            
            # Generate samples periodically
            if epoch % 5 == 0:
                samples = self.generate_samples(2)
                logger.info("Sample generations:")
                for i, sample in enumerate(samples):
                    logger.info(f"  {i+1}: {sample[:100]}...")
            
            # Check for growth
            should_grow = self.growth_monitor.update(val_metrics['loss'])
            
            if should_grow:
                logger.info("Growth conditions met, adding new block...")
                growth_success = self.grow_network()
                
                if growth_success:
                    # Plot growth event
                    if self.config_train.plot_growth:
                        self.plot_training_progress()
                else:
                    logger.info("Growth failed or not possible")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(is_best=True)
            
            # Early stopping check
            if self.config_train.early_stopping:
                if self._should_stop_early():
                    logger.info("Early stopping triggered")
                    break
        
        # Final evaluation and plotting
        logger.info("Training completed!")
        final_samples = self.generate_samples(5)
        logger.info("Final sample generations:")
        for i, sample in enumerate(final_samples):
            logger.info(f"  {i+1}: {sample}")
        
        # Save final plots
        if self.config_train.plot_growth:
            self.plot_training_progress()
            self.plot_growth_timeline()
        
        # Save final checkpoint
        self.save_checkpoint(is_final=True)
    
    def _should_stop_early(self) -> bool:
        """Check if early stopping conditions are met."""
        if len(self.training_history) < self.config_train.early_stopping_patience:
            return False
        
        recent_losses = [h['val_loss'] for h in self.training_history[-self.config_train.early_stopping_patience:]]
        best_recent = min(recent_losses)
        current_loss = self.training_history[-1]['val_loss']
        
        return (current_loss - best_recent) < self.config_train.early_stopping_min_delta
    
    def plot_training_progress(self):
        """Plot training progress with growth events."""
        if not self.training_history:
            return
        
        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h['train_loss'] for h in self.training_history]
        val_losses = [h['val_loss'] for h in self.training_history]
        num_blocks = [h['num_blocks'] for h in self.training_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot losses
        ax1.plot(epochs, train_losses, label='Train Loss', alpha=0.7)
        ax1.plot(epochs, val_losses, label='Val Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mark growth events
        for event in self.growth_events:
            ax1.axvline(x=event['epoch'], color='red', linestyle='--', alpha=0.5)
            ax1.text(event['epoch'], ax1.get_ylim()[1] * 0.9, 
                    f"Block {event['num_blocks']}", rotation=90, ha='right')
        
        # Plot number of blocks
        ax2.plot(epochs, num_blocks, marker='o', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Number of Blocks')
        ax2.set_title('Network Growth')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_growth_timeline(self):
        """Plot detailed growth timeline."""
        if not self.growth_events:
            return
        
        events = self.growth_events
        blocks = [e['num_blocks'] for e in events]
        params = [e['total_params'] for e in events]
        epochs = [e['epoch'] for e in events]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Blocks over time
        ax1.step(epochs, blocks, where='post', marker='o', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Number of Blocks')
        ax1.set_title('Network Growth Timeline')
        ax1.grid(True, alpha=0.3)
        
        # Parameters over time
        ax2.step(epochs, params, where='post', marker='s', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Total Parameters')
        ax2.set_title('Parameter Count Growth')
        ax2.grid(True, alpha=0.3)
        
        # Format y-axis for parameters
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'growth_timeline.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config_model': asdict(self.config_model),
            'config_train': asdict(self.config_train),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'growth_events': self.growth_events,
            'model_info': self.model.get_info()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
        
        # Save final model
        if is_final:
            final_path = self.checkpoint_dir / 'final_model.pt'
            torch.save(checkpoint, final_path)
            
            # Save training history as JSON
            history_path = self.checkpoint_dir / 'training_history.json'
            model_info = self.model.get_info()
            # Convert config to dict for JSON serialization
            model_info_serializable = dict(model_info)
            model_info_serializable['config_model'] = asdict(model_info['config_model'])
            
            with open(history_path, 'w') as f:
                json.dump({
                    'training_history': self.training_history,
                    'growth_events': self.growth_events,
                    'final_model_info': model_info_serializable
                }, f, indent=2)
            
            logger.info("Saved final model and training history")

def main():
    """Main training function."""
    # Load configuration
    config_model = ConfigCascadeCorrelation()
    config_train = ConfigTrain()
    config_model.print_config()
    config_train.print_config()
    
    # Create trainer
    trainer = CascadeTrainer(config_model,config_train)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()