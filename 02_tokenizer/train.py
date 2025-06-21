import os
import time
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import asdict

from config import GrowingBPEConfig, ExperimentConfigs
from tokenizer import GrowingBPETokenizer
from data import create_data_loader
from evaluate import TokenizerEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingMetrics:
    """Track training metrics and growth events."""
    
    def __init__(self):
        self.metrics_history = []
        self.growth_events = []
        self.start_time = time.time()
        
    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """Log metrics for a training step."""
        metrics['step'] = step
        metrics['timestamp'] = time.time() - self.start_time
        self.metrics_history.append(metrics)
        
    def log_growth_event(self, event_data: Dict[str, Any]):
        """Log a vocabulary growth event."""
        event_data['timestamp'] = time.time() - self.start_time
        self.growth_events.append(event_data)
        
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def save_to_file(self, filepath: str):
        """Save metrics to JSON file."""
        data = {
            'metrics_history': self.metrics_history,
            'growth_events': self.growth_events,
            'total_time': time.time() - self.start_time
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

class GrowingBPETrainer:
    """Trainer for growing BPE tokenizer."""
    
    def __init__(self, config: GrowingBPEConfig):
        self.config = config
        self.tokenizer = GrowingBPETokenizer(config)
        self.data_loader = create_data_loader(config)
        self.evaluator = TokenizerEvaluator(self.tokenizer)
        self.metrics = TrainingMetrics()
        
        # Training state
        self.tokens_processed = 0
        self.batches_processed = 0
        self.last_checkpoint_step = 0
        self.last_validation_step = 0
        
        # Create save directory
        self.save_dir = Path(config.get_save_path())
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging to file
        log_file = self.save_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Initialized trainer for run: {config.run_name}")
        
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Configuration: {self.config.run_name}")
        
        # Load dataset
        logger.info("Loading dataset...")
        self.data_loader.load_dataset()
        self.data_loader.print_dataset_info()
        
        # Save initial state
        self._save_checkpoint("initial")
        
        # Training loop
        try:
            for batch in self.data_loader.get_text_stream():
                # Process batch
                self._process_batch(batch)
                
                # Check stopping conditions
                if self.tokens_processed >= self.config.max_tokens_to_process:
                    logger.info(f"Reached maximum tokens limit: {self.config.max_tokens_to_process}")
                    break
                
                # Logging
                if self.tokens_processed - self.last_checkpoint_step >= self.config.log_interval:
                    self._log_progress()
                
                # Validation
                if (self.config.validation_interval > 0 and 
                    self.tokens_processed - self.last_validation_step >= self.config.validation_interval):
                    self._run_validation()
                    self.last_validation_step = self.tokens_processed
                
                # Checkpointing
                if (self.config.checkpoint_interval > 0 and
                    self.tokens_processed - self.last_checkpoint_step >= self.config.checkpoint_interval):
                    self._save_checkpoint(f"step_{self.tokens_processed}")
                    self.last_checkpoint_step = self.tokens_processed
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Final checkpoint and evaluation
            self._save_final_results()
            
        logger.info("Training completed!")
        
    def _process_batch(self, batch: List[str]):
        """Process a batch of texts."""
        batch_start_time = time.time()
        batch_tokens = 0
        vocab_size_before = self.tokenizer.get_vocab_size()
        
        # Tokenize each text in the batch
        for text in batch:
            if not text.strip():
                continue
                
            tokens = self.tokenizer.tokenize(text)
            batch_tokens += len(tokens)
        
        # Update counters
        self.tokens_processed += batch_tokens
        self.batches_processed += 1
        
        # Check for growth events
        vocab_size_after = self.tokenizer.get_vocab_size()
        if vocab_size_after > vocab_size_before:
            growth_events = self.tokenizer.get_growth_history()
            if growth_events:
                latest_event = growth_events[-1]
                self.metrics.log_growth_event({
                    'step': self.tokens_processed,
                    'batch': self.batches_processed,
                    'vocab_size_before': vocab_size_before,
                    'vocab_size_after': vocab_size_after,
                    'tokens_added': latest_event.tokens_added,
                    'trigger_reason': latest_event.trigger_reason
                })
                
                logger.info(f"Vocabulary growth at step {self.tokens_processed}: " +
                          f"{vocab_size_before} -> {vocab_size_after} tokens")
        
        # Log batch metrics
        batch_time = time.time() - batch_start_time
        self.metrics.log_metrics(self.tokens_processed, {
            'batch': self.batches_processed,
            'batch_tokens': batch_tokens,
            'batch_time': batch_time,
            'tokens_per_second': batch_tokens / batch_time if batch_time > 0 else 0,
            'vocab_size': vocab_size_after,
            'compression_ratio': self._estimate_compression_ratio(batch)
        })
    
    def _estimate_compression_ratio(self, texts: List[str]) -> float:
        """Estimate compression ratio for a batch of texts."""
        if not texts:
            return 0.0
            
        total_chars = sum(len(text) for text in texts)
        if total_chars == 0:
            return 0.0
            
        # Sample a few texts to avoid recomputation
        sample_texts = texts[:min(5, len(texts))]
        sample_chars = sum(len(text) for text in sample_texts)
        sample_tokens = sum(len(self.tokenizer._tokenize_text(text)) for text in sample_texts)
        
        return sample_tokens / sample_chars if sample_chars > 0 else 0.0
    
    def _log_progress(self):
        """Log current training progress."""
        latest_metrics = self.metrics.get_latest_metrics()
        if not latest_metrics:
            return
            
        logger.info(f"Step {self.tokens_processed}: " +
                   f"Processed {self.tokens_processed} tokens, " +
                   f"Vocab size: {latest_metrics['vocab_size']}, " +
                   f"Compression: {latest_metrics['compression_ratio']:.3f}")
        
        # Log growth events
        growth_events = len(self.tokenizer.get_growth_history())
        if growth_events > 0:
            logger.info(f"Growth events: {growth_events}")
    
    def _run_validation(self):
        """Run validation evaluation."""
        logger.info("Running validation...")
        
        val_texts = self.data_loader.get_validation_texts(max_samples=100)
        if not val_texts:
            logger.warning("No validation texts available")
            return
        
        # Run evaluation
        compression_metrics = self.evaluator.evaluate_compression(val_texts)
        utilization_metrics = self.evaluator.evaluate_vocabulary_utilization(val_texts)
        quality_metrics = self.evaluator.evaluate_subword_quality(val_texts)
        
        # Log validation results
        logger.info(f"Validation - Compression ratio: {compression_metrics['compression_ratio']:.3f}")
        logger.info(f"Validation - Vocab utilization: {utilization_metrics['vocabulary_utilization']:.3f}")
        logger.info(f"Validation - Avg token length: {quality_metrics['avg_token_length']:.2f}")
        
        # Save validation metrics
        validation_metrics = {
            'step': self.tokens_processed,
            'compression': compression_metrics,
            'utilization': utilization_metrics,
            'quality': quality_metrics
        }
        
        val_file = self.save_dir / "validation_metrics.json"
        
        # Load existing validation data if it exists
        validation_history = []
        if val_file.exists():
            try:
                with open(val_file, 'r') as f:
                    validation_history = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing validation data: {e}")
        
        validation_history.append(validation_metrics)
        
        with open(val_file, 'w') as f:
            json.dump(validation_history, f, indent=2)
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save training checkpoint."""
        checkpoint_dir = self.save_dir / "checkpoints" / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        tokenizer_file = checkpoint_dir / "tokenizer.json"
        self.tokenizer.save(str(tokenizer_file))
        
        # Save training metrics
        metrics_file = checkpoint_dir / "metrics.json"
        self.metrics.save_to_file(str(metrics_file))
        
        # Save config
        config_file = checkpoint_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save training state
        state_file = checkpoint_dir / "training_state.json"
        training_state = {
            'tokens_processed': self.tokens_processed,
            'batches_processed': self.batches_processed,
            'last_checkpoint_step': self.last_checkpoint_step,
            'last_validation_step': self.last_validation_step
        }
        with open(state_file, 'w') as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"Saved checkpoint: {checkpoint_name}")
    
    def _save_final_results(self):
        """Save final training results and generate plots."""
        logger.info("Saving final results...")
        
        # Save final tokenizer
        final_tokenizer_file = self.save_dir / "final_tokenizer.json"
        self.tokenizer.save(str(final_tokenizer_file))
        
        # Save final metrics
        final_metrics_file = self.save_dir / "final_metrics.json"
        self.metrics.save_to_file(str(final_metrics_file))
        
        # Generate training plots
        if self.config.plot_growth:
            self._generate_plots()
        
        # Final evaluation
        logger.info("Running final evaluation...")
        val_texts = self.data_loader.get_validation_texts(max_samples=500)
        if val_texts:
            self._run_final_evaluation(val_texts)
        
        # Save tokenizer statistics
        stats_file = self.save_dir / "tokenizer_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.tokenizer.get_statistics(), f, indent=2)
        
        logger.info(f"Results saved to: {self.save_dir}")
    
    def _generate_plots(self):
        """Generate training visualization plots."""
        try:
            plots_dir = self.save_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Plot 1: Vocabulary growth over time
            if self.metrics.metrics_history:
                steps = [m['step'] for m in self.metrics.metrics_history]
                vocab_sizes = [m['vocab_size'] for m in self.metrics.metrics_history]
                
                plt.figure(figsize=(10, 6))
                plt.plot(steps, vocab_sizes, 'b-', linewidth=2)
                plt.xlabel('Tokens Processed')
                plt.ylabel('Vocabulary Size')
                plt.title('Vocabulary Growth Over Time')
                plt.grid(True, alpha=0.3)
                
                # Mark growth events
                for event in self.metrics.growth_events:
                    plt.axvline(x=event['step'], color='red', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(plots_dir / "vocabulary_growth.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            # Plot 2: Compression ratio over time
            if self.metrics.metrics_history:
                compression_ratios = [m.get('compression_ratio', 0) for m in self.metrics.metrics_history]
                
                plt.figure(figsize=(10, 6))
                plt.plot(steps, compression_ratios, 'g-', linewidth=2)
                plt.xlabel('Tokens Processed')
                plt.ylabel('Compression Ratio (tokens/char)')
                plt.title('Compression Efficiency Over Time')
                plt.grid(True, alpha=0.3)
                
                # Mark growth events
                for event in self.metrics.growth_events:
                    plt.axvline(x=event['step'], color='red', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(plots_dir / "compression_ratio.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            # Plot 3: Growth events timeline
            if self.metrics.growth_events:
                event_steps = [e['step'] for e in self.metrics.growth_events]
                vocab_sizes = [e['vocab_size_after'] for e in self.metrics.growth_events]
                
                plt.figure(figsize=(12, 6))
                plt.scatter(event_steps, vocab_sizes, c='red', s=50, alpha=0.7)
                plt.xlabel('Tokens Processed')
                plt.ylabel('Vocabulary Size After Growth')
                plt.title('Growth Events Timeline')
                plt.grid(True, alpha=0.3)
                
                # Add text annotations for major growth events
                for i, event in enumerate(self.metrics.growth_events[::max(1, len(self.metrics.growth_events)//10)]):
                    plt.annotate(f"+{len(event.get('tokens_added', []))}", 
                               (event['step'], event['vocab_size_after']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                plt.tight_layout()
                plt.savefig(plots_dir / "growth_events.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Generated plots in: {plots_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
    
    def _run_final_evaluation(self, val_texts: List[str]):
        """Run comprehensive final evaluation."""
        try:
            # Comprehensive evaluation
            compression_metrics = self.evaluator.evaluate_compression(val_texts)
            utilization_metrics = self.evaluator.evaluate_vocabulary_utilization(val_texts)
            quality_metrics = self.evaluator.evaluate_subword_quality(val_texts)
            growth_analysis = self.evaluator.analyze_growth_patterns()
            
            # Combine all metrics
            final_evaluation = {
                'compression': compression_metrics,
                'utilization': utilization_metrics,
                'quality': quality_metrics,
                'growth_analysis': growth_analysis,
                'training_summary': {
                    'total_tokens_processed': self.tokens_processed,
                    'total_batches_processed': self.batches_processed,
                    'final_vocab_size': self.tokenizer.get_vocab_size(),
                    'growth_events': len(self.tokenizer.get_growth_history()),
                    'training_time': time.time() - self.metrics.start_time
                }
            }
            
            # Save final evaluation
            eval_file = self.save_dir / "final_evaluation.json"
            with open(eval_file, 'w') as f:
                json.dump(final_evaluation, f, indent=2)
            
            # Print summary
            logger.info("=== Final Evaluation Summary ===")
            logger.info(f"Tokens processed: {self.tokens_processed:,}")
            logger.info(f"Final vocabulary size: {self.tokenizer.get_vocab_size()}")
            logger.info(f"Growth events: {len(self.tokenizer.get_growth_history())}")
            logger.info(f"Compression ratio: {compression_metrics['compression_ratio']:.3f}")
            logger.info(f"Vocabulary utilization: {utilization_metrics['vocabulary_utilization']:.3f}")
            logger.info(f"Average token length: {quality_metrics['avg_token_length']:.2f}")
            logger.info("=" * 32)
            
        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train a growing BPE tokenizer")
    parser.add_argument("--config", type=str, default="default",
                       help="Configuration to use (default, aggressive, conservative, hybrid, minimal)")
    parser.add_argument("--max-tokens", type=int, help="Override max tokens to process")
    parser.add_argument("--max-samples", type=int, help="Limit dataset samples for testing")
    parser.add_argument("--run-name", type=str, help="Override run name")
    parser.add_argument("--save-dir", type=str, help="Override save directory")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == "default":
        config = GrowingBPEConfig()
    elif args.config == "aggressive":
        config = ExperimentConfigs.aggressive_growth()
    elif args.config == "conservative":
        config = ExperimentConfigs.conservative_growth()
    elif args.config == "hybrid":
        config = ExperimentConfigs.hybrid_strategy()
    elif args.config == "minimal":
        config = ExperimentConfigs.minimal_experiment()
    else:
        # Try to load as file path
        if args.config.endswith('.py'):
            # Load config from Python file
            import importlib.util
            spec = importlib.util.spec_from_file_location("config_module", args.config)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            config = config_module.config
        else:
            raise ValueError(f"Unknown config: {args.config}")
    
    # Apply overrides
    if args.max_tokens:
        config.max_tokens_to_process = args.max_tokens
    if args.max_samples:
        config.max_samples = args.max_samples
    if args.run_name:
        config.run_name = args.run_name
    if args.save_dir:
        config.save_dir = args.save_dir
    
    # Print configuration
    config.print_config()
    
    # Create trainer and run
    trainer = GrowingBPETrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()