import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import logging

from config import ResidualTransformerConfig
from model import ResidualTransformer
from baseline_model import BaselineTransformer
from data import get_data_loaders, decode_batch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive evaluation of residual vs baseline transformer."""
    
    def __init__(self, config: ResidualTransformerConfig):
        self.config = config
        self.device = torch.device(config.device)
        
    def load_model(self, model_type: str, checkpoint_path: str) -> torch.nn.Module:
        """Load a trained model from checkpoint."""
        if model_type == 'residual':
            model = ResidualTransformer(self.config)
        elif model_type == 'baseline':
            model = BaselineTransformer(self.config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded {model_type} model from {checkpoint_path}")
        return model
    
    @torch.no_grad()
    def evaluate_perplexity(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate perplexity on a dataset."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits, loss = model(input_ids, labels)
            
            # Calculate number of non-padding tokens
            mask = (labels != -100).float()
            num_tokens = mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens,
            'num_batches': num_batches
        }
    
    @torch.no_grad()
    def evaluate_generation_quality(self, model: torch.nn.Module, tokenizer, 
                                  prompts: List[str], max_length: int = 100) -> Dict[str, List[str]]:
        """Evaluate text generation quality."""
        model.eval()
        
        results = {
            'prompts': prompts,
            'generations': [],
            'lengths': []
        }
        
        for prompt in prompts:
            # Encode prompt
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Generate
            generated = model.generate(input_ids, max_length=max_length)
            
            # Decode and store
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            results['generations'].append(generated_text)
            results['lengths'].append(len(generated[0]))
        
        return results
    
    @torch.no_grad()
    def analyze_attention_patterns(self, model: torch.nn.Module, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze attention patterns (for residual model)."""
        if not isinstance(model, ResidualTransformer):
            logger.warning("Attention analysis only available for ResidualTransformer")
            return {}
        
        model.eval()
        
        # Hook to capture attention weights
        attention_weights = []
        
        def attention_hook(module, input, output):
            if hasattr(module, 'attention') and hasattr(module.attention, 'dropout'):
                # This is a processing block - we'd need to modify the model to expose attention weights
                pass
        
        # For now, just return basic info
        # TODO: Modify model to expose attention weights if needed for analysis
        return {'message': 'Attention analysis requires model modification to expose weights'}
    
    def compare_models(self, residual_checkpoint: str, baseline_checkpoint: str, 
                      test_loader: DataLoader, tokenizer) -> Dict[str, any]:
        """Comprehensive comparison between residual and baseline models."""
        
        results = {}
        
        # Load models
        residual_model = self.load_model('residual', residual_checkpoint)
        baseline_model = self.load_model('baseline', baseline_checkpoint)
        
        # Count parameters
        residual_params = sum(p.numel() for p in residual_model.parameters() if p.requires_grad)
        baseline_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
        
        results['parameter_count'] = {
            'residual': residual_params,
            'baseline': baseline_params,
            'ratio': residual_params / baseline_params
        }
        
        # Evaluate perplexity
        logger.info("Evaluating perplexity...")
        residual_metrics = self.evaluate_perplexity(residual_model, test_loader)
        baseline_metrics = self.evaluate_perplexity(baseline_model, test_loader)
        
        results['perplexity'] = {
            'residual': residual_metrics,
            'baseline': baseline_metrics,
            'improvement': (baseline_metrics['perplexity'] - residual_metrics['perplexity']) / baseline_metrics['perplexity']
        }
        
        # Test generation quality
        logger.info("Testing generation quality...")
        test_prompts = [
            "Once upon a time",
            "The little girl",
            "In a magical forest",
            "The brave knight",
            "There was a dragon"
        ]
        
        residual_generations = self.evaluate_generation_quality(residual_model, tokenizer, test_prompts)
        baseline_generations = self.evaluate_generation_quality(baseline_model, tokenizer, test_prompts)
        
        results['generation'] = {
            'residual': residual_generations,
            'baseline': baseline_generations
        }
        
        # Memory and speed analysis
        logger.info("Analyzing computational efficiency...")
        results['efficiency'] = self.analyze_efficiency(residual_model, baseline_model, test_loader)
        
        return results
    
    @torch.no_grad()
    def analyze_efficiency(self, residual_model: torch.nn.Module, baseline_model: torch.nn.Module, 
                          test_loader: DataLoader) -> Dict[str, float]:
        """Analyze memory usage and inference speed."""
        
        # Get a test batch
        test_batch = next(iter(test_loader))
        input_ids = test_batch['input_ids'][:4].to(self.device)  # Small batch for testing
        
        results = {}
        
        # Memory usage
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test residual model
        residual_model.eval()
        _ = residual_model(input_ids)
        residual_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test baseline model
        baseline_model.eval()
        _ = baseline_model(input_ids)
        baseline_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        results['memory_usage'] = {
            'residual_mb': residual_memory / 1024 / 1024,
            'baseline_mb': baseline_memory / 1024 / 1024,
            'ratio': residual_memory / baseline_memory if baseline_memory > 0 else 0
        }
        
        # Inference speed
        import time
        
        # Warm up
        for _ in range(5):
            _ = residual_model(input_ids)
            _ = baseline_model(input_ids)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Time residual model
        start_time = time.time()
        for _ in range(50):
            _ = residual_model(input_ids)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        residual_time = (time.time() - start_time) / 50
        
        # Time baseline model
        start_time = time.time()
        for _ in range(50):
            _ = baseline_model(input_ids)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        baseline_time = (time.time() - start_time) / 50
        
        results['inference_speed'] = {
            'residual_ms': residual_time * 1000,
            'baseline_ms': baseline_time * 1000,
            'speedup': baseline_time / residual_time
        }
        
        return results
    
    def plot_comparison(self, results: Dict, save_dir: str):
        """Create comparison plots."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Parameter count comparison
        models = ['Residual', 'Baseline']
        param_counts = [results['parameter_count']['residual'], results['parameter_count']['baseline']]
        
        axes[0, 0].bar(models, param_counts)
        axes[0, 0].set_title('Parameter Count')
        axes[0, 0].set_ylabel('Parameters')
        
        # Perplexity comparison
        perplexities = [results['perplexity']['residual']['perplexity'], 
                       results['perplexity']['baseline']['perplexity']]
        
        axes[0, 1].bar(models, perplexities)
        axes[0, 1].set_title('Perplexity (lower is better)')
        axes[0, 1].set_ylabel('Perplexity')
        
        # Memory usage
        memory_usage = [results['efficiency']['memory_usage']['residual_mb'],
                       results['efficiency']['memory_usage']['baseline_mb']]
        
        axes[0, 2].bar(models, memory_usage)
        axes[0, 2].set_title('Memory Usage')
        axes[0, 2].set_ylabel('Memory (MB)')
        
        # Inference speed
        inference_times = [results['efficiency']['inference_speed']['residual_ms'],
                          results['efficiency']['inference_speed']['baseline_ms']]
        
        axes[1, 0].bar(models, inference_times)
        axes[1, 0].set_title('Inference Time')
        axes[1, 0].set_ylabel('Time (ms)')
        
        # Generation length comparison
        residual_lengths = results['generation']['residual']['lengths']
        baseline_lengths = results['generation']['baseline']['lengths']
        
        axes[1, 1].boxplot([residual_lengths, baseline_lengths], labels=models)
        axes[1, 1].set_title('Generated Text Length')
        axes[1, 1].set_ylabel('Length (tokens)')
        
        # Efficiency scatter plot
        param_ratio = results['parameter_count']['ratio']
        perplexity_improvement = results['perplexity']['improvement']
        
        axes[1, 2].scatter([param_ratio], [perplexity_improvement], s=100, c='red', alpha=0.7)
        axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 2].axvline(x=1, color='k', linestyle='--', alpha=0.5)
        axes[1, 2].set_xlabel('Parameter Ratio (Residual/Baseline)')
        axes[1, 2].set_ylabel('Perplexity Improvement')
        axes[1, 2].set_title('Efficiency Trade-off')
        
        plt.tight_layout()
        plt.savefig(save_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save generation examples
        self.save_generation_examples(results, save_path)
    
    def save_generation_examples(self, results: Dict, save_path: Path):
        """Save generation examples to text file."""
        with open(save_path / 'generation_examples.txt', 'w') as f:
            prompts = results['generation']['residual']['prompts']
            residual_gens = results['generation']['residual']['generations']
            baseline_gens = results['generation']['baseline']['generations']
            
            f.write("=== Generation Quality Comparison ===\n\n")
            
            for i, prompt in enumerate(prompts):
                f.write(f"Prompt: {prompt}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Residual Model:\n{residual_gens[i]}\n\n")
                f.write(f"Baseline Model:\n{baseline_gens[i]}\n\n")
                f.write("=" * 80 + "\n\n")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Residual Transformer')
    parser.add_argument('--residual_checkpoint', type=str, required=True, 
                       help='Path to residual model checkpoint')
    parser.add_argument('--baseline_checkpoint', type=str, required=True,
                       help='Path to baseline model checkpoint')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--test_samples', type=int, default=1000,
                       help='Number of test samples to use')
    
    args = parser.parse_args()
    
    # Create config (should match training config)
    config = ResidualTransformerConfig()
    
    # Load test data
    logger.info("Loading test data...")
    _, test_loader, tokenizer = get_data_loaders(
        max_train_samples=0,  # No training data needed
        max_val_samples=args.test_samples,
        batch_size=16,
        num_workers=0
    )
    
    # Create evaluator and run comparison
    evaluator = ModelEvaluator(config)
    results = evaluator.compare_models(
        args.residual_checkpoint, 
        args.baseline_checkpoint, 
        test_loader, 
        tokenizer
    )
    
    # Save results
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    with open(save_path / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create plots
    evaluator.plot_comparison(results, args.save_dir)
    
    # Print summary
    logger.info("\n=== Evaluation Summary ===")
    logger.info(f"Residual Model Parameters: {results['parameter_count']['residual']:,}")
    logger.info(f"Baseline Model Parameters: {results['parameter_count']['baseline']:,}")
    logger.info(f"Parameter Ratio: {results['parameter_count']['ratio']:.3f}")
    logger.info(f"Residual Perplexity: {results['perplexity']['residual']['perplexity']:.2f}")
    logger.info(f"Baseline Perplexity: {results['perplexity']['baseline']['perplexity']:.2f}")
    logger.info(f"Perplexity Improvement: {results['perplexity']['improvement']:.3f}")
    logger.info(f"Memory Ratio: {results['efficiency']['memory_usage']['ratio']:.3f}")
    logger.info(f"Speed Ratio: {results['efficiency']['inference_speed']['speedup']:.3f}")
    
    logger.info(f"\nDetailed results saved to: {save_path}")

if __name__ == "__main__":
    main()