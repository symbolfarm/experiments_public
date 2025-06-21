import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from pathlib import Path
import logging

from tokenizer import GrowingBPETokenizer
from config import GrowingBPEConfig
from data import create_data_loader, TextProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenizerEvaluator:
    """Comprehensive evaluation suite for growing BPE tokenizers."""
    
    def __init__(self, tokenizer: GrowingBPETokenizer):
        self.tokenizer = tokenizer
        
    def evaluate_compression(self, texts: List[str]) -> Dict[str, float]:
        """Evaluate compression efficiency."""
        if not texts:
            return {}
        
        total_chars = 0
        total_tokens = 0
        token_lengths = []
        
        for text in texts:
            if not text:
                continue
            chars = len(text)
            tokens = self.tokenizer.tokenize(text)
            
            total_chars += chars
            total_tokens += len(tokens)
            token_lengths.extend([len(self.tokenizer.id_to_token.get(t, '')) for t in tokens])
        
        if total_chars == 0:
            return {}
        
        return {
            'compression_ratio': total_tokens / total_chars,
            'avg_tokens_per_text': total_tokens / len(texts),
            'avg_chars_per_token': total_chars / total_tokens if total_tokens > 0 else 0,
            'token_length_std': np.std(token_lengths) if token_lengths else 0,
            'token_length_mean': np.mean(token_lengths) if token_lengths else 0,
            'compression_efficiency': total_chars / total_tokens if total_tokens > 0 else 0  # chars per token
        }
    
    def evaluate_vocabulary_utilization(self, texts: List[str]) -> Dict[str, float]:
        """Evaluate how well the vocabulary is utilized."""
        token_counts = defaultdict(int)
        total_tokens = 0
        
        for text in texts:
            if not text:
                continue
            tokens = self.tokenizer.tokenize(text)
            total_tokens += len(tokens)
            for token in tokens:
                token_counts[token] += 1
        
        if total_tokens == 0:
            return {}
        
        # Calculate utilization metrics
        active_vocab = len([t for t, c in token_counts.items() if c > 0])
        total_vocab = self.tokenizer.get_vocab_size()
        
        # Token frequency distribution
        freq_counts = Counter(token_counts.values())
        singleton_tokens = freq_counts.get(1, 0)
        low_freq_tokens = sum(count for freq, count in freq_counts.items() if freq <= 5)
        
        # Entropy of token usage
        frequencies = np.array(list(token_counts.values()))
        probabilities = frequencies / frequencies.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return {
            'vocabulary_utilization': active_vocab / total_vocab if total_vocab > 0 else 0,
            'active_vocabulary_size': active_vocab,
            'total_vocabulary_size': total_vocab,
            'singleton_tokens': singleton_tokens,
            'low_frequency_tokens': low_freq_tokens,
            'unused_tokens': total_vocab - active_vocab,
            'token_entropy': entropy,
            'max_token_frequency': max(token_counts.values()) if token_counts else 0,
            'avg_token_frequency': np.mean(list(token_counts.values())) if token_counts else 0
        }
    
    def evaluate_subword_quality(self, texts: List[str]) -> Dict[str, float]:
        """Evaluate the quality of learned subwords."""
        vocab_tokens = list(self.tokenizer.id_to_token.values())
        
        if not vocab_tokens:
            return {}
        
        # Token length statistics
        token_lengths = [len(token) for token in vocab_tokens]
        avg_token_length = np.mean(token_lengths)
        token_length_std = np.std(token_lengths)
        
        # Character diversity
        all_chars = set(''.join(vocab_tokens))
        char_diversity = len(all_chars) / 256 if 256 > 0 else 0  # Normalize by ASCII range
        
        # Morphological pattern matching (simple heuristics)
        morphological_patterns = [
            'ing', 'ed', 'er', 'est', 'ly', 'tion', 'ness', 'ment',
            'un', 're', 'pre', 'dis', 'over', 'under', 'out', 'up'
        ]
        
        morphological_tokens = 0
        for token in vocab_tokens:
            if any(pattern in token.lower() for pattern in morphological_patterns):
                morphological_tokens += 1
        
        morphological_alignment = morphological_tokens / len(vocab_tokens)
        
        # Subword coherence (simple measures)
        # Ratio of alphabetic characters to total characters
        alpha_ratio = 0
        total_chars = 0
        alpha_chars = 0
        
        for token in vocab_tokens:
            total_chars += len(token)
            alpha_chars += sum(1 for c in token if c.isalpha())
        
        if total_chars > 0:
            alpha_ratio = alpha_chars / total_chars
        
        # Repeated character penalty
        repeated_char_tokens = 0
        for token in vocab_tokens:
            if len(set(token)) == 1 and len(token) > 1:  # All same character
                repeated_char_tokens += 1
        
        repeated_char_penalty = repeated_char_tokens / len(vocab_tokens)
        
        return {
            'avg_token_length': avg_token_length,
            'token_length_std': token_length_std,
            'character_diversity': char_diversity,
            'morphological_alignment': morphological_alignment,
            'alphabetic_ratio': alpha_ratio,
            'repeated_char_penalty': repeated_char_penalty,
            'subword_coherence_score': (morphological_alignment + alpha_ratio - repeated_char_penalty) / 2
        }
    
    def analyze_growth_patterns(self) -> Dict[str, Any]:
        """Analyze vocabulary growth patterns."""
        growth_history = self.tokenizer.get_growth_history()
        
        if not growth_history:
            return {'growth_events': 0}
        
        # Growth timing analysis
        growth_steps = [event.tokens_processed for event in growth_history]
        growth_intervals = [growth_steps[i] - growth_steps[i-1] for i in range(1, len(growth_steps))]
        
        # Vocabulary size progression
        vocab_sizes = [event.vocab_size_after for event in growth_history]
        growth_amounts = [vocab_sizes[i] - vocab_sizes[i-1] for i in range(1, len(vocab_sizes))]
        
        # Trigger analysis
        trigger_counts = defaultdict(int)
        for event in growth_history:
            trigger_counts[event.trigger_reason] += 1
        
        # Compression improvement analysis
        compression_improvements = []
        for event in growth_history:
            if hasattr(event, 'compression_ratio_before') and hasattr(event, 'compression_ratio_after'):
                improvement = event.compression_ratio_before - event.compression_ratio_after
                compression_improvements.append(improvement)
        
        return {
            'growth_events': len(growth_history),
            'avg_growth_interval': np.mean(growth_intervals) if growth_intervals else 0,
            'growth_interval_std': np.std(growth_intervals) if growth_intervals else 0,
            'avg_tokens_per_growth': np.mean(growth_amounts) if growth_amounts else 0,
            'total_vocab_growth': vocab_sizes[-1] - vocab_sizes[0] if vocab_sizes else 0,
            'trigger_distribution': dict(trigger_counts),
            'avg_compression_improvement': np.mean(compression_improvements) if compression_improvements else 0,
            'growth_efficiency': (vocab_sizes[-1] - vocab_sizes[0]) / len(growth_history) if growth_history else 0
        }
    
    def evaluate_domain_adaptation(self, texts: List[str]) -> Dict[str, float]:
        """Evaluate how well the tokenizer adapts to different text complexities."""
        if not texts:
            return {}
        
        # Categorize texts by complexity
        complexity_scores = [TextProcessor.get_text_complexity_score(text) for text in texts]
        
        # Divide into complexity quartiles
        quartiles = np.percentile(complexity_scores, [25, 50, 75])
        
        complexity_categories = {
            'simple': [],
            'medium': [],
            'complex': [],
            'very_complex': []
        }
        
        for i, score in enumerate(complexity_scores):
            if score <= quartiles[0]:
                complexity_categories['simple'].append(texts[i])
            elif score <= quartiles[1]:
                complexity_categories['medium'].append(texts[i])
            elif score <= quartiles[2]:
                complexity_categories['complex'].append(texts[i])
            else:
                complexity_categories['very_complex'].append(texts[i])
        
        # Evaluate tokenization quality for each category
        category_metrics = {}
        for category, category_texts in complexity_categories.items():
            if category_texts:
                compression = self.evaluate_compression(category_texts)
                category_metrics[category] = {
                    'compression_ratio': compression.get('compression_ratio', 0),
                    'avg_chars_per_token': compression.get('avg_chars_per_token', 0),
                    'text_count': len(category_texts)
                }
        
        return {
            'complexity_adaptation': category_metrics,
            'complexity_score_range': (min(complexity_scores), max(complexity_scores)),
            'avg_complexity': np.mean(complexity_scores)
        }
    
    def compare_with_baseline(self, texts: List[str], baseline_vocab_size: int = 1000) -> Dict[str, Any]:
        """Compare with a static BPE tokenizer of similar size."""
        # This is a simplified comparison - in practice you'd train a static BPE
        # For now, we'll simulate by creating a character-only baseline
        
        # Character-level baseline
        char_tokens = 0
        char_compression = 0
        
        for text in texts:
            if not text:
                continue
            char_tokens += len(text)  # Each character is a token
        
        total_chars = sum(len(text) for text in texts if text)
        char_compression = char_tokens / total_chars if total_chars > 0 else 0
        
        # Our tokenizer
        our_compression = self.evaluate_compression(texts)
        our_utilization = self.evaluate_vocabulary_utilization(texts)
        
        return {
            'baseline_compression_ratio': char_compression,
            'our_compression_ratio': our_compression.get('compression_ratio', 0),
            'compression_improvement': char_compression - our_compression.get('compression_ratio', 0),
            'vocab_size_comparison': {
                'baseline': baseline_vocab_size,
                'ours': self.tokenizer.get_vocab_size(),
                'efficiency_ratio': our_compression.get('compression_ratio', 0) / char_compression if char_compression > 0 else 0
            },
            'vocabulary_utilization': our_utilization.get('vocabulary_utilization', 0)
        }
    
    def generate_report(self, texts: List[str], output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        logger.info("Generating comprehensive evaluation report...")
        
        # Run all evaluations
        compression_metrics = self.evaluate_compression(texts)
        utilization_metrics = self.evaluate_vocabulary_utilization(texts)
        quality_metrics = self.evaluate_subword_quality(texts)
        growth_analysis = self.analyze_growth_patterns()
        domain_adaptation = self.evaluate_domain_adaptation(texts)
        baseline_comparison = self.compare_with_baseline(texts)
        
        # Tokenizer statistics
        tokenizer_stats = self.tokenizer.get_statistics()
        
        # Create comprehensive report
        report = {
            'evaluation_summary': {
                'texts_evaluated': len(texts),
                'total_characters': sum(len(text) for text in texts),
                'tokenizer_vocab_size': self.tokenizer.get_vocab_size(),
                'growth_events': len(self.tokenizer.get_growth_history())
            },
            'compression_metrics': compression_metrics,
            'vocabulary_utilization': utilization_metrics,
            'subword_quality': quality_metrics,
            'growth_analysis': growth_analysis,
            'domain_adaptation': domain_adaptation,
            'baseline_comparison': baseline_comparison,
            'tokenizer_statistics': tokenizer_stats
        }
        
        # Calculate overall scores
        overall_scores = self._calculate_overall_scores(report)
        report['overall_scores'] = overall_scores
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Evaluation report saved to: {output_file}")
        
        return report
    
    def _calculate_overall_scores(self, report: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall performance scores."""
        scores = {}
        
        # Compression efficiency score (lower is better, so invert)
        compression_ratio = report['compression_metrics'].get('compression_ratio', 1.0)
        scores['compression_score'] = max(0, 1.0 - compression_ratio) if compression_ratio > 0 else 0
        
        # Vocabulary efficiency score
        utilization = report['vocabulary_utilization'].get('vocabulary_utilization', 0)
        scores['vocabulary_score'] = utilization
        
        # Quality score
        quality = report['subword_quality'].get('subword_coherence_score', 0)
        scores['quality_score'] = max(0, quality)
        
        # Growth efficiency score
        growth_events = report['growth_analysis'].get('growth_events', 0)
        vocab_size = report['evaluation_summary'].get('tokenizer_vocab_size', 1)
        scores['growth_score'] = min(1.0, growth_events / (vocab_size / 100)) if vocab_size > 0 else 0
        
        # Overall score (weighted average)
        weights = {
            'compression_score': 0.3,
            'vocabulary_score': 0.3,
            'quality_score': 0.2,
            'growth_score': 0.2
        }
        
        scores['overall_score'] = sum(scores[key] * weights[key] for key in weights.keys())
        
        return scores
    
    def visualize_results(self, report: Dict[str, Any], output_dir: str):
        """Generate visualization plots for the evaluation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Plot 1: Vocabulary utilization
            utilization = report['vocabulary_utilization']
            active = utilization.get('active_vocabulary_size', 0)
            total = utilization.get('total_vocabulary_size', 1)
            unused = total - active
            
            plt.figure(figsize=(8, 6))
            plt.pie([active, unused], labels=['Used', 'Unused'], autopct='%1.1f%%', startangle=90)
            plt.title('Vocabulary Utilization')
            plt.savefig(output_dir / 'vocabulary_utilization.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Growth events over time
            growth_history = self.tokenizer.get_growth_history()
            if growth_history:
                steps = [event.tokens_processed for event in growth_history]
                vocab_sizes = [event.vocab_size_after for event in growth_history]
                
                plt.figure(figsize=(10, 6))
                plt.plot(steps, vocab_sizes, 'o-', linewidth=2, markersize=6)
                plt.xlabel('Tokens Processed')
                plt.ylabel('Vocabulary Size')
                plt.title('Vocabulary Growth Timeline')
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'growth_timeline.png', dpi=150, bbox_inches='tight')
                plt.close()
            
            # Plot 3: Token length distribution
            vocab_tokens = list(self.tokenizer.id_to_token.values())
            token_lengths = [len(token) for token in vocab_tokens]
            
            plt.figure(figsize=(10, 6))
            plt.hist(token_lengths, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Token Length (characters)')
            plt.ylabel('Frequency')
            plt.title('Token Length Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'token_length_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot 4: Overall scores radar chart
            scores = report['overall_scores']
            categories = ['Compression', 'Vocabulary', 'Quality', 'Growth']
            values = [
                scores['compression_score'],
                scores['vocabulary_score'],
                scores['quality_score'],
                scores['growth_score']
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            ax.plot(angles, values, 'o-', linewidth=2, color='blue')
            ax.fill(angles, values, alpha=0.25, color='blue')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('Overall Performance Scores', pad=20)
            plt.savefig(output_dir / 'performance_radar.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization plots saved to: {output_dir}")
        
        except Exception as e:
            logger.error(f"Failed to generate visualization plots: {e}")

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate a growing BPE tokenizer")
    parser.add_argument("--tokenizer", type=str, required=True,
                       help="Path to saved tokenizer JSON file")
    parser.add_argument("--output", type=str, default="evaluation_report.json",
                       help="Output file for evaluation report")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--max-texts", type=int, default=1000,
                       help="Maximum number of texts to evaluate")
    
    args = parser.parse_args()
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from: {args.tokenizer}")
    tokenizer = GrowingBPETokenizer.load(args.tokenizer)
    
    # Create evaluator
    evaluator = TokenizerEvaluator(tokenizer)
    
    # Load evaluation data
    logger.info("Loading evaluation data...")
    config = tokenizer.config
    data_loader = create_data_loader(config)
    data_loader.load_dataset()
    
    # Get evaluation texts
    eval_texts = []
    for batch in data_loader.get_text_stream(batch_size=100):
        eval_texts.extend(batch)
        if len(eval_texts) >= args.max_texts:
            break
    
    eval_texts = eval_texts[:args.max_texts]
    logger.info(f"Evaluating on {len(eval_texts)} texts")
    
    # Generate comprehensive report
    report = evaluator.generate_report(eval_texts, args.output)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    summary = report['evaluation_summary']
    print(f"Texts evaluated: {summary['texts_evaluated']}")
    print(f"Total characters: {summary['total_characters']:,}")
    print(f"Vocabulary size: {summary['tokenizer_vocab_size']}")
    print(f"Growth events: {summary['growth_events']}")
    
    compression = report['compression_metrics']
    print(f"\nCompression ratio: {compression['compression_ratio']:.3f}")
    print(f"Chars per token: {compression['avg_chars_per_token']:.2f}")
    
    utilization = report['vocabulary_utilization']
    print(f"\nVocab utilization: {utilization['vocabulary_utilization']:.3f}")
    print(f"Active tokens: {utilization['active_vocabulary_size']}/{utilization['total_vocabulary_size']}")
    
    quality = report['subword_quality']
    print(f"\nAvg token length: {quality['avg_token_length']:.2f}")
    print(f"Subword coherence: {quality['subword_coherence_score']:.3f}")
    
    scores = report['overall_scores']
    print(f"\nOverall score: {scores['overall_score']:.3f}")
    print("=" * 26)
    
    # Generate visualizations if requested
    if args.visualize:
        output_dir = Path(args.output).parent / "evaluation_plots"
        evaluator.visualize_results(report, str(output_dir))

if __name__ == "__main__":
    main()