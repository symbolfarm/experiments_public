#!/usr/bin/env python3
"""
Quick test script to verify models work correctly.
"""

import torch
from transformers import GPT2Tokenizer

from config import ResidualTransformerConfig
from model import ResidualTransformer
from baseline_model import BaselineTransformer

def test_models():
    """Test that both models can run forward passes without errors."""
    
    print("Testing Residual Transformer models...")
    
    # Create small test config
    config = ResidualTransformerConfig(
        d_model=32,
        n_heads=2,
        d_ff=64,
        fixed_kv_length=16,
        n_processing_blocks=2,
        max_len=64,
        vocab_size=1000  # Small vocab for testing
    )
    
    print(f"Config: d_model={config.d_model}, fixed_kv_length={config.fixed_kv_length}, n_processing_blocks={config.n_processing_blocks}")
    
    # Initialize models
    residual_model = ResidualTransformer(config)
    baseline_model = BaselineTransformer(config)
    
    # Count parameters
    residual_params = sum(p.numel() for p in residual_model.parameters() if p.requires_grad)
    baseline_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    
    print(f"Residual model parameters: {residual_params:,}")
    print(f"Baseline model parameters: {baseline_params:,}")
    print(f"Parameter ratio: {residual_params / baseline_params:.3f}")
    
    # Create test input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Test input shape: {input_ids.shape}")
    
    # Test residual model
    print("\nTesting residual model...")
    residual_model.eval()
    with torch.no_grad():
        residual_logits, residual_loss = residual_model(input_ids, input_ids)
    
    print(f"Residual output shape: {residual_logits.shape}")
    print(f"Expected shape: {(batch_size, seq_len, config.vocab_size)}")
    print(f"Residual loss: {residual_loss.item() if residual_loss is not None else 'None'}")
    
    # Test baseline model
    print("\nTesting baseline model...")
    baseline_model.eval()
    with torch.no_grad():
        baseline_logits, baseline_loss = baseline_model(input_ids, input_ids)
    
    print(f"Baseline output shape: {baseline_logits.shape}")
    print(f"Baseline loss: {baseline_loss.item() if baseline_loss is not None else 'None'}")
    
    # Test generation
    print("\nTesting generation...")
    test_prompt = torch.randint(0, config.vocab_size, (1, 5))  # Short prompt
    
    residual_generated = residual_model.generate(test_prompt, max_length=20)
    baseline_generated = baseline_model.generate(test_prompt, max_length=20)
    
    print(f"Residual generated shape: {residual_generated.shape}")
    print(f"Baseline generated shape: {baseline_generated.shape}")
    
    # Test different initialization strategies
    print("\nTesting different residual initialization strategies...")
    
    for strategy in ["random", "pooled", "learned"]:
        config.residual_init_strategy = strategy
        model = ResidualTransformer(config)
        
        with torch.no_grad():
            logits, loss = model(input_ids, input_ids)
        
        print(f"Strategy '{strategy}': output shape {logits.shape}, loss {loss.item():.4f}")
    
    print("\n✅ All tests passed!")
    
    return True

if __name__ == "__main__":
    test_models()