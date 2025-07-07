"""
Test script for both fixed models to verify they work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from fixed_tiny_pfn import FixedTinyPFN, create_ridge_regression_data, test_fixed_tiny_pfn
from fixed_naive_transformer import FixedNaiveTransformer, test_fixed_naive_transformer


def test_both_models():
    """Test both fixed models."""
    print("=== Testing Fixed Models ===")
    
    # Test individual models
    print("\n1. Testing Fixed TinyPFN...")
    test_fixed_tiny_pfn()
    
    print("\n2. Testing Fixed Naive Transformer...")
    test_fixed_naive_transformer()
    
    print("\n3. Testing compatibility (same input/output shapes)...")
    
    # Create models
    tiny_pfn = FixedTinyPFN(num_features=10, d_model=256, n_heads=4)
    naive_tf = FixedNaiveTransformer(num_features=10, d_model=256, n_heads=4)
    
    # Create test data
    x_train, y_train, x_test, y_test = create_ridge_regression_data(batch_size=4, seq_len=20, num_features=10)
    
    # Test forward pass
    with torch.no_grad():
        logits_tiny = tiny_pfn(x_train, y_train, x_test)
        logits_naive = naive_tf(x_train, y_train, x_test)
        
        pred_tiny = tiny_pfn.predict_mean(logits_tiny)
        pred_naive = naive_tf.predict_mean(logits_naive)
        
        print(f"TinyPFN logits shape: {logits_tiny.shape}")
        print(f"Naive TF logits shape: {logits_naive.shape}")
        print(f"TinyPFN predictions shape: {pred_tiny.shape}")
        print(f"Naive TF predictions shape: {pred_naive.shape}")
        
        # Check shapes match
        assert logits_tiny.shape == logits_naive.shape, f"Logits shapes don't match: {logits_tiny.shape} vs {logits_naive.shape}"
        assert pred_tiny.shape == pred_naive.shape, f"Prediction shapes don't match: {pred_tiny.shape} vs {pred_naive.shape}"
        
        print("✓ Output shapes match!")
    
    print("\n4. Testing training step...")
    
    # Test training step
    optimizer_tiny = optim.Adam(tiny_pfn.parameters(), lr=0.001)
    optimizer_naive = optim.Adam(naive_tf.parameters(), lr=0.001)
    
    # Forward pass
    logits_tiny = tiny_pfn(x_train, y_train, x_test)
    logits_naive = naive_tf(x_train, y_train, x_test)
    
    # Calculate loss
    loss_tiny = tiny_pfn.compute_loss(logits_tiny, y_test.squeeze(-1))
    loss_naive = naive_tf.compute_loss(logits_naive, y_test.squeeze(-1))
    
    print(f"TinyPFN loss: {loss_tiny.item():.4f}")
    print(f"Naive TF loss: {loss_naive.item():.4f}")
    
    # Backward pass
    optimizer_tiny.zero_grad()
    loss_tiny.backward()
    optimizer_tiny.step()
    
    optimizer_naive.zero_grad()
    loss_naive.backward()
    optimizer_naive.step()
    
    print("✓ Training steps completed successfully!")
    
    print("\n5. Architecture comparison...")
    
    # Count parameters
    tiny_pfn_params = sum(p.numel() for p in tiny_pfn.parameters())
    naive_tf_params = sum(p.numel() for p in naive_tf.parameters())
    
    print(f"Fixed TinyPFN parameters: {tiny_pfn_params:,}")
    print(f"Fixed Naive Transformer parameters: {naive_tf_params:,}")
    
    # Show architecture differences
    print("\nArchitecture differences:")
    print("Fixed TinyPFN:")
    print("- Feature attention (intra-row)")
    print("- Item attention (inter-row)")
    print("- Dual attention mechanism")
    print()
    print("Fixed Naive Transformer:")
    print("- Sequential attention over flattened tokens")
    print("- Standard transformer architecture")
    print("- Same cell-level encoding")
    
    print("\n✓ All tests passed! Both models are ready for comparison.")


if __name__ == "__main__":
    test_both_models() 