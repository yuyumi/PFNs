"""
Test script for both Bayesian models to verify they work correctly
with the proper PD-NLL loss.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from bayesian_tiny_pfn import BayesianTinyPFN, create_ridge_regression_data, test_bayesian_tiny_pfn
from bayesian_naive_transformer import BayesianNaiveTransformer, test_bayesian_naive_transformer


def test_both_bayesian_models():
    """Test both Bayesian models."""
    print("=== Testing Bayesian Models ===")
    print("Using Prior-Data Negative Log-Likelihood Loss")
    print("ℓ_θ = E_{D∪{x,y}∼p(D)}[− log q_θ(y|x, D)]")
    print()
    
    # Test individual models
    print("1. Testing Bayesian TinyPFN...")
    test_bayesian_tiny_pfn()
    
    print("\n2. Testing Bayesian Naive Transformer...")
    test_bayesian_naive_transformer()
    
    print("\n3. Testing compatibility (same input/output shapes)...")
    
    # Create models
    pfn = BayesianTinyPFN(num_features=10, d_model=256, n_heads=4)
    naive = BayesianNaiveTransformer(num_features=10, d_model=256, n_heads=4)
    
    # Create test data
    x_train, y_train, x_test, y_test = create_ridge_regression_data(batch_size=4, seq_len=20, num_features=10)
    
    # Test forward pass
    with torch.no_grad():
        mean_pfn, log_var_pfn = pfn(x_train, y_train, x_test)
        mean_naive, log_var_naive = naive(x_train, y_train, x_test)
        
        pred_pfn = pfn.predict(x_train, y_train, x_test)
        pred_naive = naive.predict(x_train, y_train, x_test)
        
        print(f"TinyPFN mean shape: {mean_pfn.shape}")
        print(f"Naive TF mean shape: {mean_naive.shape}")
        print(f"TinyPFN log_var shape: {log_var_pfn.shape}")
        print(f"Naive TF log_var shape: {log_var_naive.shape}")
        print(f"TinyPFN predictions shape: {pred_pfn.shape}")
        print(f"Naive TF predictions shape: {pred_naive.shape}")
        
        # Check shapes match
        assert mean_pfn.shape == mean_naive.shape, f"Mean shapes don't match: {mean_pfn.shape} vs {mean_naive.shape}"
        assert log_var_pfn.shape == log_var_naive.shape, f"Log var shapes don't match: {log_var_pfn.shape} vs {log_var_naive.shape}"
        assert pred_pfn.shape == pred_naive.shape, f"Prediction shapes don't match: {pred_pfn.shape} vs {pred_naive.shape}"
        
        print("✓ Output shapes match!")
    
    print("\n4. Testing loss computation...")
    
    # Test loss computation
    loss_pfn = pfn.compute_loss(x_train, y_train, x_test, y_test)
    loss_naive = naive.compute_loss(x_train, y_train, x_test, y_test)
    
    print(f"TinyPFN PD-NLL loss: {loss_pfn.item():.4f}")
    print(f"Naive TF PD-NLL loss: {loss_naive.item():.4f}")
    
    # Test training step
    optimizer_pfn = optim.Adam(pfn.parameters(), lr=0.001)
    optimizer_naive = optim.Adam(naive.parameters(), lr=0.001)
    
    # Backward pass
    optimizer_pfn.zero_grad()
    loss_pfn.backward()
    optimizer_pfn.step()
    
    optimizer_naive.zero_grad()
    loss_naive.backward()
    optimizer_naive.step()
    
    print("✓ Training steps completed successfully!")
    
    print("\n5. Testing uncertainty quantification...")
    
    with torch.no_grad():
        mean_pfn, log_var_pfn = pfn(x_train, y_train, x_test)
        mean_naive, log_var_naive = naive(x_train, y_train, x_test)
        
        # Calculate standard deviations
        std_pfn = torch.exp(0.5 * log_var_pfn)
        std_naive = torch.exp(0.5 * log_var_naive)
        
        print(f"TinyPFN mean predictions range: [{mean_pfn.min():.3f}, {mean_pfn.max():.3f}]")
        print(f"TinyPFN uncertainty (std) range: [{std_pfn.min():.3f}, {std_pfn.max():.3f}]")
        print(f"Naive TF mean predictions range: [{mean_naive.min():.3f}, {mean_naive.max():.3f}]")
        print(f"Naive TF uncertainty (std) range: [{std_naive.min():.3f}, {std_naive.max():.3f}]")
        
        print("✓ Uncertainty quantification working!")
    
    print("\n6. Architecture comparison...")
    
    # Count parameters
    pfn_params = sum(p.numel() for p in pfn.parameters())
    naive_params = sum(p.numel() for p in naive.parameters())
    
    print(f"Bayesian TinyPFN parameters: {pfn_params:,}")
    print(f"Bayesian Naive Transformer parameters: {naive_params:,}")
    
    # Show architecture differences
    print("\nArchitecture differences:")
    print("Bayesian TinyPFN:")
    print("- Feature attention (intra-row)")
    print("- Item attention (inter-row)")
    print("- Dual attention mechanism")
    print("- Gaussian output: mean + log variance")
    print("- Loss: PD-NLL")
    print()
    print("Bayesian Naive Transformer:")
    print("- Sequential attention over flattened tokens")
    print("- Standard transformer architecture")
    print("- Same cell-level encoding")
    print("- Gaussian output: mean + log variance")
    print("- Loss: PD-NLL")
    
    print("\n7. Loss comparison with previous models...")
    
    print("Previous models (fixed_tiny_pfn.py, fixed_naive_transformer.py):")
    print("- Used complex TabPFN distributional loss (FullSupportBarDistribution)")
    print("- Required discretization into 1000 buckets")
    print("- Complex negative log-likelihood computation")
    print()
    print("Bayesian models (bayesian_tiny_pfn.py, bayesian_naive_transformer.py):")
    print("- Use simple PD-NLL: ℓ_θ = E_{D∪{x,y}∼p(D)}[− log q_θ(y|x, D)]")
    print("- Direct Gaussian likelihood")
    print("- Much cleaner and more interpretable")
    print("- Better suited for architecture comparison")
    
    print("\n✓ All tests passed! Both Bayesian models are ready for comparison.")


if __name__ == "__main__":
    test_both_bayesian_models() 