"""
Compare Bayesian Models with Prior-Data Negative Log-Likelihood Loss

This experiment compares the Bayesian TinyPFN (dual attention) vs Bayesian naive transformer
(sequential attention) using the proper Bayesian PD-NLL loss from equation (2).

ℓ_θ = E_{D∪{x,y}∼p(D)}[− log q_θ(y|x, D)]

This is much cleaner than the complex TabPFN distributional loss.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from bayesian_tiny_pfn import BayesianTinyPFN
from bayesian_naive_transformer import BayesianNaiveTransformer


def create_ridge_regression_data(batch_size=32, seq_len=50, num_features=10):
    """Create ridge regression data for testing."""
    # Generate random ridge regression weights
    true_weights = torch.randn(batch_size, num_features, 1) * 0.5
    
    # Generate input data
    x = torch.randn(batch_size, seq_len, num_features)
    
    # Generate targets using ridge regression
    y = torch.bmm(x, true_weights)
    
    # Add noise
    noise = torch.randn_like(y) * 0.1
    y = y + noise
    
    # Split into train/test
    train_len = seq_len // 2
    x_train = x[:, :train_len]
    y_train = y[:, :train_len]
    x_test = x[:, train_len:]
    y_test = y[:, train_len:]
    
    return x_train, y_train, x_test, y_test


def train_bayesian_model(model, num_epochs=1000, batch_size=32, seq_len=50, num_features=10):
    """Train Bayesian model using proper PD-NLL loss."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    
    for epoch in range(num_epochs):
        # Generate fresh data for each epoch (like PFN)
        x_train, y_train, x_test, y_test = create_ridge_regression_data(
            batch_size=batch_size, seq_len=seq_len, num_features=num_features
        )
        
        # Forward pass and compute PD-NLL loss
        loss = model.compute_loss(x_train, y_train, x_test, y_test)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, PD-NLL Loss: {loss.item():.4f}")
    
    return losses


def evaluate_bayesian_model(model, num_batches=50, seq_len=50, num_features=10):
    """Evaluate Bayesian model on test data."""
    model.eval()
    
    total_mse = 0
    total_nll = 0
    total_samples = 0
    
    with torch.no_grad():
        for _ in range(num_batches):
            # Generate test data
            x_train, y_train, x_test, y_test = create_ridge_regression_data(
                batch_size=1, seq_len=seq_len, num_features=num_features
            )
            
            # Get predictions
            predictions = model.predict(x_train, y_train, x_test)
            
            # Calculate MSE
            mse = torch.mean((predictions - y_test.squeeze(-1)) ** 2)
            total_mse += mse.item()
            
            # Calculate NLL (for comparison)
            nll = model.compute_loss(x_train, y_train, x_test, y_test)
            total_nll += nll.item()
            
            total_samples += 1
    
    avg_mse = total_mse / total_samples
    avg_nll = total_nll / total_samples
    return avg_mse, avg_nll


def sklearn_baseline(num_tests=50, seq_len=50, num_features=10):
    """Ridge regression baseline using sklearn."""
    mse_scores = []
    
    for _ in range(num_tests):
        x_train, y_train, x_test, y_test = create_ridge_regression_data(
            batch_size=1, seq_len=seq_len, num_features=num_features
        )
        
        # Convert to numpy
        x_train_np = x_train.squeeze(0).numpy()
        y_train_np = y_train.squeeze(0).squeeze(-1).numpy()
        x_test_np = x_test.squeeze(0).numpy()
        y_test_np = y_test.squeeze(0).squeeze(-1).numpy()
        
        # Fit Ridge regression
        ridge = Ridge(alpha=0.1)
        ridge.fit(x_train_np, y_train_np)
        
        # Predict
        y_pred = ridge.predict(x_test_np)
        
        # Calculate MSE
        mse = mean_squared_error(y_test_np, y_pred)
        mse_scores.append(mse)
    
    return np.mean(mse_scores)


def main():
    """Main Bayesian comparison experiment."""
    print("=== Bayesian Models Comparison ===")
    print("Using Prior-Data Negative Log-Likelihood Loss")
    print("ℓ_θ = E_{D∪{x,y}∼p(D)}[− log q_θ(y|x, D)]")
    print()
    
    # Hyperparameters
    num_features = 10
    d_model = 256
    n_heads = 4
    seq_len = 50
    num_epochs = 1000
    batch_size = 32
    
    # Create models
    print("Creating Bayesian models...")
    bayesian_pfn = BayesianTinyPFN(num_features=num_features, d_model=d_model, n_heads=n_heads)
    bayesian_naive = BayesianNaiveTransformer(num_features=num_features, d_model=d_model, n_heads=n_heads)
    
    # Count parameters
    pfn_params = sum(p.numel() for p in bayesian_pfn.parameters())
    naive_params = sum(p.numel() for p in bayesian_naive.parameters())
    
    print(f"Bayesian TinyPFN parameters: {pfn_params:,}")
    print(f"Bayesian Naive Transformer parameters: {naive_params:,}")
    print()
    
    # Train models
    print("Training Bayesian TinyPFN...")
    pfn_losses = train_bayesian_model(bayesian_pfn, num_epochs=num_epochs, batch_size=batch_size, 
                                     seq_len=seq_len, num_features=num_features)
    
    print("\nTraining Bayesian Naive Transformer...")
    naive_losses = train_bayesian_model(bayesian_naive, num_epochs=num_epochs, batch_size=batch_size,
                                       seq_len=seq_len, num_features=num_features)
    
    # Evaluate models
    print("\nEvaluating models...")
    pfn_mse, pfn_nll = evaluate_bayesian_model(bayesian_pfn, num_batches=50, seq_len=seq_len, num_features=num_features)
    naive_mse, naive_nll = evaluate_bayesian_model(bayesian_naive, num_batches=50, seq_len=seq_len, num_features=num_features)
    
    # Sklearn baseline
    print("Computing sklearn baseline...")
    sklearn_mse = sklearn_baseline(num_tests=50, seq_len=seq_len, num_features=num_features)
    
    # Results
    print("\n=== RESULTS ===")
    print("MSE (Lower is better):")
    print(f"Bayesian TinyPFN MSE: {pfn_mse:.6f}")
    print(f"Bayesian Naive Transformer MSE: {naive_mse:.6f}")
    print(f"Sklearn Ridge MSE: {sklearn_mse:.6f}")
    print()
    print("NLL (Lower is better):")
    print(f"Bayesian TinyPFN NLL: {pfn_nll:.6f}")
    print(f"Bayesian Naive Transformer NLL: {naive_nll:.6f}")
    print()
    
    # Performance comparison
    if pfn_mse < naive_mse:
        improvement = ((naive_mse - pfn_mse) / naive_mse) * 100
        print(f"✓ Bayesian TinyPFN outperforms Naive Transformer by {improvement:.1f}%")
    else:
        degradation = ((pfn_mse - naive_mse) / naive_mse) * 100
        print(f"✗ Bayesian TinyPFN underperforms Naive Transformer by {degradation:.1f}%")
    
    # Compare to sklearn
    print(f"Bayesian TinyPFN vs Sklearn: {pfn_mse/sklearn_mse:.2f}x MSE")
    print(f"Bayesian Naive Transformer vs Sklearn: {naive_mse/sklearn_mse:.2f}x MSE")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Training curves
    plt.subplot(1, 3, 1)
    plt.plot(pfn_losses, label='Bayesian TinyPFN', alpha=0.7)
    plt.plot(naive_losses, label='Bayesian Naive TF', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('PD-NLL Loss')
    plt.title('Training Loss (PD-NLL)')
    plt.legend()
    plt.yscale('log')
    
    # MSE comparison
    plt.subplot(1, 3, 2)
    models = ['Bayesian TinyPFN', 'Bayesian Naive TF', 'Sklearn Ridge']
    mse_values = [pfn_mse, naive_mse, sklearn_mse]
    colors = ['blue', 'orange', 'green']
    
    bars = plt.bar(models, mse_values, color=colors, alpha=0.7)
    plt.ylabel('MSE')
    plt.title('MSE Comparison')
    plt.yscale('log')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, mse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{value:.4f}', ha='center', va='bottom')
    
    # NLL comparison
    plt.subplot(1, 3, 3)
    nll_models = ['Bayesian TinyPFN', 'Bayesian Naive TF']
    nll_values = [pfn_nll, naive_nll]
    colors = ['blue', 'orange']
    
    bars = plt.bar(nll_models, nll_values, color=colors, alpha=0.7)
    plt.ylabel('NLL')
    plt.title('NLL Comparison')
    plt.yscale('log')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, nll_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('bayesian_models_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Architecture analysis
    print("\n=== ARCHITECTURE ANALYSIS ===")
    print("Bayesian TinyPFN:")
    print("- Uses dual attention: feature attention (intra-row) + item attention (inter-row)")
    print("- Each cell gets specialized encoding")
    print("- Proper cell-level representations")
    print("- Gaussian output: mean + log variance")
    print("- Loss: PD-NLL (Bayesian)")
    print()
    print("Bayesian Naive Transformer:")
    print("- Uses sequential attention over flattened cell tokens")
    print("- Same cell-level encoding as TinyPFN")
    print("- Standard transformer architecture")
    print("- Gaussian output: mean + log variance")
    print("- Loss: PD-NLL (Bayesian)")
    print()
    print("Key difference: Attention mechanism (dual vs sequential)")
    print("Everything else is identical for fair comparison")
    
    # Loss analysis
    print("\n=== LOSS ANALYSIS ===")
    print("Previous models used complex TabPFN distributional loss (FullSupportBarDistribution)")
    print("Bayesian models use simple PD-NLL: ℓ_θ = E_{D∪{x,y}∼p(D)}[− log q_θ(y|x, D)]")
    print("This is much cleaner and more appropriate for architecture comparison")
    print()
    print("Benefits of Bayesian approach:")
    print("- Simpler, more interpretable loss")
    print("- Direct regression (no complex discretization)")
    print("- Uncertainty quantification (variance estimation)")
    print("- Better suited for continuous targets")


if __name__ == "__main__":
    main() 