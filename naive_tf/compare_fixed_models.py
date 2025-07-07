"""
Compare Fixed Models with Proper Cell-Level Representations

This experiment compares the fixed TinyPFN (dual attention) vs fixed naive transformer
(sequential attention) using proper cell-level representations and correct training protocol.
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

from fixed_tiny_pfn import FixedTinyPFN
from fixed_naive_transformer import FixedNaiveTransformer


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


def train_model(model, num_epochs=1000, batch_size=32, seq_len=50, num_features=10):
    """Train model using proper PFN protocol."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    
    for epoch in range(num_epochs):
        # Generate fresh data for each epoch (like PFN)
        x_train, y_train, x_test, y_test = create_ridge_regression_data(
            batch_size=batch_size, seq_len=seq_len, num_features=num_features
        )
        
        # Forward pass
        logits = model(x_train, y_train, x_test)
        
        # Calculate loss using continuous NLL
        loss = model.compute_loss(logits, y_test.squeeze(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return losses


def evaluate_model(model, num_batches=50, seq_len=50, num_features=10):
    """Evaluate model on test data."""
    model.eval()
    
    total_mse = 0
    total_samples = 0
    
    with torch.no_grad():
        for _ in range(num_batches):
            # Generate test data
            x_train, y_train, x_test, y_test = create_ridge_regression_data(
                batch_size=1, seq_len=seq_len, num_features=num_features
            )
            
            # Get predictions
            logits = model(x_train, y_train, x_test)
            predictions = model.predict_mean(logits)
            
            # Calculate MSE
            mse = torch.mean((predictions - y_test.squeeze(-1)) ** 2)
            total_mse += mse.item()
            total_samples += 1
    
    avg_mse = total_mse / total_samples
    return avg_mse


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
    """Main comparison experiment."""
    print("=== Fixed Models Comparison ===")
    print("Using proper cell-level representations")
    print()
    
    # Hyperparameters
    num_features = 10
    d_model = 256
    n_heads = 4
    seq_len = 50
    num_epochs = 1000
    batch_size = 32
    
    # Create models
    print("Creating models...")
    tiny_pfn = FixedTinyPFN(num_features=num_features, d_model=d_model, n_heads=n_heads)
    naive_tf = FixedNaiveTransformer(num_features=num_features, d_model=d_model, n_heads=n_heads)
    
    # Count parameters
    tiny_pfn_params = sum(p.numel() for p in tiny_pfn.parameters())
    naive_tf_params = sum(p.numel() for p in naive_tf.parameters())
    
    print(f"Fixed TinyPFN parameters: {tiny_pfn_params:,}")
    print(f"Fixed Naive Transformer parameters: {naive_tf_params:,}")
    print()
    
    # Train models
    print("Training Fixed TinyPFN...")
    tiny_pfn_losses = train_model(tiny_pfn, num_epochs=num_epochs, batch_size=batch_size, 
                                  seq_len=seq_len, num_features=num_features)
    
    print("\nTraining Fixed Naive Transformer...")
    naive_tf_losses = train_model(naive_tf, num_epochs=num_epochs, batch_size=batch_size,
                                  seq_len=seq_len, num_features=num_features)
    
    # Evaluate models
    print("\nEvaluating models...")
    tiny_pfn_mse = evaluate_model(tiny_pfn, num_batches=50, seq_len=seq_len, num_features=num_features)
    naive_tf_mse = evaluate_model(naive_tf, num_batches=50, seq_len=seq_len, num_features=num_features)
    
    # Sklearn baseline
    print("Computing sklearn baseline...")
    sklearn_mse = sklearn_baseline(num_tests=50, seq_len=seq_len, num_features=num_features)
    
    # Results
    print("\n=== RESULTS ===")
    print(f"Fixed TinyPFN MSE: {tiny_pfn_mse:.6f}")
    print(f"Fixed Naive Transformer MSE: {naive_tf_mse:.6f}")
    print(f"Sklearn Ridge MSE: {sklearn_mse:.6f}")
    print()
    
    # Performance comparison
    if tiny_pfn_mse < naive_tf_mse:
        improvement = ((naive_tf_mse - tiny_pfn_mse) / naive_tf_mse) * 100
        print(f"✓ Fixed TinyPFN outperforms Naive Transformer by {improvement:.1f}%")
    else:
        degradation = ((tiny_pfn_mse - naive_tf_mse) / naive_tf_mse) * 100
        print(f"✗ Fixed TinyPFN underperforms Naive Transformer by {degradation:.1f}%")
    
    # Compare to sklearn
    print(f"Fixed TinyPFN vs Sklearn: {tiny_pfn_mse/sklearn_mse:.2f}x MSE")
    print(f"Fixed Naive Transformer vs Sklearn: {naive_tf_mse/sklearn_mse:.2f}x MSE")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(tiny_pfn_losses, label='Fixed TinyPFN', alpha=0.7)
    plt.plot(naive_tf_losses, label='Fixed Naive Transformer', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    models = ['Fixed TinyPFN', 'Fixed Naive TF', 'Sklearn Ridge']
    mse_values = [tiny_pfn_mse, naive_tf_mse, sklearn_mse]
    colors = ['blue', 'orange', 'green']
    
    plt.bar(models, mse_values, color=colors, alpha=0.7)
    plt.ylabel('MSE')
    plt.title('Final Performance Comparison')
    plt.yscale('log')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('fixed_models_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Architecture analysis
    print("\n=== ARCHITECTURE ANALYSIS ===")
    print("Fixed TinyPFN:")
    print("- Uses dual attention: feature attention (intra-row) + item attention (inter-row)")
    print("- Each cell gets specialized encoding")
    print("- Proper cell-level representations")
    print()
    print("Fixed Naive Transformer:")
    print("- Uses sequential attention over flattened cell tokens")
    print("- Same cell-level encoding as TinyPFN")
    print("- Standard transformer architecture")
    print()


if __name__ == "__main__":
    main() 