"""
Comprehensive Model Comparison

Compares three approaches:
1. Basic Bayesian TinyPFN (dual attention)
2. Enhanced TinyPFN (with real TabPFN techniques)
3. Basic Bayesian Naive Transformer (sequential attention)
4. Sklearn Ridge (baseline)

This will show if TabPFN's sophisticated techniques help in single-layer setting.
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
from enhanced_tiny_pfn import EnhancedTinyPFN


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


def train_model(model, model_name, num_epochs=1000, batch_size=32, seq_len=50, num_features=10):
    """Train any model using Bayesian PD-NLL loss."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    
    print(f"Training {model_name}...")
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
        
        if epoch % 200 == 0:
            print(f"  Epoch {epoch}, PD-NLL Loss: {loss.item():.4f}")
    
    return losses


def evaluate_model(model, model_name, num_batches=50, seq_len=50, num_features=10):
    """Evaluate model on test data."""
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
            
            # Calculate NLL
            nll = model.compute_loss(x_train, y_train, x_test, y_test)
            total_nll += nll.item()
            
            total_samples += 1
    
    avg_mse = total_mse / total_samples
    avg_nll = total_nll / total_samples
    
    print(f"{model_name} - MSE: {avg_mse:.6f}, NLL: {avg_nll:.6f}")
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
    
    avg_mse = np.mean(mse_scores)
    print(f"Sklearn Ridge - MSE: {avg_mse:.6f}")
    return avg_mse


def main():
    """Comprehensive model comparison."""
    print("=== Comprehensive Model Comparison ===")
    print("Comparing TabPFN techniques in single-layer setting")
    print()
    
    # Hyperparameters
    num_features = 10
    d_model = 256
    n_heads = 4
    seq_len = 50
    num_epochs = 1000
    batch_size = 32
    
    # Create all models
    print("Creating models...")
    
    # 1. Basic Bayesian TinyPFN
    basic_pfn = BayesianTinyPFN(
        num_features=num_features, 
        d_model=d_model, 
        n_heads=n_heads
    )
    
    # 2. Enhanced TinyPFN with TabPFN techniques
    enhanced_pfn = EnhancedTinyPFN(
        num_features=num_features,
        d_model=d_model,
        n_heads=n_heads,
        multiquery_feature_attention=True,  # TabPFN technique
        multiquery_item_attention=False,    # Keep item attention full
        zero_init=True,                     # TabPFN technique
        init_gain=1.0,                      # TabPFN technique  
        second_mlp=True,                    # TabPFN technique
        feature_pos_embedding="learned",    # TabPFN technique
        activation="gelu"                   # TabPFN default
    )
    
    # 3. Basic Bayesian Naive Transformer
    naive_tf = BayesianNaiveTransformer(
        num_features=num_features,
        d_model=d_model,
        n_heads=n_heads
    )
    
    # Count parameters
    basic_pfn_params = sum(p.numel() for p in basic_pfn.parameters())
    enhanced_pfn_params = sum(p.numel() for p in enhanced_pfn.parameters()) 
    naive_tf_params = sum(p.numel() for p in naive_tf.parameters())
    
    print(f"Basic Bayesian TinyPFN parameters: {basic_pfn_params:,}")
    print(f"Enhanced TinyPFN parameters: {enhanced_pfn_params:,}")
    print(f"Naive Transformer parameters: {naive_tf_params:,}")
    print()
    
    # Train all models
    basic_pfn_losses = train_model(basic_pfn, "Basic Bayesian TinyPFN", 
                                  num_epochs, batch_size, seq_len, num_features)
    
    enhanced_pfn_losses = train_model(enhanced_pfn, "Enhanced TinyPFN", 
                                     num_epochs, batch_size, seq_len, num_features)
    
    naive_tf_losses = train_model(naive_tf, "Naive Transformer", 
                                 num_epochs, batch_size, seq_len, num_features)
    
    # Evaluate all models
    print("\nEvaluating models...")
    basic_pfn_mse, basic_pfn_nll = evaluate_model(basic_pfn, "Basic Bayesian TinyPFN", 
                                                  num_batches=50, seq_len=seq_len, num_features=num_features)
    
    enhanced_pfn_mse, enhanced_pfn_nll = evaluate_model(enhanced_pfn, "Enhanced TinyPFN", 
                                                       num_batches=50, seq_len=seq_len, num_features=num_features)
    
    naive_tf_mse, naive_tf_nll = evaluate_model(naive_tf, "Naive Transformer", 
                                               num_batches=50, seq_len=seq_len, num_features=num_features)
    
    # Sklearn baseline
    print("Computing sklearn baseline...")
    sklearn_mse = sklearn_baseline(num_tests=50, seq_len=seq_len, num_features=num_features)
    
    # Results summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    results = [
        ("Basic Bayesian TinyPFN", basic_pfn_mse, basic_pfn_nll, basic_pfn_params),
        ("Enhanced TinyPFN", enhanced_pfn_mse, enhanced_pfn_nll, enhanced_pfn_params),
        ("Naive Transformer", naive_tf_mse, naive_tf_nll, naive_tf_params),
        ("Sklearn Ridge", sklearn_mse, None, None)
    ]
    
    # Sort by MSE
    results.sort(key=lambda x: x[1])
    
    print(f"{'Rank':<5} {'Model':<25} {'MSE':<12} {'NLL':<12} {'Params':<12}")
    print("-" * 70)
    
    for i, (name, mse, nll, params) in enumerate(results, 1):
        nll_str = f"{nll:.6f}" if nll is not None else "N/A"
        params_str = f"{params:,}" if params is not None else "N/A"
        print(f"{i:<5} {name:<25} {mse:<12.6f} {nll_str:<12} {params_str:<12}")
    
    # Performance analysis
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    best_model = results[0][0]
    best_mse = results[0][1]
    
    print(f"🏆 Best Model: {best_model} (MSE: {best_mse:.6f})")
    print()
    
    # Compare enhanced vs basic
    if enhanced_pfn_mse < basic_pfn_mse:
        improvement = ((basic_pfn_mse - enhanced_pfn_mse) / basic_pfn_mse) * 100
        print(f"✅ Enhanced TinyPFN outperforms Basic TinyPFN by {improvement:.1f}%")
    else:
        degradation = ((enhanced_pfn_mse - basic_pfn_mse) / basic_pfn_mse) * 100
        print(f"❌ Enhanced TinyPFN underperforms Basic TinyPFN by {degradation:.1f}%")
    
    # Compare vs sklearn
    for name, mse, _, _ in results[:-1]:  # Exclude sklearn from this comparison
        ratio = mse / sklearn_mse
        if ratio < 2.0:
            print(f"🎯 {name} is competitive with sklearn ({ratio:.2f}x MSE)")
        else:
            print(f"📈 {name} needs improvement ({ratio:.2f}x sklearn MSE)")
    
    # Plot comprehensive results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    ax1.plot(basic_pfn_losses, label='Basic Bayesian TinyPFN', alpha=0.7)
    ax1.plot(enhanced_pfn_losses, label='Enhanced TinyPFN', alpha=0.7)
    ax1.plot(naive_tf_losses, label='Naive Transformer', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('PD-NLL Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend()
    ax1.set_yscale('log')
    
    # MSE comparison
    models = [name for name, _, _, _ in results]
    mse_values = [mse for _, mse, _, _ in results]
    colors = ['blue', 'red', 'orange', 'green']
    
    bars = ax2.bar(models, mse_values, color=colors, alpha=0.7)
    ax2.set_ylabel('MSE')
    ax2.set_title('MSE Comparison')
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, mse_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    # NLL comparison (excluding sklearn and any models with None NLL)
    nll_data = [(name, nll) for name, _, nll, _ in results[:-1] if nll is not None]
    
    if nll_data:
        nll_models = [name for name, nll in nll_data]
        nll_values = [nll for name, nll in nll_data]
        
        bars = ax3.bar(nll_models, nll_values, color=colors[:len(nll_models)], alpha=0.7)
        ax3.set_ylabel('NLL')
        ax3.set_title('NLL Comparison')
        ax3.set_yscale('log')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, nll_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No NLL data available', ha='center', va='center', transform=ax3.transAxes)
    
    # Parameter count comparison (excluding sklearn and any models with None params)
    param_data = [(name, params) for name, _, _, params in results[:-1] if params is not None]
    
    if param_data:
        param_models = [name for name, params in param_data]
        param_values = [params for name, params in param_data]
        
        bars = ax4.bar(param_models, param_values, color=colors[:len(param_models)], alpha=0.7)
        ax4.set_ylabel('Parameters')
        ax4.set_title('Parameter Count')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, param_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                    f'{value:,}', ha='center', va='bottom', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No parameter data available', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig('comprehensive_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Technical analysis
    print("\n" + "="*60)
    print("TECHNICAL ANALYSIS")
    print("="*60)
    
    print("TabPFN Techniques in Enhanced Model:")
    print("✓ Multiquery feature attention (shared KV)")
    print("✓ Learnable attention temperature")
    print("✓ Zero initialization (identity startup)")
    print("✓ Attention init gain")
    print("✓ Second MLP between attention layers")
    print("✓ Learned feature positional embeddings")
    print("✓ GELU activation")
    print()
    
    print("Key Insights:")
    if enhanced_pfn_mse < basic_pfn_mse:
        print("- TabPFN techniques DO help in single-layer setting")
        print("- Sophisticated initialization and architecture matter")
    else:
        print("- TabPFN techniques don't show clear benefits with single layer")
        print("- May need more layers for these techniques to shine")
    
    print(f"- All neural models still {results[0][1]/sklearn_mse:.1f}x sklearn Ridge MSE")
    print("- Single layers may be fundamentally limited for this task")
    print("- Cell-level representations were the key architectural fix")


if __name__ == "__main__":
    main() 