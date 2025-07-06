"""
In-Context Learning Test: TinyPFN vs Naive Transformer on Ridge Regression

This script tests the actual in-context learning capabilities of both models
on ridge regression tasks, similar to the notebook examples.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Import models
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tiny_pfn'))
from tiny_pfn import TinyPFN
from naive_transformer import NaiveTransformer


def create_ridge_regression_data(batch_size=32, seq_len=50, num_features=10, alpha=0.1):
    """
    Create ridge regression data exactly like in the notebook.
    """
    # Generate random ridge regression coefficients
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
    
    return x_train, y_train, x_test, y_test, true_weights


def test_in_context_learning(model, num_tests=100, context_sizes=[5, 10, 20, 40]):
    """
    Test in-context learning performance with different context sizes.
    """
    model.eval()
    results = {}
    
    for context_size in context_sizes:
        print(f"Testing context size: {context_size}")
        
        mse_scores = []
        
        for _ in range(num_tests):
            # Create fresh data for each test
            seq_len = context_size + 10  # context + test points
            x_train, y_train, x_test, y_test, true_weights = create_ridge_regression_data(
                batch_size=1, seq_len=seq_len, num_features=10
            )
            
            with torch.no_grad():
                # Get model predictions
                logits = model(x_train, y_train, x_test)
                predictions = model.predict_mean(logits).squeeze()
                
                # Calculate MSE
                mse = torch.mean((predictions - y_test.squeeze())**2).item()
                mse_scores.append(mse)
        
        results[context_size] = {
            'mean_mse': np.mean(mse_scores),
            'std_mse': np.std(mse_scores)
        }
        
        print(f"  MSE: {results[context_size]['mean_mse']:.4f} ± {results[context_size]['std_mse']:.4f}")
    
    return results


def compare_to_sklearn_ridge(num_tests=100, context_sizes=[5, 10, 20, 40]):
    """
    Compare to sklearn Ridge regression baseline.
    """
    results = {}
    
    for context_size in context_sizes:
        print(f"Testing sklearn Ridge with context size: {context_size}")
        
        mse_scores = []
        
        for _ in range(num_tests):
            # Create fresh data
            seq_len = context_size + 10
            x_train, y_train, x_test, y_test, true_weights = create_ridge_regression_data(
                batch_size=1, seq_len=seq_len, num_features=10
            )
            
            # Convert to numpy for sklearn
            x_train_np = x_train.squeeze().numpy()
            y_train_np = y_train.squeeze().numpy()
            x_test_np = x_test.squeeze().numpy()
            y_test_np = y_test.squeeze().numpy()
            
            # Fit ridge regression
            ridge = Ridge(alpha=0.1)
            ridge.fit(x_train_np, y_train_np)
            
            # Predict
            predictions = ridge.predict(x_test_np)
            
            # Calculate MSE
            mse = mean_squared_error(y_test_np, predictions)
            mse_scores.append(mse)
        
        results[context_size] = {
            'mean_mse': np.mean(mse_scores),
            'std_mse': np.std(mse_scores)
        }
        
        print(f"  MSE: {results[context_size]['mean_mse']:.4f} ± {results[context_size]['std_mse']:.4f}")
    
    return results


def test_feature_understanding(model, num_tests=50):
    """
    Test if the model understands feature relationships.
    """
    model.eval()
    
    print("Testing feature understanding...")
    
    # Create data where only specific features matter
    batch_size = 1
    seq_len = 30
    num_features = 10
    
    correlation_scores = []
    
    for test_idx in range(num_tests):
        # Create data where only features 0, 2, 4 have non-zero coefficients
        true_weights = torch.zeros(batch_size, num_features, 1)
        important_features = [0, 2, 4]
        for feat in important_features:
            true_weights[:, feat, :] = torch.randn(batch_size, 1) * 0.5
        
        x = torch.randn(batch_size, seq_len, num_features)
        y = torch.bmm(x, true_weights) + torch.randn(batch_size, seq_len, 1) * 0.1
        
        train_len = seq_len // 2
        x_train = x[:, :train_len]
        y_train = y[:, :train_len]
        x_test = x[:, train_len:]
        y_test = y[:, train_len:]
        
        with torch.no_grad():
            logits = model(x_train, y_train, x_test)
            predictions = model.predict_mean(logits).squeeze()
            
            # Calculate correlation between predictions and true targets
            correlation = torch.corrcoef(torch.stack([predictions, y_test.squeeze()]))[0, 1]
            if not torch.isnan(correlation):
                correlation_scores.append(correlation.item())
    
    mean_correlation = np.mean(correlation_scores)
    std_correlation = np.std(correlation_scores)
    
    print(f"  Feature understanding (correlation): {mean_correlation:.4f} ± {std_correlation:.4f}")
    
    return mean_correlation, std_correlation


def visualize_predictions(model, model_name):
    """
    Visualize predictions on a single example.
    """
    model.eval()
    
    # Create a single example
    x_train, y_train, x_test, y_test, true_weights = create_ridge_regression_data(
        batch_size=1, seq_len=30, num_features=10
    )
    
    with torch.no_grad():
        logits = model(x_train, y_train, x_test)
        predictions = model.predict_mean(logits).squeeze()
    
    # Convert to numpy for plotting
    x_train_np = x_train.squeeze().numpy()
    y_train_np = y_train.squeeze().numpy()
    x_test_np = x_test.squeeze().numpy()
    y_test_np = y_test.squeeze().numpy()
    predictions_np = predictions.numpy()
    
    # Plot first feature vs target
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(x_train_np[:, 0], y_train_np, alpha=0.7, label='Training data')
    plt.scatter(x_test_np[:, 0], y_test_np, alpha=0.7, label='True test')
    plt.scatter(x_test_np[:, 0], predictions_np, alpha=0.7, label=f'{model_name} predictions')
    plt.xlabel('Feature 0')
    plt.ylabel('Target')
    plt.title(f'{model_name}: Feature 0 vs Target')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(y_test_np, label='True values')
    plt.plot(predictions_np, label=f'{model_name} predictions')
    plt.xlabel('Test point')
    plt.ylabel('Target value')
    plt.title(f'{model_name}: Predictions vs True')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.scatter(y_test_np, predictions_np, alpha=0.7)
    plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--')
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.title(f'{model_name}: Prediction scatter')
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to test in-context learning capabilities.
    """
    print("🧪 In-Context Learning Test: TinyPFN vs Naive Transformer")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create models (smaller for faster testing)
    print("Creating models...")
    tinypfn = TinyPFN(num_features=10, d_model=256, n_heads=4, max_seq_len=60)
    naive_tf = NaiveTransformer(num_features=10, d_model=256, n_heads=4, max_seq_len=60)
    
    # First, let's train them briefly on ridge regression
    print("\n📚 Quick training on ridge regression...")
    
    def quick_train(model, num_epochs=20):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            x_train, y_train, x_test, y_test, _ = create_ridge_regression_data(batch_size=32)
            
            logits = model(x_train, y_train, x_test)
            loss = model.criterion(logits, y_test).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    print("Training TinyPFN...")
    quick_train(tinypfn)
    
    print("\nTraining Naive Transformer...")
    quick_train(naive_tf)
    
    # Test in-context learning with different context sizes
    print("\n🎯 Testing TinyPFN in-context learning...")
    tinypfn_results = test_in_context_learning(tinypfn)
    
    print("\n🎯 Testing Naive Transformer in-context learning...")
    naive_results = test_in_context_learning(naive_tf)
    
    print("\n🎯 Testing sklearn Ridge baseline...")
    sklearn_results = compare_to_sklearn_ridge()
    
    # Test feature understanding
    print("\n🔍 Testing feature understanding...")
    tinypfn_corr, tinypfn_corr_std = test_feature_understanding(tinypfn)
    naive_corr, naive_corr_std = test_feature_understanding(naive_tf)
    
    # Visualize predictions
    print("\n📊 Visualizing predictions...")
    visualize_predictions(tinypfn, "TinyPFN")
    visualize_predictions(naive_tf, "Naive Transformer")
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    # MSE comparison
    plt.subplot(1, 3, 1)
    context_sizes = [5, 10, 20, 40]
    
    tinypfn_mse = [tinypfn_results[size]['mean_mse'] for size in context_sizes]
    naive_mse = [naive_results[size]['mean_mse'] for size in context_sizes]
    sklearn_mse = [sklearn_results[size]['mean_mse'] for size in context_sizes]
    
    plt.plot(context_sizes, tinypfn_mse, 'b-o', label='TinyPFN')
    plt.plot(context_sizes, naive_mse, 'r-o', label='Naive Transformer')
    plt.plot(context_sizes, sklearn_mse, 'g-o', label='sklearn Ridge')
    plt.xlabel('Context Size')
    plt.ylabel('MSE')
    plt.title('In-Context Learning Performance')
    plt.legend()
    plt.yscale('log')
    
    # Feature understanding comparison
    plt.subplot(1, 3, 2)
    models = ['TinyPFN', 'Naive Transformer']
    correlations = [tinypfn_corr, naive_corr]
    errors = [tinypfn_corr_std, naive_corr_std]
    
    bars = plt.bar(models, correlations, yerr=errors, alpha=0.7, 
                   color=['blue', 'red'], capsize=5)
    plt.ylabel('Correlation with True Values')
    plt.title('Feature Understanding')
    plt.ylim(0, 1)
    
    # Add value labels
    for bar, corr in zip(bars, correlations):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{corr:.3f}', ha='center', va='bottom')
    
    # Context size effect
    plt.subplot(1, 3, 3)
    tinypfn_improvement = [(sklearn_mse[i] - tinypfn_mse[i]) / sklearn_mse[i] * 100 
                          for i in range(len(context_sizes))]
    naive_improvement = [(sklearn_mse[i] - naive_mse[i]) / sklearn_mse[i] * 100 
                        for i in range(len(context_sizes))]
    
    plt.plot(context_sizes, tinypfn_improvement, 'b-o', label='TinyPFN vs sklearn')
    plt.plot(context_sizes, naive_improvement, 'r-o', label='Naive Transformer vs sklearn')
    plt.xlabel('Context Size')
    plt.ylabel('Improvement over sklearn (%)')
    plt.title('In-Context Learning Advantage')
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('in_context_learning_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary
    print("\n📊 Summary:")
    print("=" * 60)
    
    # Find best context size performance
    best_context_size = min(context_sizes, key=lambda x: tinypfn_results[x]['mean_mse'])
    tinypfn_best = tinypfn_results[best_context_size]['mean_mse']
    naive_best = naive_results[best_context_size]['mean_mse']
    sklearn_best = sklearn_results[best_context_size]['mean_mse']
    
    print(f"Best MSE @ context size {best_context_size}:")
    print(f"  TinyPFN: {tinypfn_best:.4f}")
    print(f"  Naive Transformer: {naive_best:.4f}")
    print(f"  sklearn Ridge: {sklearn_best:.4f}")
    
    improvement = (naive_best - tinypfn_best) / naive_best * 100
    vs_sklearn = (sklearn_best - tinypfn_best) / sklearn_best * 100
    
    print(f"\nTinyPFN vs Naive Transformer: {improvement:.1f}% improvement")
    print(f"TinyPFN vs sklearn Ridge: {vs_sklearn:.1f}% improvement")
    
    print(f"\nFeature Understanding:")
    print(f"  TinyPFN correlation: {tinypfn_corr:.3f} ± {tinypfn_corr_std:.3f}")
    print(f"  Naive Transformer correlation: {naive_corr:.3f} ± {naive_corr_std:.3f}")
    
    if tinypfn_corr > naive_corr:
        print(f"  ✅ TinyPFN shows better feature understanding!")
    else:
        print(f"  ❌ Naive Transformer shows better feature understanding.")
    
    print(f"\n💡 Key Insights:")
    if tinypfn_best < naive_best:
        print("✅ TinyPFN outperforms Naive Transformer on ridge regression!")
        print("✅ The dual attention mechanism helps with tabular in-context learning.")
    else:
        print("❌ Naive Transformer still outperforms TinyPFN.")
        print("🤔 This suggests our dual attention implementation may need refinement.")


if __name__ == "__main__":
    main() 