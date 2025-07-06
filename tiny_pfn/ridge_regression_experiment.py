"""
Ridge Regression Experiment with TinyPFN
Recreates the ridge regression experiment from the original PFN notebook
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tiny_pfn import TinyPFN

def get_batch_for_ridge_regression(batch_size=2, seq_len=100, num_features=1, 
                                   hyperparameters=None, device='cpu', **kwargs):
    """
    Generate ridge regression batches exactly like the original PFN notebook.
    
    Prior: f = x^T w, y ~ Normal(f, a^2 I)
    where w ~ Normal(0, b^2 I)
    """
    if hyperparameters is None:
        hyperparameters = {'a': 0.1, 'b': 1.0}
    
    # Sample weights w ~ Normal(0, b^2 I) for each dataset
    ws = torch.distributions.Normal(
        torch.zeros(num_features + 1), 
        hyperparameters['b']
    ).sample((batch_size,))
    
    # Sample inputs x ~ Uniform(0, 1)
    xs = torch.rand(batch_size, seq_len, num_features)
    
    # Add bias term (concatenate with ones)
    concatenated_xs = torch.cat([xs, torch.ones(batch_size, seq_len, 1)], 2)
    
    # Compute y = x^T w + noise
    ys = torch.distributions.Normal(
        torch.einsum('nmf, nf -> nm', concatenated_xs, ws),
        hyperparameters['a']
    ).sample()[..., None]
    
    return {
        'x': concatenated_xs.to(device),
        'y': ys.to(device), 
        'target_y': ys.to(device)
    }

def train_tiny_pfn_on_ridge_regression(epochs=20, batch_size=16, steps_per_epoch=50):
    """Train TinyPFN on ridge regression data following the original experiment"""
    print("Training TinyPFN on Ridge Regression Data")
    print("=" * 50)
    
    # Create TinyPFN model
    model = TinyPFN(
        num_features=2,  # 1 feature + bias
        d_model=64,
        n_heads=4,
        dropout=0.1,
        max_seq_len=100
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for step in range(steps_per_epoch):
            # Generate new synthetic data each step - this is the key PFN innovation
            batch = get_batch_for_ridge_regression(
                batch_size=batch_size, 
                seq_len=50,  # Smaller sequences for faster training
                num_features=1,
                hyperparameters={'a': 0.1, 'b': 1.0}
            )
            
            # Split into train/test
            train_len = 25
            x_train = batch['x'][:, :train_len, :]
            y_train = batch['y'][:, :train_len, :]
            x_test = batch['x'][:, train_len:, :]
            y_test = batch['y'][:, train_len:, :]
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(x_train, y_train, x_test)
            loss = criterion(predictions, y_test)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")
    
    print(f"Training complete! Final loss: {losses[-1]:.4f}")
    return model, losses

def analyze_pfn_performance(model):
    """Analyze the performance of our trained PFN following the original notebook"""
    print("\nAnalysis of the performance of our trained PFN")
    print("=" * 50)
    
    # Sample some datasets to look at
    batch = get_batch_for_ridge_regression(seq_len=100, batch_size=10)
    
    model.eval()
    
    # Analyze multiple examples
    for batch_index in range(3):  # Look at first 3 examples
        print(f"\nExample {batch_index + 1}:")
        
        num_training_points = 4
        
        train_x = batch['x'][batch_index, :num_training_points]
        train_y = batch['y'][batch_index, :num_training_points]
        test_x = batch['x'][batch_index]
        
        with torch.no_grad():
            # Add batch dimension as transformer expects that
            predictions = model(train_x[None], train_y[None], test_x[None])
            predictions = predictions[0].squeeze()
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.scatter(train_x[..., 0], train_y.squeeze(), color='blue', 
                   label='Training Data', s=50)
        
        # Sort test points for plotting
        order_test_x = test_x[..., 0].argsort()
        plt.plot(test_x[order_test_x, 0], predictions[order_test_x], 
                color='red', linewidth=2, label='TinyPFN Prediction')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'TinyPFN Ridge Regression - Example {batch_index + 1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Calculate simple metrics
        mse = torch.mean((predictions[num_training_points:] - 
                         batch['target_y'][batch_index, num_training_points:].squeeze())**2)
        print(f"MSE on test points: {mse:.4f}")

def demonstrate_ridge_regression_theory():
    """Demonstrate the ridge regression prior by sampling datasets"""
    print("\nDemonstrating Ridge Regression Prior")
    print("=" * 40)
    
    # Sample datasets from the prior
    batch = get_batch_for_ridge_regression(batch_size=10, seq_len=100, num_features=1)
    
    plt.figure(figsize=(12, 8))
    for dataset_index in range(len(batch['x'])):
        plt.scatter(batch['x'][dataset_index, :, 0].numpy(), 
                   batch['y'][dataset_index, :].numpy(), alpha=0.6)
    
    plt.title('Ridge Regression Prior: f = x^T w, y ~ Normal(f, a^2)\nw ~ Normal(0, b^2)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Each color represents a different dataset sampled from the ridge regression prior.")
    print("Notice how each dataset follows a linear trend with noise.")

def plot_training_progress(losses):
    """Plot training loss over time"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('TinyPFN Training Progress on Ridge Regression')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_context_sizes(model):
    """Show how performance improves with more context"""
    print("\nIn-Context Learning: Effect of Context Size")
    print("=" * 45)
    
    # Generate a test dataset
    batch = get_batch_for_ridge_regression(batch_size=1, seq_len=100, num_features=1)
    test_x = batch['x'][0]
    test_y = batch['y'][0]
    
    context_sizes = [2, 5, 10, 20]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Effect of Context Size on TinyPFN Predictions', fontsize=14)
    
    for i, context_size in enumerate(context_sizes):
        ax = axes[i // 2, i % 2]
        
        train_x = test_x[:context_size]
        train_y = test_y[:context_size]
        
        with torch.no_grad():
            predictions = model(train_x[None], train_y[None], test_x[None])
            predictions = predictions[0].squeeze()
        
        # Plot
        ax.scatter(train_x[:, 0].numpy(), train_y.numpy().ravel(), 
                  c='blue', s=50, label=f'Training Data (n={context_size})')
        
        order_test_x = test_x[:, 0].argsort()
        ax.plot(test_x[order_test_x, 0], predictions[order_test_x], 
                c='red', linewidth=2, label='TinyPFN Prediction')
        
        ax.set_title(f'Context Size: {context_size}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Run the complete ridge regression experiment"""
    print("TinyPFN Ridge Regression Experiment")
    print("Following the original PFN notebook")
    print("=" * 50)
    
    # 1. Demonstrate the ridge regression prior
    demonstrate_ridge_regression_theory()
    
    # 2. Train TinyPFN on ridge regression data
    trained_model, losses = train_tiny_pfn_on_ridge_regression(
        epochs=20, 
        batch_size=16, 
        steps_per_epoch=50
    )
    
    # 3. Plot training progress
    plot_training_progress(losses)
    
    # 4. Analyze PFN performance
    analyze_pfn_performance(trained_model)
    
    # 5. Show effect of context size
    compare_context_sizes(trained_model)
    
    print("\nRidge Regression Experiment Complete!")
    print("Key observations:")
    print("- TinyPFN learns ridge regression from synthetic data")
    print("- Model shows in-context learning behavior")
    print("- More context improves prediction quality")
    print("- Each training step uses completely new synthetic data")

if __name__ == "__main__":
    main() 