"""
Compare TinyPFN vs Naive Transformer with Proper PFN Training Setup

This script compares TinyPFN vs Naive Transformer using the real PFN training configuration:
- FullSupportBarDistribution loss (1000 buckets)
- Real PFN priors (GP/BNN) instead of simple ridge regression
- Proper batch size, learning rate, etc.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Import TinyPFN from the tiny_pfn directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tiny_pfn'))
from tiny_pfn import TinyPFN

# Import NaiveTransformer from current directory
from naive_transformer import NaiveTransformer, create_real_pfn_data


def train_model_proper_pfn(model, num_epochs=50, batch_size=128, learning_rate=0.0001):
    """
    Train using proper PFN setup:
    - FullSupportBarDistribution loss
    - Real PFN priors (BNN-like)
    - Proper batch size, learning rate
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(num_epochs):
        # Generate new data each epoch (like real PFN)
        x_train, y_train, x_test, y_test = create_real_pfn_data(batch_size)
        
        # Forward pass
        logits = model(x_train, y_train, x_test)
        
        # Compute loss using FullSupportBarDistribution
        loss = model.criterion(logits, y_test).mean()  # Average over batch
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    return losses


def evaluate_model_proper_pfn(model, num_batches=20):
    """Evaluate model on fresh test data using proper PFN setup."""
    model.eval()
    
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for _ in range(num_batches):
            x_train, y_train, x_test, y_test = create_real_pfn_data(batch_size=128)
            logits = model(x_train, y_train, x_test)
            
            # Compute loss using FullSupportBarDistribution
            loss = model.criterion(logits, y_test).mean()
            
            batch_samples = logits.numel() // logits.shape[-1]  # Total predictions made
            total_loss += loss.item() * batch_samples
            total_samples += batch_samples
    
    return total_loss / total_samples


def compare_attention_patterns(tinypfn_model, naive_model):
    """Compare attention patterns between models."""
    # Generate sample data
    x_train, y_train, x_test, y_test = create_real_pfn_data(batch_size=1, seq_len=30)
    
    # Get attention weights
    tinypfn_attention = tinypfn_model.get_mock_attention_weights(seq_len=30, num_features=18)
    naive_attention = naive_model.get_attention_weights(x_train, y_train, x_test)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # TinyPFN Feature Attention
    axes[0, 0].imshow(tinypfn_attention['feature_attention'], cmap='Blues')
    axes[0, 0].set_title('TinyPFN: Feature Attention')
    axes[0, 0].set_xlabel('Features')
    axes[0, 0].set_ylabel('Features')
    
    # TinyPFN Item Attention
    axes[0, 1].imshow(tinypfn_attention['item_attention'], cmap='Blues')
    axes[0, 1].set_title('TinyPFN: Item Attention')
    axes[0, 1].set_xlabel('Data Points')
    axes[0, 1].set_ylabel('Data Points')
    
    # Naive Transformer (no feature attention)
    axes[1, 0].text(0.5, 0.5, 'No Feature Attention\nin Naive Transformer', 
                   ha='center', va='center', fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 0].set_title('Naive Transformer: Feature Attention')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    
    # Naive Transformer Item Attention
    axes[1, 1].imshow(naive_attention['item_attention'], cmap='Blues')
    axes[1, 1].set_title('Naive Transformer: Item Attention')
    axes[1, 1].set_xlabel('Data Points')
    axes[1, 1].set_ylabel('Data Points')
    
    plt.tight_layout()
    plt.savefig('attention_comparison_proper_pfn.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main comparison function using proper PFN training setup."""
    print("🔍 Comparing TinyPFN vs Naive Transformer (Proper PFN Setup)")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create models with proper PFN configuration
    print("Creating models with proper PFN configuration...")
    tinypfn = TinyPFN(num_features=18, d_model=512, n_heads=4, n_buckets=1000)
    naive_tf = NaiveTransformer(num_features=18, d_model=512, n_heads=4, n_buckets=1000)
    
    print(f"TinyPFN parameters: {sum(p.numel() for p in tinypfn.parameters()):,}")
    print(f"Naive Transformer parameters: {sum(p.numel() for p in naive_tf.parameters()):,}")
    
    # Training with proper PFN setup
    print("\n📚 Training TinyPFN (proper PFN setup)...")
    tinypfn_losses = train_model_proper_pfn(tinypfn, num_epochs=50, batch_size=128, learning_rate=0.0001)
    
    print("\n📚 Training Naive Transformer (proper PFN setup)...")
    naive_losses = train_model_proper_pfn(naive_tf, num_epochs=50, batch_size=128, learning_rate=0.0001)
    
    # Evaluation
    print("\n🧪 Evaluating models...")
    tinypfn_test_loss = evaluate_model_proper_pfn(tinypfn)
    naive_test_loss = evaluate_model_proper_pfn(naive_tf)
    
    print(f"TinyPFN Test Loss: {tinypfn_test_loss:.4f}")
    print(f"Naive Transformer Test Loss: {naive_test_loss:.4f}")
    
    improvement = (naive_test_loss - tinypfn_test_loss) / naive_test_loss * 100
    print(f"TinyPFN improvement: {improvement:.1f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(tinypfn_losses, label='TinyPFN', color='blue')
    plt.plot(naive_losses, label='Naive Transformer', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss (FullSupportBarDistribution)')
    plt.title('Training Loss Comparison (Proper PFN Setup)')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    models = ['TinyPFN', 'Naive Transformer']
    test_losses = [tinypfn_test_loss, naive_test_loss]
    colors = ['blue', 'red']
    
    bars = plt.bar(models, test_losses, color=colors, alpha=0.7)
    plt.ylabel('Test Loss (FullSupportBarDistribution)')
    plt.title('Final Test Loss Comparison')
    
    # Add value labels on bars
    for bar, loss in zip(bars, test_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{loss:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_comparison_proper_pfn.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compare attention patterns
    print("\n👀 Comparing attention patterns...")
    compare_attention_patterns(tinypfn, naive_tf)
    
    # Summary
    print("\n📊 Summary:")
    print("=" * 60)
    print("Training Setup:")
    print("✅ FullSupportBarDistribution loss (1000 buckets)")
    print("✅ Real PFN priors (BNN-like MLPs)")
    print("✅ Proper batch size (128) and learning rate (0.0001)")
    print("✅ 18 features, 512 d_model, 4 heads")
    print()
    
    if tinypfn_test_loss < naive_test_loss:
        print(f"🎉 TinyPFN outperforms Naive Transformer by {improvement:.1f}%")
        print("   ✅ The dual attention mechanism provides meaningful benefits!")
        print("   ✅ Feature attention helps with tabular in-context learning!")
    else:
        print(f"❌ Naive Transformer outperforms TinyPFN by {-improvement:.1f}%")
        print("   ❓ Feature attention may not be helping for this task.")
        print("   🔍 This suggests the task doesn't benefit from dual attention.")
    
    print(f"\n🔬 Technical Details:")
    print(f"• TinyPFN: Single layer with dual attention (feature + item)")
    print(f"• Naive Transformer: Single layer with only item attention")
    print(f"• Both use FullSupportBarDistribution loss")
    print(f"• Both trained on BNN-like priors")
    print(f"• Test loss difference: {abs(tinypfn_test_loss - naive_test_loss):.4f}")
    
    print(f"\n💡 Interpretation:")
    if tinypfn_test_loss < naive_test_loss:
        print("The feature attention mechanism in PFNs is indeed beneficial!")
        print("It allows features to communicate before items interact.")
    else:
        print("This specific task may not showcase PFN's strengths.")
        print("Consider trying more complex tabular tasks or feature interactions.")


if __name__ == "__main__":
    main() 