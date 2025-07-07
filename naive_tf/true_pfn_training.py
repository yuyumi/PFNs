"""
True PFN Training Protocol

This implements the exact training protocol from the PFN paper:
ℓ_θ = E_{D∪{x,y}∼p(D)}[− log q_θ(y|x, D)]

Key differences from our previous approach:
1. Single holdout prediction per dataset (not multiple test points)
2. Continuous NLL loss (not discretized bar distribution)
3. Fresh dataset generation every step
4. Proper prior sampling protocol
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Import models but we'll modify their loss functions
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tiny_pfn'))
from tiny_pfn import TinyPFN
from naive_transformer import NaiveTransformer


class SimplePFN(nn.Module):
    """
    Simplified PFN that uses continuous NLL loss instead of FullSupportBarDistribution.
    This follows the exact PFN paper protocol.
    """
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Replace the complex bar distribution with simple Gaussian output
        # Remove the bar distribution components
        self.base_model.criterion = None
        
        # Add simple Gaussian output (mean and log_std)
        self.mean_head = nn.Linear(base_model.d_model, 1)
        self.log_std_head = nn.Linear(base_model.d_model, 1)
        
    def forward(self, x_context, y_context, x_target):
        """
        Forward pass following PFN protocol:
        - x_context, y_context: context dataset D
        - x_target: single target input x
        """
        batch_size = x_context.shape[0]
        context_len = x_context.shape[1]
        
        # Combine context and target
        x_combined = torch.cat([x_context, x_target.unsqueeze(1)], dim=1)
        
        # Create target placeholders (NaN for target position)
        y_target_placeholder = torch.full(
            (batch_size, 1, 1), float('nan'), 
            device=x_target.device, dtype=x_target.dtype
        )
        y_combined = torch.cat([y_context, y_target_placeholder], dim=1)
        
        # Get representations BEFORE the output projection layer
        # We need to manually run the base model's forward pass without the final projection
        
        seq_len = x_combined.shape[1]
        
        # Handle different model types
        if hasattr(self.base_model, 'dual_attention_layer'):  # TinyPFN
            # Encode features and targets
            x_encoded = self.base_model._encode_features(x_combined)
            y_encoded = self.base_model._encode_targets(y_combined)
            
            # Combine features and targets
            combined = x_encoded + y_encoded
            
            # Add positional encoding
            pos_encoding = self.base_model.positional_encoding[:seq_len].unsqueeze(0).unsqueeze(2)
            combined = combined + pos_encoding
            
            # Apply dual attention layer (this gives us the representation we want)
            transformed = self.base_model.dual_attention_layer(
                combined,
                single_eval_pos=context_len
            )
            
            # Extract target representation (last position, feature 0)
            target_repr = transformed[:, -1, 0, :]  # [batch_size, d_model]
            
        else:  # NaiveTransformer
            # Encode features and targets
            x_encoded = self.base_model.feature_projection(x_combined)
            y_cleaned = torch.where(torch.isnan(y_combined), torch.zeros_like(y_combined), y_combined)
            y_encoded = self.base_model.target_projection(y_cleaned)
            
            # Combine features and targets (simple addition)
            combined = x_encoded + y_encoded
            
            # Add positional encoding
            pos_encoding = self.base_model.positional_encoding[:seq_len].unsqueeze(0)
            combined = combined + pos_encoding
            
            # Apply item attention
            attn_output, _ = self.base_model.item_attention(combined, combined, combined)
            combined = self.base_model.norm1(combined + attn_output)
            
            # Apply MLP
            mlp_output = self.base_model.mlp(combined)
            combined = self.base_model.norm2(combined + mlp_output)
            
            # Extract target representation (last position)
            target_repr = combined[:, -1, :]  # [batch_size, d_model]
        
        # Predict mean and log_std
        mean = self.mean_head(target_repr)  # [batch_size, 1]
        log_std = self.log_std_head(target_repr)  # [batch_size, 1]
        
        return mean, log_std
    
    def nll_loss(self, mean, log_std, y_target):
        """
        Continuous NLL loss as in PFN paper: -log q_θ(y|x, D)
        """
        std = torch.exp(log_std)
        
        # Gaussian NLL: -log p(y|μ,σ) = 0.5 * ((y-μ)/σ)² + log(σ) + 0.5*log(2π)
        nll = 0.5 * ((y_target - mean) / std)**2 + log_std + 0.5 * np.log(2 * np.pi)
        
        return nll.mean()


def sample_ridge_prior(batch_size, context_size, num_features=10):
    """
    Sample from ridge regression prior exactly as in PFN paper protocol.
    
    Returns:
    - x_context: [batch_size, context_size, num_features]
    - y_context: [batch_size, context_size, 1]  
    - x_target: [batch_size, num_features]
    - y_target: [batch_size, 1]
    """
    # Sample ridge regression weights from prior
    true_weights = torch.randn(batch_size, num_features, 1) * 0.5
    
    # Sample context points
    x_context = torch.randn(batch_size, context_size, num_features)
    y_context = torch.bmm(x_context, true_weights) + torch.randn(batch_size, context_size, 1) * 0.1
    
    # Sample single target point from same prior
    x_target = torch.randn(batch_size, num_features)
    y_target = torch.mm(x_target, true_weights.squeeze(-1).T).diag().unsqueeze(1)
    y_target = y_target + torch.randn(batch_size, 1) * 0.1
    
    return x_context, y_context, x_target, y_target


def train_true_pfn_protocol(model, num_steps=1000, batch_size=32, learning_rate=0.001):
    """
    Train using the exact PFN protocol from the paper.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for step in range(num_steps):
        # Sample fresh dataset every step (key PFN principle!)
        context_size = np.random.randint(5, 25)  # Random context size
        
        x_context, y_context, x_target, y_target = sample_ridge_prior(
            batch_size, context_size, num_features=10
        )
        
        # Forward pass
        mean, log_std = model(x_context, y_context, x_target)
        
        # Compute PFN loss: -log q_θ(y|x, D)
        loss = model.nll_loss(mean, log_std, y_target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")
    
    return losses


def evaluate_true_pfn(model, num_tests=100, context_sizes=[5, 10, 20]):
    """
    Evaluate using true PFN protocol.
    """
    model.eval()
    results = {}
    
    for context_size in context_sizes:
        print(f"Testing context size: {context_size}")
        
        mse_scores = []
        
        for _ in range(num_tests):
            x_context, y_context, x_target, y_target = sample_ridge_prior(
                batch_size=1, context_size=context_size, num_features=10
            )
            
            with torch.no_grad():
                mean, log_std = model(x_context, y_context, x_target)
                
                # Use mean as prediction
                mse = ((mean - y_target)**2).item()
                mse_scores.append(mse)
        
        results[context_size] = {
            'mean_mse': np.mean(mse_scores),
            'std_mse': np.std(mse_scores)
        }
        
        print(f"  MSE: {results[context_size]['mean_mse']:.4f} ± {results[context_size]['std_mse']:.4f}")
    
    return results


def compare_to_sklearn_ridge_single(num_tests=100, context_sizes=[5, 10, 20]):
    """
    Compare to sklearn Ridge regression on single target prediction.
    """
    results = {}
    
    for context_size in context_sizes:
        print(f"Testing sklearn Ridge with context size: {context_size}")
        
        mse_scores = []
        
        for _ in range(num_tests):
            x_context, y_context, x_target, y_target = sample_ridge_prior(
                batch_size=1, context_size=context_size, num_features=10
            )
            
            # Convert to numpy
            x_context_np = x_context.squeeze().numpy()
            y_context_np = y_context.squeeze().numpy()
            x_target_np = x_target.squeeze().numpy()
            y_target_np = y_target.squeeze().numpy()
            
            # Fit ridge regression
            ridge = Ridge(alpha=0.1)
            ridge.fit(x_context_np, y_context_np)
            
            # Predict single target
            prediction = ridge.predict(x_target_np.reshape(1, -1))[0]
            
            # Calculate MSE
            mse = (prediction - y_target_np)**2
            mse_scores.append(mse)
        
        results[context_size] = {
            'mean_mse': np.mean(mse_scores),
            'std_mse': np.std(mse_scores)
        }
        
        print(f"  MSE: {results[context_size]['mean_mse']:.4f} ± {results[context_size]['std_mse']:.4f}")
    
    return results


def main():
    """
    Main function implementing true PFN training protocol.
    """
    print("🧪 True PFN Training Protocol Test")
    print("=" * 60)
    print("Following exact paper protocol:")
    print("ℓ_θ = E_{D∪{x,y}∼p(D)}[− log q_θ(y|x, D)]")
    print()
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create base models
    print("Creating models...")
    tinypfn_base = TinyPFN(num_features=10, d_model=256, n_heads=4, max_seq_len=60)
    naive_base = NaiveTransformer(num_features=10, d_model=256, n_heads=4, max_seq_len=60)
    
    # Wrap with SimplePFN (continuous NLL loss)
    tinypfn_simple = SimplePFN(tinypfn_base)
    naive_simple = SimplePFN(naive_base)
    
    print(f"TinyPFN parameters: {sum(p.numel() for p in tinypfn_simple.parameters()):,}")
    print(f"Naive Transformer parameters: {sum(p.numel() for p in naive_simple.parameters()):,}")
    
    # Train using true PFN protocol
    print("\n📚 Training TinyPFN (true PFN protocol)...")
    tinypfn_losses = train_true_pfn_protocol(tinypfn_simple, num_steps=500, batch_size=32)
    
    print("\n📚 Training Naive Transformer (true PFN protocol)...")
    naive_losses = train_true_pfn_protocol(naive_simple, num_steps=500, batch_size=32)
    
    # Evaluate
    print("\n🎯 Evaluating TinyPFN...")
    tinypfn_results = evaluate_true_pfn(tinypfn_simple)
    
    print("\n🎯 Evaluating Naive Transformer...")
    naive_results = evaluate_true_pfn(naive_simple)
    
    print("\n🎯 Testing sklearn Ridge baseline...")
    sklearn_results = compare_to_sklearn_ridge_single()
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Training curves
    plt.subplot(1, 3, 1)
    plt.plot(tinypfn_losses, label='TinyPFN', color='blue')
    plt.plot(naive_losses, label='Naive Transformer', color='red')
    plt.xlabel('Training Steps')
    plt.ylabel('NLL Loss')
    plt.title('Training Loss (True PFN Protocol)')
    plt.legend()
    plt.yscale('log')
    
    # MSE comparison
    plt.subplot(1, 3, 2)
    context_sizes = [5, 10, 20]
    
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
    
    # Improvement over sklearn
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
    plt.savefig('true_pfn_protocol_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary
    print("\n📊 Summary (True PFN Protocol):")
    print("=" * 60)
    
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
    
    print(f"\n✅ Key Protocol Differences:")
    print(f"• Single holdout prediction per dataset (not multiple test points)")
    print(f"• Continuous NLL loss (not discretized bar distribution)")
    print(f"• Fresh dataset generation every step")
    print(f"• Proper PFN training protocol: ℓ_θ = E_{{D∪{{x,y}}∼p(D)}}[− log q_θ(y|x, D)]")


if __name__ == "__main__":
    main() 