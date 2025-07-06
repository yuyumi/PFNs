"""
TinyPFN: Simplified Implementation with Attention Visualization

This version implements the dual attention mechanism without importing the original PFN codebase,
making it compatible with older Python versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math


class SimpleDualAttentionLayer(nn.Module):
    """
    A simplified implementation of the dual attention mechanism.
    
    This follows the same pattern as the real PFN PerFeatureLayer:
    Feature Attention → Item Attention → MLP
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Feature attention: features attend to each other within data points
        self.feature_attention = nn.MultiHeadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Item attention: data points attend to each other across sequence
        self.item_attention = nn.MultiHeadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP (feed-forward network)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Store attention weights for visualization
        self.feature_attention_weights = None
        self.item_attention_weights = None
        
    def forward(self, x):
        """
        Forward pass implementing Feature → Item → MLP pipeline.
        
        Args:
            x: Input tensor (batch_size, seq_len, num_features, d_model)
            
        Returns:
            Output tensor of same shape
        """
        batch_size, seq_len, num_features, d_model = x.shape
        
        # 1. Feature Attention: Features attend to each other within data points
        # Reshape to (batch_size * seq_len, num_features, d_model)
        x_flat = x.view(batch_size * seq_len, num_features, d_model)
        
        # Apply feature attention
        feature_out, feature_weights = self.feature_attention(
            x_flat, x_flat, x_flat, average_attn_weights=True
        )
        
        # Store attention weights for visualization
        self.feature_attention_weights = feature_weights.detach()
        
        # Add & norm
        x_flat = self.norm1(x_flat + feature_out)
        
        # Reshape back to (batch_size, seq_len, num_features, d_model)
        x = x_flat.view(batch_size, seq_len, num_features, d_model)
        
        # 2. Item Attention: Data points attend to each other across sequence
        # Reshape to (batch_size * num_features, seq_len, d_model)
        x_transposed = x.transpose(1, 2).contiguous().view(
            batch_size * num_features, seq_len, d_model
        )
        
        # Apply item attention
        item_out, item_weights = self.item_attention(
            x_transposed, x_transposed, x_transposed, average_attn_weights=True
        )
        
        # Store attention weights for visualization
        self.item_attention_weights = item_weights.detach()
        
        # Add & norm
        x_transposed = self.norm2(x_transposed + item_out)
        
        # Reshape back to (batch_size, seq_len, num_features, d_model)
        x = x_transposed.view(batch_size, num_features, seq_len, d_model).transpose(1, 2)
        
        # 3. MLP: Feed-forward processing
        # Apply MLP to each position
        x_flat = x.view(batch_size * seq_len * num_features, d_model)
        mlp_out = self.mlp(x_flat)
        mlp_out = self.norm3(x_flat + mlp_out)  # Residual connection
        
        # Reshape back to original shape
        x = mlp_out.view(batch_size, seq_len, num_features, d_model)
        
        return x


class TinyPFNSimple(nn.Module):
    """
    Simplified TinyPFN implementation with attention visualization.
    
    Compatible with older Python versions by avoiding the original PFN imports.
    """
    
    def __init__(self, num_features, d_model=64, n_heads=4, dropout=0.1, max_seq_len=100):
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        # Simple feature encoder
        self.feature_encoder = nn.Linear(1, d_model)
        
        # Simple target encoder
        self.target_encoder = nn.Linear(1, d_model)
        
        # Single dual attention layer
        self.dual_attention_layer = SimpleDualAttentionLayer(d_model, n_heads, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.1)
        
    def forward(self, x_train, y_train, x_test):
        """Forward pass for in-context learning."""
        batch_size, train_len, num_features = x_train.shape
        test_len = x_test.shape[1]
        
        # Combine training and test data
        x_combined = torch.cat([x_train, x_test], dim=1)
        
        # Create targets with zeros for test positions
        y_test_placeholder = torch.zeros(batch_size, test_len, 1, device=x_test.device)
        y_combined = torch.cat([y_train, y_test_placeholder], dim=1)
        
        return self._forward_combined(x_combined, y_combined, train_len)
    
    def _forward_combined(self, x, y, train_len):
        """Internal forward pass."""
        batch_size, seq_len, num_features = x.shape
        
        # Encode features
        x_encoded = self._encode_features(x)  # (batch, seq_len, num_features, d_model)
        
        # Encode targets
        y_encoded = self._encode_targets(y)  # (batch, seq_len, d_model)
        
        # Add target as an additional "feature"
        y_expanded = y_encoded.unsqueeze(2)  # (batch, seq_len, 1, d_model)
        combined = torch.cat([x_encoded, y_expanded], dim=2)  # (batch, seq_len, num_features+1, d_model)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:seq_len].unsqueeze(0).unsqueeze(2)
        combined = combined + pos_encoding
        
        # Apply dual attention layer
        transformed = self.dual_attention_layer(combined)
        
        # Extract predictions from target feature (last dimension)
        test_representations = transformed[:, train_len:, -1, :]  # (batch, test_len, d_model)
        predictions = self.output_projection(test_representations)
        
        return predictions
    
    def _encode_features(self, x):
        """Encode features."""
        batch_size, seq_len, num_features = x.shape
        x_flat = x.view(-1, 1)  # Treat each feature value separately
        encoded = self.feature_encoder(x_flat)
        return encoded.view(batch_size, seq_len, num_features, self.d_model)
    
    def _encode_targets(self, y):
        """Encode targets."""
        batch_size, seq_len, _ = y.shape
        y_flat = y.view(-1, 1)
        encoded = self.target_encoder(y_flat)
        return encoded.view(batch_size, seq_len, self.d_model)
    
    def get_attention_weights(self):
        """Get attention weights for visualization."""
        return {
            'feature_attention': self.dual_attention_layer.feature_attention_weights,
            'item_attention': self.dual_attention_layer.item_attention_weights
        }


def visualize_attention_heatmaps(model, x_train, y_train, x_test, y_test):
    """Visualize attention heatmaps."""
    print("📊 Generating attention visualizations...")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(x_train, y_train, x_test)
    
    # Get attention weights
    attention_weights = model.get_attention_weights()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TinyPFN Dual Attention Visualization', fontsize=16)
    
    # Feature attention heatmap
    if attention_weights['feature_attention'] is not None:
        # Average across heads and batches for visualization
        feature_attn = attention_weights['feature_attention'][0].cpu().numpy()
        im1 = axes[0, 0].imshow(feature_attn, cmap='Blues', aspect='auto')
        axes[0, 0].set_title('Feature Attention\n(Features attend to each other)')
        axes[0, 0].set_xlabel('Key Features')
        axes[0, 0].set_ylabel('Query Features')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Add text annotations for clarity
        for i in range(min(feature_attn.shape[0], 5)):
            for j in range(min(feature_attn.shape[1], 5)):
                axes[0, 0].text(j, i, f'{feature_attn[i, j]:.2f}', 
                              ha='center', va='center', fontsize=8)
    
    # Item attention heatmap
    if attention_weights['item_attention'] is not None:
        # Average across heads and batches for visualization
        item_attn = attention_weights['item_attention'][0].cpu().numpy()
        im2 = axes[0, 1].imshow(item_attn, cmap='Reds', aspect='auto')
        axes[0, 1].set_title('Item Attention\n(Data points attend to each other)')
        axes[0, 1].set_xlabel('Key Items (Training Context)')
        axes[0, 1].set_ylabel('Query Items')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Add text annotations for clarity  
        for i in range(min(item_attn.shape[0], 5)):
            for j in range(min(item_attn.shape[1], 5)):
                axes[0, 1].text(j, i, f'{item_attn[i, j]:.2f}', 
                              ha='center', va='center', fontsize=8)
    
    # Training data
    train_targets = y_train[0].cpu().numpy().flatten()
    axes[1, 0].bar(range(len(train_targets)), train_targets, alpha=0.7, color='blue')
    axes[1, 0].set_title('Training Data (Context)')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Target Value')
    axes[1, 0].set_ylim([0, 1])
    
    # Predictions vs actual
    pred_values = predictions[0].cpu().numpy().flatten()
    test_targets = y_test[0].cpu().numpy().flatten()
    
    x_pos = range(len(test_targets))
    width = 0.35
    axes[1, 1].bar([x - width/2 for x in x_pos], test_targets, width, 
                   label='Actual', alpha=0.7, color='red')
    axes[1, 1].bar([x + width/2 for x in x_pos], pred_values, width, 
                   label='Predicted', alpha=0.7, color='blue')
    axes[1, 1].set_title('Test: Predictions vs Actual')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Target Value')
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()
    
    return fig


def create_synthetic_data(batch_size=4, train_len=8, test_len=4, num_features=4):
    """Create synthetic data with clear patterns."""
    # Training data
    x_train = torch.randn(batch_size, train_len, num_features)
    y_train = (x_train.sum(dim=-1, keepdim=True) > 0).float()
    
    # Test data  
    x_test = torch.randn(batch_size, test_len, num_features)
    y_test = (x_test.sum(dim=-1, keepdim=True) > 0).float()
    
    return x_train, y_train, x_test, y_test


def test_tiny_pfn_simple():
    """Test the simplified TinyPFN implementation."""
    print("🚀 Testing TinyPFN Simplified Implementation")
    print("=" * 60)
    
    # Create model
    model = TinyPFNSimple(num_features=4, d_model=32, n_heads=4)
    
    print(f"📊 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create test data
    x_train, y_train, x_test, y_test = create_synthetic_data()
    
    print(f"📦 Data shapes:")
    print(f"   Training: {x_train.shape} -> {y_train.shape}")
    print(f"   Test: {x_test.shape} -> {y_test.shape}")
    
    # Test forward pass
    try:
        predictions = model(x_train, y_train, x_test)
        
        print(f"✅ Forward pass successful!")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        
        # Calculate accuracy
        accuracy = ((predictions > 0.5) == y_test).float().mean()
        print(f"   Accuracy: {accuracy:.3f}")
        
        # Visualize attention
        visualize_attention_heatmaps(model, x_train, y_train, x_test, y_test)
        
        print(f"✅ Visualization complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏗️  Architecture:")
    print(f"   ✓ Feature Attention → Item Attention → MLP")  
    print(f"   ✓ Single layer implementation")
    print(f"   ✓ Attention visualization")
    print(f"   ✓ Compatible with older Python versions")


if __name__ == "__main__":
    test_tiny_pfn_simple() 