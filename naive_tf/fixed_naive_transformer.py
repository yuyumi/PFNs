"""
Fixed Naive Transformer with Proper Cell-Level Representations

This implementation uses the same cell-level representation as the fixed TinyPFN,
but with standard sequential attention instead of dual attention.

Key changes:
- Input shape: [batch, seq, features+1, d_model] (cell-level)
- Flattens to sequential tokens: [batch, seq*(features+1), d_model]
- Standard transformer attention across all tokens
- Still single layer for comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from pfns.model.bar_distribution import FullSupportBarDistribution, get_bucket_borders


class FixedNaiveTransformer(nn.Module):
    """
    Naive transformer with proper cell-level representations.
    
    Each cell in the table gets its own token/representation:
    - Row 1: [repr_x1, repr_x2, repr_y]
    - Row 2: [repr_x1, repr_x2, repr_y]  
    - etc.
    
    Then flattened to sequential tokens for standard attention:
    [repr_x1_row1, repr_x2_row1, repr_y_row1, repr_x1_row2, repr_x2_row2, repr_y_row2, ...]
    """
    
    def __init__(
        self,
        num_features,
        d_model=256,
        n_heads=4,
        dropout=0.1,
        max_seq_len=60,
        n_buckets=1000
    ):
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.n_buckets = n_buckets
        self.num_cells = num_features + 1  # features + target
        
        # Individual cell encoders - each cell type gets its own encoder
        self.feature_encoders = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])
        self.target_encoder = nn.Linear(1, d_model)
        
        # Standard transformer attention (sequential)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Positional encoding for sequential tokens
        max_tokens = max_seq_len * self.num_cells
        self.positional_encoding = nn.Parameter(torch.randn(max_tokens, d_model) * 0.1)
        
        # Create bucket borders for loss
        bucket_borders = get_bucket_borders(
            num_outputs=n_buckets,
            full_range=(-3.0, 3.0)
        )
        self.criterion = FullSupportBarDistribution(bucket_borders)
        
        # Output projection (only for target cells)
        self.output_projection = nn.Linear(d_model, n_buckets)
        
    def encode_table(self, x, y):
        """
        Encode table into cell-level representations.
        
        Input:
        - x: [batch, seq, num_features]
        - y: [batch, seq, 1]
        
        Output:
        - encoded: [batch, seq, num_cells, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Encode each cell separately
        encoded_cells = []
        
        # Encode feature cells
        for feat_idx in range(self.num_features):
            feat_values = x[:, :, feat_idx:feat_idx+1]  # [batch, seq, 1]
            encoded_feat = self.feature_encoders[feat_idx](feat_values)  # [batch, seq, d_model]
            encoded_cells.append(encoded_feat)
        
        # Encode target cells (handle NaN for test positions)
        y_clean = torch.where(torch.isnan(y), torch.zeros_like(y), y)
        encoded_target = self.target_encoder(y_clean)
        
        # Zero out NaN positions
        nan_mask = torch.isnan(y)
        encoded_target = torch.where(
            nan_mask.expand_as(encoded_target), 
            torch.zeros_like(encoded_target), 
            encoded_target
        )
        encoded_cells.append(encoded_target)
        
        # Stack into [batch, seq, num_cells, d_model]
        encoded = torch.stack(encoded_cells, dim=2)
        
        return encoded
        
    def flatten_for_attention(self, x):
        """
        Flatten cell-level representations to sequential tokens.
        
        Input: [batch, seq, num_cells, d_model]
        Output: [batch, seq*num_cells, d_model]
        """
        batch_size, seq_len, num_cells, d_model = x.shape
        
        # Flatten to sequential tokens
        x_flat = x.view(batch_size, seq_len * num_cells, d_model)
        
        # Add positional encoding
        seq_tokens = seq_len * num_cells
        pos_enc = self.positional_encoding[:seq_tokens].unsqueeze(0)
        x_flat = x_flat + pos_enc
        
        return x_flat
    
    def unflatten_from_attention(self, x_flat, seq_len):
        """
        Unflatten sequential tokens back to cell-level representations.
        
        Input: [batch, seq*num_cells, d_model]
        Output: [batch, seq, num_cells, d_model]
        """
        batch_size, _, d_model = x_flat.shape
        return x_flat.view(batch_size, seq_len, self.num_cells, d_model)
    
    def forward(self, x_train, y_train, x_test):
        """
        Forward pass with sequential attention over all cells.
        """
        batch_size, train_len, num_features = x_train.shape
        test_len = x_test.shape[1]
        
        # Combine training and test data
        x_combined = torch.cat([x_train, x_test], dim=1)
        
        # Create targets with NaN for test positions
        y_test_placeholder = torch.full(
            (batch_size, test_len, 1),
            float('nan'),
            device=x_test.device,
            dtype=x_test.dtype
        )
        y_combined = torch.cat([y_train, y_test_placeholder], dim=1)
        
        # Encode table into cell-level representations
        encoded = self.encode_table(x_combined, y_combined)  # [batch, seq, num_cells, d_model]
        
        # Flatten for sequential attention
        seq_len = encoded.shape[1]
        encoded_flat = self.flatten_for_attention(encoded)  # [batch, seq*num_cells, d_model]
        
        # Apply standard transformer attention
        attn_output, _ = self.attention(encoded_flat, encoded_flat, encoded_flat)
        
        # Residual connection + layer norm
        encoded_flat = self.norm1(encoded_flat + attn_output)
        
        # Apply MLP
        mlp_output = self.mlp(encoded_flat)
        
        # Residual + norm
        encoded_flat = self.norm2(encoded_flat + mlp_output)
        
        # Unflatten back to cell-level
        encoded = self.unflatten_from_attention(encoded_flat, seq_len)
        
        # Extract test predictions from target cells only
        test_encoded = encoded[:, train_len:, -1, :]  # [batch, test_len, d_model] (target cell only)
        
        # Project to output space
        logits = self.output_projection(test_encoded)
        
        return logits
    
    def predict_mean(self, logits):
        """Get mean prediction from distributional output."""
        return self.criterion.mean(logits)
    
    def compute_loss(self, logits, targets):
        """Compute loss using the bar distribution."""
        return self.criterion(logits, targets).mean()


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


def test_fixed_naive_transformer():
    """Test the fixed naive transformer implementation."""
    print("Testing Fixed Naive Transformer with cell-level representations...")
    
    # Create model
    model = FixedNaiveTransformer(num_features=10, d_model=256, n_heads=4)
    
    # Create test data
    x_train, y_train, x_test, y_test = create_ridge_regression_data(batch_size=4, seq_len=20)
    
    print(f"Input shapes: x_train {x_train.shape}, y_train {y_train.shape}, x_test {x_test.shape}")
    
    # Test forward pass
    logits = model(x_train, y_train, x_test)
    print(f"Output logits shape: {logits.shape}")
    
    # Test mean prediction
    mean_pred = model.predict_mean(logits)
    print(f"Mean predictions shape: {mean_pred.shape}")
    
    print("✓ Fixed Naive Transformer test passed!")
    
    # Show internal representations
    with torch.no_grad():
        x_combined = torch.cat([x_train, x_test], dim=1)
        y_test_placeholder = torch.full(
            (4, x_test.shape[1], 1), float('nan'),
            device=x_test.device, dtype=x_test.dtype
        )
        y_combined = torch.cat([y_train, y_test_placeholder], dim=1)
        
        encoded = model.encode_table(x_combined, y_combined)
        print(f"Cell-level encoded shape: {encoded.shape}")
        
        encoded_flat = model.flatten_for_attention(encoded)
        print(f"Flattened sequential shape: {encoded_flat.shape}")
        print(f"Sequential tokens: {encoded_flat.shape[1]} = {encoded.shape[1]} rows × {encoded.shape[2]} cells")


if __name__ == "__main__":
    test_fixed_naive_transformer() 