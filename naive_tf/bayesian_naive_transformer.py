"""
Bayesian Naive Transformer with Prior-Data Negative Log-Likelihood Loss

Uses the same Bayesian loss as BayesianTinyPFN but with sequential attention
instead of dual attention, for fair comparison.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class BayesianNaiveTransformer(nn.Module):
    """
    Naive transformer with proper cell-level representations and Bayesian PD-NLL loss.
    
    Each cell in the table gets its own token/representation:
    - Row 1: [repr_x1, repr_x2, repr_y]
    - Row 2: [repr_x1, repr_x2, repr_y]  
    - etc.
    
    Then flattened to sequential tokens for standard attention:
    [repr_x1_row1, repr_x2_row1, repr_y_row1, repr_x1_row2, repr_x2_row2, repr_y_row2, ...]
    
    Output: Direct regression (no distributional modeling)
    Loss: Simple MSE / Gaussian NLL
    """
    
    def __init__(
        self,
        num_features,
        d_model=256,
        n_heads=4,
        dropout=0.1,
        max_seq_len=60
    ):
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
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
        
        # Output projection (mean and log variance for Gaussian)
        self.mean_head = nn.Linear(d_model, 1)
        self.log_var_head = nn.Linear(d_model, 1)
        
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
        Returns mean and log variance for Gaussian likelihood.
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
        
        # Project to mean and log variance
        mean = self.mean_head(test_encoded).squeeze(-1)  # [batch, test_len]
        log_var = self.log_var_head(test_encoded).squeeze(-1)  # [batch, test_len]
        
        return mean, log_var
    
    def predict(self, x_train, y_train, x_test):
        """Get point predictions (mean)."""
        mean, _ = self.forward(x_train, y_train, x_test)
        return mean
    
    def compute_loss(self, x_train, y_train, x_test, y_test):
        """
        Compute Bayesian PD-NLL loss.
        
        ℓ_θ = E_{D∪{x,y}∼p(D)}[− log q_θ(y|x, D)]
        
        For Gaussian likelihood: -log N(y; μ, σ²) = 0.5 * (log(2π) + log(σ²) + (y-μ)²/σ²)
        """
        mean, log_var = self.forward(x_train, y_train, x_test)
        
        # Gaussian negative log-likelihood
        var = torch.exp(log_var)
        nll = 0.5 * (torch.log(2 * torch.pi * var) + (y_test.squeeze(-1) - mean) ** 2 / var)
        
        return nll.mean()


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


def test_bayesian_naive_transformer():
    """Test the Bayesian naive transformer implementation."""
    print("Testing Bayesian Naive Transformer with PD-NLL loss...")
    
    # Create model
    model = BayesianNaiveTransformer(num_features=10, d_model=256, n_heads=4)
    
    # Create test data
    x_train, y_train, x_test, y_test = create_ridge_regression_data(batch_size=4, seq_len=20)
    
    print(f"Input shapes: x_train {x_train.shape}, y_train {y_train.shape}, x_test {x_test.shape}")
    
    # Test forward pass
    mean, log_var = model(x_train, y_train, x_test)
    print(f"Mean predictions shape: {mean.shape}")
    print(f"Log variance shape: {log_var.shape}")
    
    # Test loss computation
    loss = model.compute_loss(x_train, y_train, x_test, y_test)
    print(f"PD-NLL loss: {loss.item():.4f}")
    
    # Test prediction
    pred = model.predict(x_train, y_train, x_test)
    print(f"Point predictions shape: {pred.shape}")
    
    print("✓ Bayesian Naive Transformer test passed!")


if __name__ == "__main__":
    test_bayesian_naive_transformer() 