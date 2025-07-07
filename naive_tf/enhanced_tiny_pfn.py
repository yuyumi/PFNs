"""
Enhanced TinyPFN with Real TabPFN Techniques

This implementation incorporates the actual sophisticated techniques that TabPFN uses:
- Multiquery attention (shared KV across heads)
- Learnable attention scale/temperature
- Zero initialization (layers start as identity)
- Attention init gain for proper initialization
- Optional second MLP between attention layers
- Feature positional embeddings

Still single-layer but much more sophisticated, using Bayesian PD-NLL loss.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class EnhancedMultiHeadAttention(nn.Module):
    """
    Enhanced multi-head attention with TabPFN's techniques:
    - Multiquery attention (shared KV across heads)
    - Learnable softmax scale/temperature
    - Proper initialization with gain
    - Zero initialization option
    """
    
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        multiquery=False,
        zero_init=True,
        init_gain=1.0,
        learnable_temperature=True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.multiquery = multiquery
        self.zero_init = zero_init
        self.init_gain = init_gain
        
        # Query projection (always full heads)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        
        if multiquery:
            # Shared KV across heads (like TabPFN's multiquery_item_attention)
            self.w_k = nn.Linear(d_model, self.d_k, bias=False)
            self.w_v = nn.Linear(d_model, self.d_k, bias=False)
            self.n_heads_kv = 1
        else:
            # Full multi-head KV
            self.w_k = nn.Linear(d_model, d_model, bias=False)
            self.w_v = nn.Linear(d_model, d_model, bias=False)
            self.n_heads_kv = n_heads
        
        # Output projection
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        
        # Learnable attention temperature (like TabPFN's softmax_scale)
        if learnable_temperature:
            self.softmax_scale = nn.Parameter(torch.ones(1) / math.sqrt(self.d_k))
        else:
            self.register_buffer('softmax_scale', torch.tensor(1.0 / math.sqrt(self.d_k)))
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with TabPFN's scheme."""
        # Standard initialization for input projections
        for module in [self.w_q, self.w_k, self.w_v]:
            std = math.sqrt(2.0 / float(self.n_heads * self.d_k + self.d_model)) * self.init_gain
            a = math.sqrt(3.0) * std
            nn.init.uniform_(module.weight, -a, a)
        
        # Zero initialization for output (if requested)
        if self.zero_init:
            nn.init.zeros_(self.w_out.weight)
        else:
            std = math.sqrt(2.0 / float(self.n_heads * self.d_k + self.d_model)) * self.init_gain
            a = math.sqrt(3.0) * std
            nn.init.uniform_(self.w_out.weight, -a, a)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V
        q = self.w_q(x)  # [batch, seq, d_model]
        k = self.w_k(x)  # [batch, seq, d_k] or [batch, seq, d_model]
        v = self.w_v(x)  # [batch, seq, d_k] or [batch, seq, d_model]
        
        # Reshape Q for multi-head
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch, heads, seq, d_k]
        
        if self.multiquery:
            # Shared KV: repeat across heads
            k = k.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # [batch, heads, seq, d_k]
            v = v.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # [batch, heads, seq, d_k]
        else:
            # Full multi-head KV
            k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch, heads, seq, d_k]
            v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch, heads, seq, d_k]
        
        # Scaled dot-product attention with learnable temperature
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale  # [batch, heads, seq, seq]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq, d_k]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_out(attn_output)
        
        return output


class EnhancedMLP(nn.Module):
    """
    Enhanced MLP with TabPFN's zero initialization option.
    """
    
    def __init__(self, d_model, d_ff, activation="gelu", dropout=0.1, zero_init=True):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Zero initialization for output layer (if requested)
        if zero_init:
            nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EnhancedTinyPFN(nn.Module):
    """
    Enhanced TinyPFN incorporating real TabPFN techniques:
    
    - Cell-level representations (each table cell gets its own token)
    - Dual attention with TabPFN enhancements:
      * Feature attention: multiquery for efficiency
      * Item attention: full multi-head for expressiveness
    - Learnable attention temperature
    - Zero initialization (layers start as identity)
    - Optional second MLP between attention layers
    - Feature positional embeddings
    - Bayesian PD-NLL loss
    """
    
    def __init__(
        self,
        num_features,
        d_model=256,
        n_heads=4,
        dropout=0.1,
        max_seq_len=60,
        multiquery_feature_attention=True,
        multiquery_item_attention=False,
        zero_init=True,
        init_gain=1.0,
        second_mlp=False,
        feature_pos_embedding="learned",
        activation="gelu"
    ):
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.num_cells = num_features + 1  # features + target
        self.second_mlp = second_mlp
        
        # Individual cell encoders - each cell type gets its own encoder
        self.feature_encoders = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])
        self.target_encoder = nn.Linear(1, d_model)
        
        # Feature positional embeddings (like TabPFN)
        if feature_pos_embedding == "learned":
            self.cell_positional = nn.Parameter(torch.randn(self.num_cells, d_model) * 0.1)
        elif feature_pos_embedding == "normal_rand_vec":
            self.register_buffer('cell_positional', torch.randn(self.num_cells, d_model) * 0.1)
        else:
            self.cell_positional = None
        
        # Row positional embeddings
        self.row_positional = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.1)
        
        # Enhanced dual attention with TabPFN techniques
        self.feature_attention = EnhancedMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            multiquery=multiquery_feature_attention,
            zero_init=zero_init,
            init_gain=init_gain
        )
        
        self.item_attention = EnhancedMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            multiquery=multiquery_item_attention,
            zero_init=zero_init,
            init_gain=init_gain
        )
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Optional second MLP (like TabPFN)
        if second_mlp:
            self.second_mlp_layer = EnhancedMLP(
                d_model=d_model,
                d_ff=d_model * 2,
                activation=activation,
                dropout=dropout,
                zero_init=zero_init
            )
            self.norm_second_mlp = nn.LayerNorm(d_model)
        
        # Main MLP
        self.mlp = EnhancedMLP(
            d_model=d_model,
            d_ff=d_model * 2,
            activation=activation,
            dropout=dropout,
            zero_init=zero_init
        )
        
        # Output projections (Gaussian mean and log variance)
        self.mean_head = nn.Linear(d_model, 1)
        self.log_var_head = nn.Linear(d_model, 1)
        
        # Initialize output heads
        if zero_init:
            nn.init.zeros_(self.mean_head.weight)
            nn.init.zeros_(self.mean_head.bias)
            nn.init.zeros_(self.log_var_head.weight)
            nn.init.zeros_(self.log_var_head.bias)
    
    def encode_table(self, x, y):
        """
        Encode table into cell-level representations with positional embeddings.
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
        
        # Add positional encodings
        row_pos = self.row_positional[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, d_model]
        encoded = encoded + row_pos
        
        if self.cell_positional is not None:
            cell_pos = self.cell_positional.unsqueeze(0).unsqueeze(0)  # [1, 1, num_cells, d_model]
            encoded = encoded + cell_pos
        
        return encoded
    
    def feature_attention_layer(self, x):
        """
        Apply enhanced feature attention: cells within each row attend to each other.
        """
        batch_size, seq_len, num_cells, d_model = x.shape
        
        # Reshape to process each row independently
        x_reshaped = x.view(batch_size * seq_len, num_cells, d_model)
        
        # Apply enhanced attention within each row
        attn_output = self.feature_attention(x_reshaped)
        
        # Reshape back
        attn_output = attn_output.view(batch_size, seq_len, num_cells, d_model)
        
        # Residual connection + layer norm
        return self.norm1(x + attn_output)
    
    def item_attention_layer(self, x):
        """
        Apply enhanced item attention: corresponding cells across rows attend to each other.
        """
        batch_size, seq_len, num_cells, d_model = x.shape
        
        # Process each cell position separately
        outputs = []
        
        for cell_idx in range(num_cells):
            # Extract this cell position across all rows
            cell_data = x[:, :, cell_idx, :]  # [batch, seq, d_model]
            
            # Apply enhanced attention across rows for this cell position
            attn_output = self.item_attention(cell_data)
            outputs.append(attn_output)
        
        # Stack back to [batch, seq, num_cells, d_model]
        attn_output = torch.stack(outputs, dim=2)
        
        # Residual connection + layer norm
        return self.norm2(x + attn_output)
    
    def forward(self, x_train, y_train, x_test):
        """
        Enhanced forward pass with TabPFN techniques.
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
        
        # Apply enhanced feature attention (within rows)
        encoded = self.feature_attention_layer(encoded)
        
        # Optional second MLP (like TabPFN)
        if self.second_mlp:
            batch_size, seq_len, num_cells, d_model = encoded.shape
            encoded_flat = encoded.view(-1, d_model)
            mlp_output = self.second_mlp_layer(encoded_flat)
            mlp_output = mlp_output.view(batch_size, seq_len, num_cells, d_model)
            encoded = self.norm_second_mlp(encoded + mlp_output)
        
        # Apply enhanced item attention (across rows)
        encoded = self.item_attention_layer(encoded)
        
        # Apply main MLP
        batch_size, seq_len, num_cells, d_model = encoded.shape
        encoded_flat = encoded.view(-1, d_model)
        mlp_output = self.mlp(encoded_flat)
        mlp_output = mlp_output.view(batch_size, seq_len, num_cells, d_model)
        
        # Residual + norm
        encoded = self.norm3(encoded + mlp_output)
        
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


def test_enhanced_tiny_pfn():
    """Test the Enhanced TinyPFN implementation."""
    print("Testing Enhanced TinyPFN with real TabPFN techniques...")
    
    # Create model with TabPFN features
    model = EnhancedTinyPFN(
        num_features=10, 
        d_model=256, 
        n_heads=4,
        multiquery_feature_attention=True,  # TabPFN technique
        multiquery_item_attention=False,    # Keep item attention full
        zero_init=True,                     # TabPFN technique
        init_gain=1.0,                      # TabPFN technique
        second_mlp=True,                    # TabPFN technique
        feature_pos_embedding="learned"     # TabPFN technique
    )
    
    # Create test data
    x_train, y_train, x_test, y_test = create_ridge_regression_data(batch_size=4, seq_len=20)
    
    print(f"Input shapes: x_train {x_train.shape}, y_train {y_train.shape}, x_test {x_test.shape}")
    
    # Test forward pass
    mean, log_var = model(x_train, y_train, x_test)
    print(f"Mean predictions shape: {mean.shape}")
    print(f"Log variance shape: {log_var.shape}")
    
    # Test loss computation
    loss = model.compute_loss(x_train, y_train, x_test, y_test)
    print(f"Enhanced PD-NLL loss: {loss.item():.4f}")
    
    # Test prediction
    pred = model.predict(x_train, y_train, x_test)
    print(f"Point predictions shape: {pred.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Show enhanced features
    print("\n=== ENHANCED FEATURES ===")
    print(f"✓ Multiquery feature attention: {model.feature_attention.multiquery}")
    print(f"✓ Multiquery item attention: {model.item_attention.multiquery}")
    print(f"✓ Learnable attention temperature: {model.feature_attention.softmax_scale.item():.4f}")
    print(f"✓ Zero initialization: True")
    print(f"✓ Second MLP: {model.second_mlp}")
    print(f"✓ Feature positional embeddings: learned")
    print(f"✓ Attention init gain: 1.0")
    
    print("\n✓ Enhanced TinyPFN test passed!")


if __name__ == "__main__":
    test_enhanced_tiny_pfn() 