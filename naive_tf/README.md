# Naive Transformer for Comparison with TinyPFN

This directory contains a naive 1-layer transformer implementation that serves as a baseline comparison against TinyPFN's dual attention mechanism.

## Key Differences from TinyPFN

| Feature | Naive Transformer | TinyPFN |
|---------|------------------|---------|
| **Feature Attention** | ❌ No feature attention | ✅ Features attend to each other |
| **Item Attention** | ✅ Data points attend to each other | ✅ Data points attend to each other |
| **Architecture** | Standard transformer layer | Dual attention mechanism |
| **Tabular Inductive Bias** | ❌ No specific tabular bias | ✅ Designed for tabular data |

## Architecture

The naive transformer uses a standard single-layer transformer architecture:

1. **Input Encoding**: Features and targets are encoded separately and combined
2. **Item Attention**: Only data points attend to each other (standard self-attention)
3. **Feed-Forward Network**: Standard MLP with residual connections
4. **Output Projection**: Linear layer to produce predictions

## Usage

```python
from naive_tf import NaiveTransformer

# Create model
model = NaiveTransformer(num_features=4)

# Use for in-context learning
predictions = model(x_train, y_train, x_test)
```

## Purpose

This implementation helps us understand:
- How much benefit comes from TinyPFN's dual attention mechanism
- Whether feature attention provides meaningful improvements
- The importance of tabular-specific inductive biases

## Expected Results

We expect the naive transformer to perform worse than TinyPFN because:
1. **No Feature Attention**: Cannot model interactions between features within each data point
2. **No Tabular Bias**: Treats tabular data like sequential data
3. **Suboptimal Information Flow**: Features don't communicate before items interact

## Files

- `naive_transformer.py`: Main implementation
- `compare_models.py`: Comparison script with TinyPFN
- `__init__.py`: Package initialization
- `README.md`: This documentation
- `requirements.txt`: Dependencies 