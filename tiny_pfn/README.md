# TinyPFN: Real PFN Components in Minimal Form

This is a minimal implementation using **real PFN components** from the original codebase to demonstrate the core dual attention mechanism in its simplest form.

## Key Innovation: Using Real PFN Architecture

Instead of recreating the dual attention mechanism, we use the actual `PerFeatureLayer` from the original PFN implementation:

- **Authentic Implementation**: Uses real `PerFeatureLayer` from `pfns.model.layer`
- **Single Layer**: One layer containing the complete Feature → Item → MLP pipeline
- **Educational Focus**: Demonstrates the core innovation without complexity

## Architecture

```
Input: (batch, seq_len, num_features)
    ↓
Simple Encoders: (batch, seq_len, 1, d_model)
    ↓
PerFeatureLayer (Real PFN Component):
    ├── Feature Attention (within data points)
    ├── Item Attention (across sequence)  
    └── MLP (feed-forward processing)
    ↓
Output: Predictions for test data
```

## Design Decision: Single Real PFN Layer

TinyPFN uses **exactly 1 PerFeatureLayer** from the original PFN codebase:

- **Real Implementation**: Direct use of proven PFN architecture
- **Complete Pipeline**: Feature Attention → Item Attention → MLP in one layer
- **Maximum Simplicity**: Minimal wrapper around core PFN components
- **Educational Value**: Shows how to integrate with original PFN code

## Files

- `tiny_pfn.py` - Main implementation using real PFN components
- `README.md` - This documentation
- `requirements.txt` - Dependencies
- `__init__.py` - Package initialization

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Test
```bash
python tiny_pfn.py
```

### What the Test Shows

1. **Model Architecture**: Uses real `PerFeatureLayer` from PFN
2. **Forward Pass**: Demonstrates in-context learning
3. **Attention Visualization**: Shows feature and item attention patterns
4. **Performance**: Tests on synthetic tabular data

## Example Usage

```python
from tiny_pfn import TinyPFN

# Create model using real PFN components
model = TinyPFN(num_features=4, d_model=64, n_heads=4)

# Example data
x_train = torch.randn(8, 10, 4)  # 8 batches, 10 training examples, 4 features
y_train = torch.randn(8, 10, 1)  # Training targets
x_test = torch.randn(8, 5, 4)    # 5 test examples

# In-context learning (no parameter updates!)
predictions = model(x_train, y_train, x_test)
```

## Benefits of Using Real PFN Components

### Authenticity
- Uses the exact same `PerFeatureLayer` from the original PFN
- Proven architecture with optimized attention mechanisms
- Real-world tested implementation

### Educational Value
- Shows how to integrate with the original PFN codebase
- Demonstrates the core innovation without distractions
- Provides foundation for understanding full PFN implementation

### Simplicity
- Single layer implementation
- Minimal wrapper code
- Focus on the dual attention mechanism

## Understanding the Attention Mechanisms

The real `PerFeatureLayer` implements two types of attention:

1. **Feature Attention** (`self_attn_between_features`)
   - Features attend to each other within individual data points
   - Learns relationships like "income correlates with credit score"
   - Operates on feature dimension

2. **Item Attention** (`self_attn_between_items`)
   - Data points attend to each other across the sequence
   - Enables in-context learning from examples
   - Operates on sequence dimension

## Attention Visualization

The test includes attention heatmap visualization to understand:
- Which features the model finds most important
- How data points influence each other
- The learned attention patterns

## Next Steps

To understand the full PFN implementation:

1. Study the `PerFeatureLayer` in `../pfns/model/layer.py`
2. Explore prior-based training in `../pfns/priors/`
3. Examine the full transformer in `../pfns/model/transformer.py`
4. Try real-world applications in the notebooks

## References

- [PFN Paper: Transformers Can Do Bayesian Inference](https://arxiv.org/abs/2112.10510)
- [TabPFN: Tabular Data with Prior-data Fitted Networks](https://arxiv.org/abs/2207.01848)
- Original PFN implementation: `../pfns/` 