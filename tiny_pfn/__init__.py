"""
Tiny PFN: A simplified implementation of the dual attention mechanism from Prior-data Fitted Networks.

This package contains a minimal implementation focusing on the core innovation:
- Feature attention: Features attend to each other within each data point
- Item attention: Data points attend to each other across the sequence
"""

from .dual_attention import DualAttentionLayer
from .tiny_transformer import TinyPFN

__all__ = ["DualAttentionLayer", "TinyPFN"] 