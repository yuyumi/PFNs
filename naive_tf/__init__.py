"""
Naive Transformer Implementation for Comparison with TinyPFN

This package contains a naive 1-layer transformer that uses only standard
item attention (no feature attention) to serve as a baseline comparison
against TinyPFN's dual attention mechanism.
"""

from .naive_transformer import NaiveTransformer

__all__ = ['NaiveTransformer'] 