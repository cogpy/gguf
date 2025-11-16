"""
Pure Python inference implementations for TinyTransformer.

This module provides multiple implementation approaches to demonstrate
how the tiny transformer model processes inputs, showing:
- How indexed weights are applied to each input element
- How tokenized features relate to the vocabulary
- Step-by-step computation traces
- Output determination process

Different implementations show various programming paradigms:
- List-based (simple arrays)
- Dict-based (structured data)
- Class-based (OOP with dataclasses)
- Functional (pure functions)
"""

from .list_based import TinyTransformerListBased
from .dict_based import TinyTransformerDictBased
from .class_based import TinyTransformerClassBased
from .functional import tiny_transformer_functional
from .trace import InferenceTracer

__all__ = [
    'TinyTransformerListBased',
    'TinyTransformerDictBased',
    'TinyTransformerClassBased',
    'tiny_transformer_functional',
    'InferenceTracer',
]
