"""
Model representation modules for various formats.

This package provides multiple representations of transformer models:
- Hypergraph: Multi-way relationships between components
- Graph: Directed acyclic graph (DAG) representation
- Symbolic: Mathematical/algebraic representation
- Tensor Flow: Data flow representation
- AIML: Pandorabot pattern-matching representation
- OpenCog: AtomSpace typed hypergraph (URE, PLN, ECAN, MOSES)
- TOML Hypergraph: Configuration-based hypergraph with explicit tuples
"""

from .hypergraph import HypergraphRepresentation
from .graph import GraphRepresentation
from .symbolic import SymbolicRepresentation
from .comparator import RepresentationComparator
from .aiml import AIMLRepresentation
from .opencog import OpenCogAtomSpaceRepresentation
from .toml_hypergraph import TOMLHypergraphRepresentation

__all__ = [
    "HypergraphRepresentation",
    "GraphRepresentation",
    "SymbolicRepresentation",
    "RepresentationComparator",
    "AIMLRepresentation",
    "OpenCogAtomSpaceRepresentation",
    "TOMLHypergraphRepresentation",
]
