"""
OpenCog AtomSpace representation for transformer models.

OpenCog is a cognitive architecture framework that uses a typed hypergraph
(AtomSpace) for knowledge representation. This module represents the tiny
transformer in OpenCog Scheme format compatible with URE (Unified Rule Engine),
PLN (Probabilistic Logic Networks), ECAN (Economic Attention Networks), and
MOSES (Meta-Optimizing Semantic Evolutionary Search).
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np


class OpenCogAtomSpaceRepresentation:
    """
    OpenCog AtomSpace representation of a transformer model.
    
    AtomSpace uses a typed hypergraph where:
    - Atoms are nodes (ConceptNode, PredicateNode, etc.) or links
    - Links connect atoms to form knowledge structures
    - Truth values represent uncertainty (PLN)
    - Attention values guide inference (ECAN)
    - Pattern matching and rule application (URE)
    
    This format is useful for:
    - Symbolic AI reasoning about the model
    - Combining neural and symbolic approaches
    - Probabilistic inference over model structure
    - Evolutionary optimization with MOSES
    - Integration with cognitive architectures
    """
    
    def __init__(self):
        """Initialize empty AtomSpace representation."""
        self.atoms: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.knowledge_base: List[str] = []
        
    def add_concept_node(self, name: str, tv: Optional[Tuple[float, float]] = None) -> str:
        """
        Add a ConceptNode to the AtomSpace.
        
        Args:
            name: Concept name
            tv: Optional truth value (strength, confidence)
        
        Returns:
            Scheme expression for the node
        """
        if tv:
            atom = f'(ConceptNode "{name}" (stv {tv[0]} {tv[1]}))'
        else:
            atom = f'(ConceptNode "{name}")'
        self.atoms.append(atom)
        return atom
    
    def add_predicate_node(self, name: str, tv: Optional[Tuple[float, float]] = None) -> str:
        """Add a PredicateNode to the AtomSpace."""
        if tv:
            atom = f'(PredicateNode "{name}" (stv {tv[0]} {tv[1]}))'
        else:
            atom = f'(PredicateNode "{name}")'
        self.atoms.append(atom)
        return atom
    
    def add_number_node(self, value: float) -> str:
        """Add a NumberNode to the AtomSpace."""
        atom = f'(NumberNode "{value}")'
        self.atoms.append(atom)
        return atom
    
    def add_inheritance_link(self, child: str, parent: str, 
                            tv: Optional[Tuple[float, float]] = None) -> str:
        """
        Add an InheritanceLink (is-a relationship).
        
        Args:
            child: Child concept
            parent: Parent concept
            tv: Optional truth value
        
        Returns:
            Scheme expression for the link
        """
        if tv:
            link = f'(InheritanceLink (stv {tv[0]} {tv[1]})\n  (ConceptNode "{child}")\n  (ConceptNode "{parent}"))'
        else:
            link = f'(InheritanceLink\n  (ConceptNode "{child}")\n  (ConceptNode "{parent}"))'
        self.atoms.append(link)
        return link
    
    def add_evaluation_link(self, predicate: str, *args, 
                           tv: Optional[Tuple[float, float]] = None) -> str:
        """
        Add an EvaluationLink (predicate applied to arguments).
        
        Args:
            predicate: Predicate name
            args: Argument concepts
            tv: Optional truth value
        
        Returns:
            Scheme expression for the link
        """
        list_link_content = '\n    '.join([f'(ConceptNode "{arg}")' for arg in args])
        
        if tv:
            link = f'''(EvaluationLink (stv {tv[0]} {tv[1]})
  (PredicateNode "{predicate}")
  (ListLink
    {list_link_content}))'''
        else:
            link = f'''(EvaluationLink
  (PredicateNode "{predicate}")
  (ListLink
    {list_link_content}))'''
        
        self.atoms.append(link)
        return link
    
    def add_member_link(self, element: str, set_name: str) -> str:
        """Add a MemberLink (set membership)."""
        link = f'(MemberLink\n  (ConceptNode "{element}")\n  (ConceptNode "{set_name}"))'
        self.atoms.append(link)
        return link
    
    def add_execution_link(self, schema: str, *args) -> str:
        """
        Add an ExecutionLink (function execution).
        
        Args:
            schema: Schema/function name
            args: Arguments
        
        Returns:
            Scheme expression for the link
        """
        list_link_content = '\n    '.join([f'(ConceptNode "{arg}")' for arg in args])
        
        link = f'''(ExecutionLink
  (SchemaNode "{schema}")
  (ListLink
    {list_link_content}))'''
        
        self.atoms.append(link)
        return link
    
    @classmethod
    def from_tiny_transformer(cls, model_path: Optional[Path] = None) -> "OpenCogAtomSpaceRepresentation":
        """
        Create OpenCog AtomSpace representation from tiny transformer model.
        
        Args:
            model_path: Optional path to model file
        
        Returns:
            OpenCogAtomSpaceRepresentation of the model
        """
        atomspace = cls()
        
        # Metadata
        atomspace.metadata = {
            "model_name": "TinyTransformer",
            "architecture": "transformer",
            "vocab_size": 10,
            "embedding_dim": 5,
            "context_length": 5,
            "num_blocks": 1,
            "num_heads": 1,
            "feedforward_dim": 5,
            "total_parameters": 200,
        }
        
        # === Model Architecture Knowledge ===
        
        # Define the model as a concept
        atomspace.add_concept_node("TinyTransformer", tv=(1.0, 1.0))
        atomspace.add_inheritance_link("TinyTransformer", "TransformerModel", tv=(1.0, 1.0))
        atomspace.add_inheritance_link("TransformerModel", "NeuralNetwork", tv=(1.0, 1.0))
        
        # === Model Properties ===
        
        # Vocabulary size
        atomspace.add_predicate_node("hasVocabularySize")
        atomspace.add_evaluation_link("hasVocabularySize", "TinyTransformer", "VocabSize10", tv=(1.0, 1.0))
        atomspace.add_number_node(10)
        
        # Embedding dimension
        atomspace.add_predicate_node("hasEmbeddingDimension")
        atomspace.add_evaluation_link("hasEmbeddingDimension", "TinyTransformer", "EmbedDim5", tv=(1.0, 1.0))
        atomspace.add_number_node(5)
        
        # Context length
        atomspace.add_predicate_node("hasContextLength")
        atomspace.add_evaluation_link("hasContextLength", "TinyTransformer", "ContextLen5", tv=(1.0, 1.0))
        
        # Number of blocks
        atomspace.add_predicate_node("hasNumberOfBlocks")
        atomspace.add_evaluation_link("hasNumberOfBlocks", "TinyTransformer", "Blocks1", tv=(1.0, 1.0))
        
        # Number of attention heads
        atomspace.add_predicate_node("hasAttentionHeads")
        atomspace.add_evaluation_link("hasAttentionHeads", "TinyTransformer", "Heads1", tv=(1.0, 1.0))
        
        # === Layer Concepts ===
        
        # Embedding Layer
        atomspace.add_concept_node("EmbeddingLayer")
        atomspace.add_inheritance_link("EmbeddingLayer", "Layer", tv=(1.0, 1.0))
        atomspace.add_member_link("EmbeddingLayer", "TinyTransformerLayers")
        
        atomspace.add_predicate_node("hasInputDimension")
        atomspace.add_evaluation_link("hasInputDimension", "EmbeddingLayer", "Dim10", tv=(1.0, 1.0))
        
        atomspace.add_predicate_node("hasOutputDimension")
        atomspace.add_evaluation_link("hasOutputDimension", "EmbeddingLayer", "Dim5", tv=(1.0, 1.0))
        
        # Self-Attention Layer
        atomspace.add_concept_node("SelfAttentionLayer")
        atomspace.add_inheritance_link("SelfAttentionLayer", "Layer", tv=(1.0, 1.0))
        atomspace.add_member_link("SelfAttentionLayer", "TinyTransformerLayers")
        
        atomspace.add_predicate_node("hasQueryProjection")
        atomspace.add_evaluation_link("hasQueryProjection", "SelfAttentionLayer", "QueryWeights", tv=(1.0, 1.0))
        
        atomspace.add_predicate_node("hasKeyProjection")
        atomspace.add_evaluation_link("hasKeyProjection", "SelfAttentionLayer", "KeyWeights", tv=(1.0, 1.0))
        
        atomspace.add_predicate_node("hasValueProjection")
        atomspace.add_evaluation_link("hasValueProjection", "SelfAttentionLayer", "ValueWeights", tv=(1.0, 1.0))
        
        # Feed-Forward Layer
        atomspace.add_concept_node("FeedForwardLayer")
        atomspace.add_inheritance_link("FeedForwardLayer", "Layer", tv=(1.0, 1.0))
        atomspace.add_member_link("FeedForwardLayer", "TinyTransformerLayers")
        
        atomspace.add_evaluation_link("hasInputDimension", "FeedForwardLayer", "Dim5", tv=(1.0, 1.0))
        atomspace.add_evaluation_link("hasOutputDimension", "FeedForwardLayer", "Dim5", tv=(1.0, 1.0))
        
        # Output Layer
        atomspace.add_concept_node("OutputLayer")
        atomspace.add_inheritance_link("OutputLayer", "Layer", tv=(1.0, 1.0))
        atomspace.add_member_link("OutputLayer", "TinyTransformerLayers")
        
        atomspace.add_evaluation_link("hasInputDimension", "OutputLayer", "Dim5", tv=(1.0, 1.0))
        atomspace.add_evaluation_link("hasOutputDimension", "OutputLayer", "Dim10", tv=(1.0, 1.0))
        
        # === Computational Flow (Execution Links for URE) ===
        
        atomspace.add_concept_node("InputTokens")
        atomspace.add_concept_node("Embeddings")
        atomspace.add_concept_node("AttentionOutput")
        atomspace.add_concept_node("FFNOutput")
        atomspace.add_concept_node("FinalOutput")
        
        # Define computational schema
        atomspace.add_execution_link("Embed", "InputTokens", "Embeddings")
        atomspace.add_execution_link("SelfAttend", "Embeddings", "AttentionOutput")
        atomspace.add_execution_link("FeedForward", "AttentionOutput", "FFNOutput")
        atomspace.add_execution_link("Project", "FFNOutput", "FinalOutput")
        
        # === URE Rules for Inference ===
        
        # Add some PLN-style rules
        atomspace.knowledge_base.append("""
; Rule: If a model has embedding layer, it can process tokens
(ImplicationLink (stv 0.9 0.9)
  (MemberLink
    (ConceptNode "EmbeddingLayer")
    (ConceptNode "TinyTransformerLayers"))
  (EvaluationLink
    (PredicateNode "canProcessTokens")
    (ConceptNode "TinyTransformer")))
""")
        
        atomspace.knowledge_base.append("""
; Rule: If a model has attention, it can model dependencies
(ImplicationLink (stv 0.95 0.95)
  (MemberLink
    (ConceptNode "SelfAttentionLayer")
    (ConceptNode "TinyTransformerLayers"))
  (EvaluationLink
    (PredicateNode "canModelDependencies")
    (ConceptNode "TinyTransformer")))
""")
        
        atomspace.knowledge_base.append("""
; Rule: Sequential composition of layers
(BindLink
  (VariableList
    (VariableNode "$L1")
    (VariableNode "$L2")
    (VariableNode "$X")
    (VariableNode "$Y")
    (VariableNode "$Z"))
  (AndLink
    (ExecutionLink
      (VariableNode "$L1")
      (ListLink (VariableNode "$X") (VariableNode "$Y")))
    (ExecutionLink
      (VariableNode "$L2")
      (ListLink (VariableNode "$Y") (VariableNode "$Z"))))
  (ExecutionLink
    (SchemaNode "Compose")
    (ListLink
      (VariableNode "$L1")
      (VariableNode "$L2")
      (VariableNode "$X")
      (VariableNode "$Z"))))
""")
        
        # === ECAN Attention Values ===
        # (In practice, these would be dynamically computed)
        atomspace.knowledge_base.append("""
; ECAN: Critical nodes get higher attention values
; Attention layer is most important
(SetLink
  (ConceptNode "SelfAttentionLayer" (av 100 10 1))
  (ConceptNode "EmbeddingLayer" (av 80 8 1))
  (ConceptNode "OutputLayer" (av 70 7 1)))
""")
        
        # === MOSES Evolutionary Search Space ===
        atomspace.knowledge_base.append("""
; MOSES: Define search space for hyperparameters
(DefineLink
  (DefinedSchemaNode "OptimizeTransformer")
  (LambdaLink
    (VariableList
      (TypedVariableLink (VariableNode "$embedding_dim") (TypeNode "NumberNode"))
      (TypedVariableLink (VariableNode "$num_heads") (TypeNode "NumberNode"))
      (TypedVariableLink (VariableNode "$num_blocks") (TypeNode "NumberNode")))
    (ExecutionOutputLink
      (GroundedSchemaNode "py:evaluate_transformer")
      (ListLink
        (VariableNode "$embedding_dim")
        (VariableNode "$num_heads")
        (VariableNode "$num_blocks")))))
""")
        
        return atomspace
    
    def to_scheme(self) -> str:
        """
        Convert to OpenCog Scheme format.
        
        Returns:
            Scheme code string
        """
        lines = [
            "; OpenCog AtomSpace representation of TinyTransformer",
            "; Compatible with URE, PLN, ECAN, and MOSES",
            "; Generated by gguf-workbench",
            "",
            "(use-modules (opencog))",
            "(use-modules (opencog exec))",
            "(use-modules (opencog ure))",
            "(use-modules (opencog pln))",
            "",
            "; === Core Atoms ===",
            ""
        ]
        
        # Add all atoms
        for atom in self.atoms:
            lines.append(atom)
            lines.append("")
        
        # Add knowledge base rules
        if self.knowledge_base:
            lines.append("; === Knowledge Base Rules ===")
            lines.append("")
            for rule in self.knowledge_base:
                lines.append(rule)
        
        return "\n".join(lines)
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert to JSON representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "metadata": self.metadata,
            "atoms": self.atoms,
            "atom_count": len(self.atoms),
            "knowledge_base_rules": len(self.knowledge_base),
            "format": "OpenCog AtomSpace (Scheme)"
        }
    
    def save_scheme(self, path: Path) -> None:
        """Save AtomSpace to Scheme file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_scheme())
    
    def save_json(self, path: Path) -> None:
        """Save AtomSpace to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)
