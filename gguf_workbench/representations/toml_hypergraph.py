"""
TOML Hypergraph representation for transformer models.

This module represents the model as a TOML file with explicit hypergraph
tuples showing the multi-way relationships between tensors and operations.
Unlike simple TOML configs, this includes full hyperedge specifications.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


class TOMLHypergraphRepresentation:
    """
    TOML Hypergraph representation of a transformer model.
    
    Represents the model as TOML with:
    - Metadata section for model properties
    - Vertices section for tensors and parameters
    - Hyperedges section with explicit source/target tuples
    - Weight arrays for full model specification
    
    Advantages:
    - Human-readable configuration format
    - Easy to edit and version control
    - Explicit hypergraph structure
    - Widely supported TOML parsers
    - Can include actual weight values
    
    This format is useful for:
    - Configuration-based model definition
    - Hypergraph structure specification
    - Educational demonstrations
    - Integration with TOML-based tools
    """
    
    def __init__(self):
        """Initialize empty TOML hypergraph representation."""
        self.metadata: Dict[str, Any] = {}
        self.vertices: Dict[str, Dict[str, Any]] = {}
        self.hyperedges: Dict[str, Dict[str, Any]] = {}
        self.weights: Dict[str, List[List[float]]] = {}
    
    def add_vertex(self, vertex_id: str, vertex_type: str, 
                   shape: Optional[Tuple[Optional[int], ...]] = None,
                   dtype: str = "float32",
                   properties: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a vertex to the hypergraph.
        
        Args:
            vertex_id: Unique vertex identifier
            vertex_type: Type of vertex (tensor, parameter, operation)
            shape: Optional shape tuple (None for batch dimensions)
            dtype: Data type
            properties: Additional properties
        """
        vertex = {
            "type": vertex_type,
            "dtype": dtype
        }
        
        if shape is not None:
            # Convert None to "batch" for readability in TOML
            shape_list = ["batch" if s is None else s for s in shape]
            vertex["shape"] = shape_list
        
        if properties:
            vertex["properties"] = properties
        
        self.vertices[vertex_id] = vertex
    
    def add_hyperedge(self, edge_id: str, operation: str,
                     sources: List[str], targets: List[str],
                     properties: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a hyperedge to the hypergraph.
        
        Args:
            edge_id: Unique edge identifier
            operation: Operation type (matmul, add, attention, etc.)
            sources: List of source vertex IDs
            targets: List of target vertex IDs
            properties: Additional properties
        """
        hyperedge = {
            "operation": operation,
            "sources": sources,
            "targets": targets
        }
        
        if properties:
            hyperedge["properties"] = properties
        
        self.hyperedges[edge_id] = hyperedge
    
    def add_weights(self, param_id: str, weights: List[List[float]]) -> None:
        """
        Add weight values for a parameter.
        
        Args:
            param_id: Parameter vertex ID
            weights: Weight matrix (2D list)
        """
        self.weights[param_id] = weights
    
    @classmethod
    def from_tiny_transformer(cls, model_path: Optional[Path] = None,
                             include_weights: bool = True) -> "TOMLHypergraphRepresentation":
        """
        Create TOML hypergraph representation from tiny transformer.
        
        Args:
            model_path: Optional path to load actual weights
            include_weights: Whether to include weight values
        
        Returns:
            TOMLHypergraphRepresentation of the model
        """
        toml_hg = cls()
        
        # === Metadata ===
        toml_hg.metadata = {
            "model_name": "TinyTransformer",
            "architecture": "transformer",
            "representation": "hypergraph",
            "vocab_size": 10,
            "embedding_dim": 5,
            "context_length": 5,
            "num_blocks": 1,
            "num_heads": 1,
            "feedforward_dim": 5,
            "total_parameters": 200,
            "format_version": "1.0"
        }
        
        # === Vertices ===
        
        # Input tokens
        toml_hg.add_vertex(
            "input_tokens",
            "tensor",
            shape=(None, 5),  # (batch, seq_len)
            dtype="int64",
            properties={"description": "Input token IDs", "range": [0, 9]}
        )
        
        # Embedding weights
        toml_hg.add_vertex(
            "embedding_weights",
            "parameter",
            shape=(10, 5),
            dtype="float32",
            properties={"description": "Token embedding matrix", "parameter_count": 50}
        )
        
        # Embeddings
        toml_hg.add_vertex(
            "embeddings",
            "tensor",
            shape=(None, 5, 5),  # (batch, seq_len, embed_dim)
            dtype="float32",
            properties={"description": "Token embeddings"}
        )
        
        # Position embeddings (implicit in this model)
        toml_hg.add_vertex(
            "position_embeddings",
            "tensor",
            shape=(5, 5),
            dtype="float32",
            properties={"description": "Position embeddings"}
        )
        
        # Attention weights: Q, K, V projections
        toml_hg.add_vertex(
            "query_weights",
            "parameter",
            shape=(5, 5),
            dtype="float32",
            properties={"description": "Query projection weights", "parameter_count": 25}
        )
        
        toml_hg.add_vertex(
            "key_weights",
            "parameter",
            shape=(5, 5),
            dtype="float32",
            properties={"description": "Key projection weights", "parameter_count": 25}
        )
        
        toml_hg.add_vertex(
            "value_weights",
            "parameter",
            shape=(5, 5),
            dtype="float32",
            properties={"description": "Value projection weights", "parameter_count": 25}
        )
        
        # Attention intermediate tensors
        toml_hg.add_vertex(
            "queries",
            "tensor",
            shape=(None, 5, 5),
            dtype="float32",
            properties={"description": "Query vectors"}
        )
        
        toml_hg.add_vertex(
            "keys",
            "tensor",
            shape=(None, 5, 5),
            dtype="float32",
            properties={"description": "Key vectors"}
        )
        
        toml_hg.add_vertex(
            "values",
            "tensor",
            shape=(None, 5, 5),
            dtype="float32",
            properties={"description": "Value vectors"}
        )
        
        toml_hg.add_vertex(
            "attention_scores",
            "tensor",
            shape=(None, 5, 5),
            dtype="float32",
            properties={"description": "Attention score matrix"}
        )
        
        toml_hg.add_vertex(
            "attention_output",
            "tensor",
            shape=(None, 5, 5),
            dtype="float32",
            properties={"description": "Attention output"}
        )
        
        # Feed-forward weights
        toml_hg.add_vertex(
            "ffn_w1",
            "parameter",
            shape=(5, 5),
            dtype="float32",
            properties={"description": "FFN first layer weights", "parameter_count": 25}
        )
        
        toml_hg.add_vertex(
            "ffn_w2",
            "parameter",
            shape=(5, 5),
            dtype="float32",
            properties={"description": "FFN second layer weights", "parameter_count": 25}
        )
        
        # FFN intermediate
        toml_hg.add_vertex(
            "ffn_hidden",
            "tensor",
            shape=(None, 5, 5),
            dtype="float32",
            properties={"description": "FFN hidden activations"}
        )
        
        toml_hg.add_vertex(
            "ffn_output",
            "tensor",
            shape=(None, 5, 5),
            dtype="float32",
            properties={"description": "FFN output"}
        )
        
        # Output layer weights
        toml_hg.add_vertex(
            "output_weights",
            "parameter",
            shape=(5, 10),
            dtype="float32",
            properties={"description": "Output projection weights", "parameter_count": 50}
        )
        
        # Final output
        toml_hg.add_vertex(
            "logits",
            "tensor",
            shape=(None, 5, 10),
            dtype="float32",
            properties={"description": "Output logits"}
        )
        
        # === Hyperedges ===
        
        # Embedding lookup
        toml_hg.add_hyperedge(
            "embed",
            "embedding_lookup",
            sources=["input_tokens", "embedding_weights"],
            targets=["embeddings"],
            properties={"description": "Token embedding lookup"}
        )
        
        # Query projection
        toml_hg.add_hyperedge(
            "project_queries",
            "matmul",
            sources=["embeddings", "query_weights"],
            targets=["queries"],
            properties={"description": "Project to query space"}
        )
        
        # Key projection
        toml_hg.add_hyperedge(
            "project_keys",
            "matmul",
            sources=["embeddings", "key_weights"],
            targets=["keys"],
            properties={"description": "Project to key space"}
        )
        
        # Value projection
        toml_hg.add_hyperedge(
            "project_values",
            "matmul",
            sources=["embeddings", "value_weights"],
            targets=["values"],
            properties={"description": "Project to value space"}
        )
        
        # Attention scoring (Q @ K^T)
        toml_hg.add_hyperedge(
            "compute_attention_scores",
            "scaled_dot_product",
            sources=["queries", "keys"],
            targets=["attention_scores"],
            properties={"description": "Compute attention scores", "scaling_factor": 0.447}  # 1/sqrt(5)
        )
        
        # Attention aggregation (scores @ V)
        toml_hg.add_hyperedge(
            "apply_attention",
            "attention_aggregate",
            sources=["attention_scores", "values"],
            targets=["attention_output"],
            properties={"description": "Apply attention to values"}
        )
        
        # Feed-forward layer 1
        toml_hg.add_hyperedge(
            "ffn_layer1",
            "matmul_activation",
            sources=["attention_output", "ffn_w1"],
            targets=["ffn_hidden"],
            properties={"description": "FFN first layer", "activation": "relu"}
        )
        
        # Feed-forward layer 2
        toml_hg.add_hyperedge(
            "ffn_layer2",
            "matmul",
            sources=["ffn_hidden", "ffn_w2"],
            targets=["ffn_output"],
            properties={"description": "FFN second layer"}
        )
        
        # Output projection
        toml_hg.add_hyperedge(
            "output_projection",
            "matmul",
            sources=["ffn_output", "output_weights"],
            targets=["logits"],
            properties={"description": "Project to vocabulary"}
        )
        
        # === Weights (simplified example values) ===
        if include_weights:
            # Simplified weight initialization (would load from actual model)
            import numpy as np
            np.random.seed(42)
            
            # Embedding weights (10 x 5)
            toml_hg.add_weights(
                "embedding_weights",
                np.random.randn(10, 5).round(4).tolist()
            )
            
            # Attention weights (5 x 5 each)
            toml_hg.add_weights("query_weights", np.random.randn(5, 5).round(4).tolist())
            toml_hg.add_weights("key_weights", np.random.randn(5, 5).round(4).tolist())
            toml_hg.add_weights("value_weights", np.random.randn(5, 5).round(4).tolist())
            
            # FFN weights (5 x 5 each)
            toml_hg.add_weights("ffn_w1", np.random.randn(5, 5).round(4).tolist())
            toml_hg.add_weights("ffn_w2", np.random.randn(5, 5).round(4).tolist())
            
            # Output weights (5 x 10)
            toml_hg.add_weights("output_weights", np.random.randn(5, 10).round(4).tolist())
        
        return toml_hg
    
    def to_toml(self) -> str:
        """
        Convert to TOML format string.
        
        Returns:
            TOML string representation
        """
        lines = [
            "# TOML Hypergraph Representation of TinyTransformer",
            "# Generated by gguf-workbench",
            "# Format: Explicit hypergraph with vertices and hyperedges",
            "",
            "[metadata]"
        ]
        
        # Add metadata
        for key, value in self.metadata.items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
        
        lines.append("")
        
        # Add vertices
        lines.append("# === Vertices ===")
        lines.append("# Format: [vertices.<id>]")
        lines.append("")
        
        for vertex_id, vertex_data in self.vertices.items():
            lines.append(f"[vertices.{vertex_id}]")
            for key, value in vertex_data.items():
                if key == "properties":
                    # Nested properties
                    lines.append(f"[vertices.{vertex_id}.properties]")
                    for prop_key, prop_value in value.items():
                        if isinstance(prop_value, str):
                            lines.append(f'{prop_key} = "{prop_value}"')
                        else:
                            lines.append(f"{prop_key} = {prop_value}")
                elif isinstance(value, str):
                    lines.append(f'{key} = "{value}"')
                elif isinstance(value, list):
                    lines.append(f"{key} = {value}")
                else:
                    lines.append(f"{key} = {value}")
            lines.append("")
        
        # Add hyperedges
        lines.append("# === Hyperedges ===")
        lines.append("# Format: [hyperedges.<id>]")
        lines.append("# Each hyperedge has sources (inputs) and targets (outputs)")
        lines.append("")
        
        for edge_id, edge_data in self.hyperedges.items():
            lines.append(f"[hyperedges.{edge_id}]")
            lines.append(f'operation = "{edge_data["operation"]}"')
            lines.append(f'sources = {edge_data["sources"]}')
            lines.append(f'targets = {edge_data["targets"]}')
            
            if "properties" in edge_data:
                lines.append(f"[hyperedges.{edge_id}.properties]")
                for prop_key, prop_value in edge_data["properties"].items():
                    if isinstance(prop_value, str):
                        lines.append(f'{prop_key} = "{prop_value}"')
                    else:
                        lines.append(f"{prop_key} = {prop_value}")
            
            lines.append("")
        
        # Add weights if present
        if self.weights:
            lines.append("# === Weights ===")
            lines.append("# Format: [weights.<param_id>]")
            lines.append("")
            
            for param_id, weight_matrix in self.weights.items():
                lines.append(f"[weights.{param_id}]")
                lines.append("# Shape: " + str(self.vertices[param_id].get("shape", "unknown")))
                lines.append(f"values = [")
                for row in weight_matrix:
                    lines.append(f"  {row},")
                lines.append("]")
                lines.append("")
        
        return "\n".join(lines)
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert to JSON representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "metadata": self.metadata,
            "vertices": self.vertices,
            "hyperedges": self.hyperedges,
            "weights": self.weights,
            "statistics": {
                "vertex_count": len(self.vertices),
                "hyperedge_count": len(self.hyperedges),
                "parameter_count": sum(
                    v.get("properties", {}).get("parameter_count", 0)
                    for v in self.vertices.values()
                )
            }
        }
    
    def save_toml(self, path: Path) -> None:
        """Save to TOML file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_toml())
    
    def save_json(self, path: Path) -> None:
        """Save to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)
