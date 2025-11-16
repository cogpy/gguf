"""
Hypergraph representation for transformer models.

A hypergraph generalizes graphs by allowing edges (hyperedges) to connect
any number of vertices. This is ideal for representing transformer operations
where multiple tensors participate in a single operation (e.g., attention
combining query, key, and value).
"""

import json
from typing import Any, Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Vertex:
    """A vertex in the hypergraph representing a tensor or operation."""

    id: str
    type: str  # 'tensor', 'operation', 'parameter'
    properties: Dict[str, Any] = field(default_factory=dict)
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class Hyperedge:
    """A hyperedge connecting multiple vertices in the hypergraph."""

    id: str
    sources: List[str]  # Source vertex IDs
    targets: List[str]  # Target vertex IDs
    operation: str  # Operation type (matmul, add, attention, etc.)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


class HypergraphRepresentation:
    """
    Hypergraph representation of a transformer model.

    In a hypergraph:
    - Vertices represent tensors (embeddings, weights, activations)
    - Hyperedges represent operations that can involve multiple inputs/outputs
    - Example: Attention hyperedge connects Q, K, V tensors to output tensor

    Advantages:
    - Naturally represents multi-input/multi-output operations
    - Captures complex dependencies in transformer architecture
    - Enables advanced analysis (hypergraph cuts, clustering)
    - More expressive than simple graphs for neural networks

    Disadvantages:
    - More complex than simple directed graphs
    - Visualization can be challenging
    - Some algorithms don't have hypergraph equivalents
    """

    def __init__(self):
        """Initialize empty hypergraph."""
        self.vertices: Dict[str, Vertex] = {}
        self.hyperedges: Dict[str, Hyperedge] = {}
        self.metadata: Dict[str, Any] = {}

    def add_vertex(self, vertex: Vertex) -> None:
        """Add a vertex to the hypergraph."""
        self.vertices[vertex.id] = vertex

    def add_hyperedge(self, hyperedge: Hyperedge) -> None:
        """Add a hyperedge to the hypergraph."""
        # Validate that all referenced vertices exist
        all_vertices = set(hyperedge.sources + hyperedge.targets)
        for v_id in all_vertices:
            if v_id not in self.vertices:
                raise ValueError(f"Vertex {v_id} not found in hypergraph")
        self.hyperedges[hyperedge.id] = hyperedge

    def get_incident_hyperedges(self, vertex_id: str) -> List[Hyperedge]:
        """Get all hyperedges incident to a vertex."""
        result = []
        for edge in self.hyperedges.values():
            if vertex_id in edge.sources or vertex_id in edge.targets:
                result.append(edge)
        return result

    def get_neighbors(self, vertex_id: str) -> Set[str]:
        """Get all vertices connected to the given vertex."""
        neighbors = set()
        for edge in self.get_incident_hyperedges(vertex_id):
            neighbors.update(edge.sources)
            neighbors.update(edge.targets)
        neighbors.discard(vertex_id)
        return neighbors

    def to_dict(self) -> Dict[str, Any]:
        """Convert hypergraph to dictionary representation."""
        return {
            "metadata": self.metadata,
            "vertices": {v_id: v.to_dict() for v_id, v in self.vertices.items()},
            "hyperedges": {e_id: e.to_dict() for e_id, e in self.hyperedges.items()},
            "statistics": self.get_statistics(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Compute statistics about the hypergraph."""
        vertex_types = {}
        for v in self.vertices.values():
            vertex_types[v.type] = vertex_types.get(v.type, 0) + 1

        operation_types = {}
        for e in self.hyperedges.values():
            operation_types[e.operation] = operation_types.get(e.operation, 0) + 1

        hyperedge_sizes = [len(e.sources) + len(e.targets) for e in self.hyperedges.values()]
        avg_hyperedge_size = sum(hyperedge_sizes) / len(hyperedge_sizes) if hyperedge_sizes else 0

        return {
            "vertex_count": len(self.vertices),
            "hyperedge_count": len(self.hyperedges),
            "vertex_types": vertex_types,
            "operation_types": operation_types,
            "average_hyperedge_size": avg_hyperedge_size,
            "max_hyperedge_size": max(hyperedge_sizes) if hyperedge_sizes else 0,
        }

    def to_json(self, filepath: Optional[Path] = None, indent: int = 2) -> str:
        """
        Export hypergraph to JSON format.

        Args:
            filepath: Optional path to save JSON file
            indent: Indentation level for JSON output

        Returns:
            JSON string representation
        """
        json_str = json.dumps(self.to_dict(), indent=indent)
        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_tiny_transformer(cls, model_path: Optional[Path] = None) -> "HypergraphRepresentation":
        """
        Create hypergraph representation from tiny transformer model.

        Args:
            model_path: Path to model file (GGUF, PyTorch, etc.)

        Returns:
            HypergraphRepresentation of the model
        """
        hg = cls()

        # Model metadata
        hg.metadata = {
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

        # Input layer: token IDs
        hg.add_vertex(
            Vertex(
                id="input_tokens",
                type="tensor",
                shape=(None, 5),  # (batch_size, sequence_length)
                dtype="int64",
                properties={"description": "Input token IDs", "range": [0, 9]},
            )
        )

        # Embedding layer
        hg.add_vertex(
            Vertex(
                id="embedding_weights",
                type="parameter",
                shape=(10, 5),  # (vocab_size, embedding_dim)
                dtype="float32",
                properties={"description": "Token embedding matrix", "parameter_count": 50},
            )
        )

        hg.add_vertex(
            Vertex(
                id="embeddings",
                type="tensor",
                shape=(None, 5, 5),  # (batch, seq_len, embedding_dim)
                dtype="float32",
                properties={"description": "Token embeddings"},
            )
        )

        hg.add_hyperedge(
            Hyperedge(
                id="embed_tokens",
                sources=["input_tokens", "embedding_weights"],
                targets=["embeddings"],
                operation="embedding_lookup",
                properties={"description": "Convert token IDs to embeddings"},
            )
        )

        # Position embeddings (implicit in RoPE)
        hg.add_vertex(
            Vertex(
                id="position_ids",
                type="tensor",
                shape=(None, 5),
                dtype="int64",
                properties={"description": "Position indices", "range": [0, 4]},
            )
        )

        # Attention block - Query, Key, Value projections
        for proj in ["query", "key", "value"]:
            hg.add_vertex(
                Vertex(
                    id=f"attn_{proj}_weights",
                    type="parameter",
                    shape=(5, 5),
                    dtype="float32",
                    properties={
                        "description": f"Attention {proj} projection matrix",
                        "parameter_count": 25,
                    },
                )
            )

            hg.add_vertex(
                Vertex(
                    id=f"attn_{proj}",
                    type="tensor",
                    shape=(None, 5, 5),
                    dtype="float32",
                    properties={"description": f"Attention {proj} vectors"},
                )
            )

            hg.add_hyperedge(
                Hyperedge(
                    id=f"project_{proj}",
                    sources=["embeddings", f"attn_{proj}_weights"],
                    targets=[f"attn_{proj}"],
                    operation="linear",
                    properties={"description": f"Project to {proj} space"},
                )
            )

        # Attention scores computation (Q @ K^T)
        hg.add_vertex(
            Vertex(
                id="attn_scores",
                type="tensor",
                shape=(None, 5, 5),  # (batch, seq_len, seq_len)
                dtype="float32",
                properties={"description": "Attention scores before softmax"},
            )
        )

        hg.add_hyperedge(
            Hyperedge(
                id="compute_attention_scores",
                sources=["attn_query", "attn_key"],
                targets=["attn_scores"],
                operation="matmul_scaled",
                properties={
                    "description": "Compute scaled dot-product attention scores",
                    "scale_factor": 1.0 / (5**0.5),  # 1/sqrt(d_k)
                },
            )
        )

        # Attention weights (softmax)
        hg.add_vertex(
            Vertex(
                id="attn_weights",
                type="tensor",
                shape=(None, 5, 5),
                dtype="float32",
                properties={"description": "Attention weights after softmax"},
            )
        )

        hg.add_hyperedge(
            Hyperedge(
                id="apply_softmax",
                sources=["attn_scores"],
                targets=["attn_weights"],
                operation="softmax",
                properties={"description": "Normalize attention scores", "axis": -1},
            )
        )

        # Attention output (weights @ V)
        hg.add_vertex(
            Vertex(
                id="attn_output_pre",
                type="tensor",
                shape=(None, 5, 5),
                dtype="float32",
                properties={"description": "Attention output before projection"},
            )
        )

        hg.add_hyperedge(
            Hyperedge(
                id="apply_attention",
                sources=["attn_weights", "attn_value"],
                targets=["attn_output_pre"],
                operation="matmul",
                properties={"description": "Apply attention to values"},
            )
        )

        # Attention output projection
        hg.add_vertex(
            Vertex(
                id="attn_output_weights",
                type="parameter",
                shape=(5, 5),
                dtype="float32",
                properties={
                    "description": "Attention output projection matrix",
                    "parameter_count": 25,
                },
            )
        )

        hg.add_vertex(
            Vertex(
                id="attn_output",
                type="tensor",
                shape=(None, 5, 5),
                dtype="float32",
                properties={"description": "Attention block output"},
            )
        )

        hg.add_hyperedge(
            Hyperedge(
                id="project_output",
                sources=["attn_output_pre", "attn_output_weights"],
                targets=["attn_output"],
                operation="linear",
                properties={"description": "Project attention output"},
            )
        )

        # Residual connection + Layer norm
        hg.add_vertex(
            Vertex(
                id="attn_residual",
                type="tensor",
                shape=(None, 5, 5),
                dtype="float32",
                properties={"description": "After attention residual connection"},
            )
        )

        hg.add_hyperedge(
            Hyperedge(
                id="attention_residual",
                sources=["embeddings", "attn_output"],
                targets=["attn_residual"],
                operation="add",
                properties={"description": "Residual connection for attention"},
            )
        )

        # Layer norm parameters
        hg.add_vertex(
            Vertex(
                id="ln1_weight",
                type="parameter",
                shape=(5,),
                dtype="float32",
                properties={"description": "Layer norm 1 weight", "parameter_count": 5},
            )
        )

        hg.add_vertex(
            Vertex(
                id="ln1_bias",
                type="parameter",
                shape=(5,),
                dtype="float32",
                properties={"description": "Layer norm 1 bias", "parameter_count": 5},
            )
        )

        hg.add_vertex(
            Vertex(
                id="ln1_output",
                type="tensor",
                shape=(None, 5, 5),
                dtype="float32",
                properties={"description": "After layer norm 1"},
            )
        )

        hg.add_hyperedge(
            Hyperedge(
                id="layer_norm_1",
                sources=["attn_residual", "ln1_weight", "ln1_bias"],
                targets=["ln1_output"],
                operation="layer_norm",
                properties={"description": "Layer normalization after attention", "epsilon": 1e-5},
            )
        )

        # Feed-forward network
        hg.add_vertex(
            Vertex(
                id="ffn_w1",
                type="parameter",
                shape=(5, 5),
                dtype="float32",
                properties={"description": "FFN first layer weight", "parameter_count": 25},
            )
        )

        hg.add_vertex(
            Vertex(
                id="ffn_hidden",
                type="tensor",
                shape=(None, 5, 5),
                dtype="float32",
                properties={"description": "FFN hidden activation"},
            )
        )

        hg.add_hyperedge(
            Hyperedge(
                id="ffn_layer1",
                sources=["ln1_output", "ffn_w1"],
                targets=["ffn_hidden"],
                operation="linear_gelu",
                properties={"description": "FFN first layer with GELU activation"},
            )
        )

        hg.add_vertex(
            Vertex(
                id="ffn_w2",
                type="parameter",
                shape=(5, 5),
                dtype="float32",
                properties={"description": "FFN second layer weight", "parameter_count": 25},
            )
        )

        hg.add_vertex(
            Vertex(
                id="ffn_output",
                type="tensor",
                shape=(None, 5, 5),
                dtype="float32",
                properties={"description": "FFN output"},
            )
        )

        hg.add_hyperedge(
            Hyperedge(
                id="ffn_layer2",
                sources=["ffn_hidden", "ffn_w2"],
                targets=["ffn_output"],
                operation="linear",
                properties={"description": "FFN second layer"},
            )
        )

        # FFN residual + layer norm
        hg.add_vertex(
            Vertex(
                id="ffn_residual",
                type="tensor",
                shape=(None, 5, 5),
                dtype="float32",
                properties={"description": "After FFN residual connection"},
            )
        )

        hg.add_hyperedge(
            Hyperedge(
                id="ffn_residual_add",
                sources=["ln1_output", "ffn_output"],
                targets=["ffn_residual"],
                operation="add",
                properties={"description": "Residual connection for FFN"},
            )
        )

        # Output projection
        hg.add_vertex(
            Vertex(
                id="output_weights",
                type="parameter",
                shape=(5, 10),
                dtype="float32",
                properties={
                    "description": "Output projection to vocabulary",
                    "parameter_count": 50,
                },
            )
        )

        hg.add_vertex(
            Vertex(
                id="logits",
                type="tensor",
                shape=(None, 5, 10),
                dtype="float32",
                properties={"description": "Output logits over vocabulary"},
            )
        )

        hg.add_hyperedge(
            Hyperedge(
                id="output_projection",
                sources=["ffn_residual", "output_weights"],
                targets=["logits"],
                operation="linear",
                properties={"description": "Project to vocabulary size"},
            )
        )

        return hg

    def get_computational_paths(self, start_vertex: str, end_vertex: str) -> List[List[str]]:
        """
        Find all computational paths from start to end vertex.

        This is useful for understanding data flow through the model.
        """
        paths = []
        visited = set()

        def dfs(current: str, path: List[str]):
            if current == end_vertex:
                paths.append(path.copy())
                return

            visited.add(current)
            for edge in self.hyperedges.values():
                if current in edge.sources:
                    for target in edge.targets:
                        if target not in visited:
                            dfs(target, path + [edge.id, target])
            visited.remove(current)

        dfs(start_vertex, [start_vertex])
        return paths

    def export_graphviz(self, filepath: Optional[Path] = None) -> str:
        """
        Export hypergraph to Graphviz DOT format for visualization.

        Note: Hyperedges are represented using intermediate nodes.
        """
        lines = ["digraph TinyTransformer {"]
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box];")
        lines.append("")

        # Add vertices
        for v_id, vertex in self.vertices.items():
            label = f"{v_id}\\n{vertex.type}"
            if vertex.shape:
                label += f"\\nshape: {vertex.shape}"

            color = {
                "tensor": "lightblue",
                "parameter": "lightgreen",
                "operation": "lightyellow",
            }.get(vertex.type, "white")

            lines.append(f'  "{v_id}" [label="{label}", fillcolor="{color}", style=filled];')

        lines.append("")

        # Add hyperedges as intermediate nodes
        for e_id, edge in self.hyperedges.items():
            lines.append(
                f'  "{e_id}" [label="{edge.operation}", shape=diamond, '
                'fillcolor="orange", style=filled];'
            )

            # Connect sources to hyperedge
            for source in edge.sources:
                lines.append(f'  "{source}" -> "{e_id}";')

            # Connect hyperedge to targets
            for target in edge.targets:
                lines.append(f'  "{e_id}" -> "{target}";')

        lines.append("}")
        dot_str = "\n".join(lines)

        if filepath:
            with open(filepath, "w") as f:
                f.write(dot_str)

        return dot_str
