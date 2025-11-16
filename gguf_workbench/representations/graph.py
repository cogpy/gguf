"""
Directed Acyclic Graph (DAG) representation for transformer models.

A standard directed graph where edges connect single nodes, representing
simpler pairwise relationships between operations and tensors.
"""

import json
from typing import Any, Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Node:
    """A node in the graph representing a tensor or operation."""
    id: str
    type: str  # 'tensor', 'operation', 'parameter'
    properties: Dict[str, Any] = field(default_factory=dict)
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class Edge:
    """A directed edge connecting two nodes."""
    id: str
    source: str  # Source node ID
    target: str  # Target node ID
    label: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


class GraphRepresentation:
    """
    Directed Acyclic Graph (DAG) representation of a transformer model.
    
    In a DAG:
    - Nodes represent tensors, operations, and parameters
    - Edges represent data flow between nodes
    - Operations are explicit nodes (unlike hypergraph where they're edges)
    
    Advantages:
    - Simpler structure than hypergraph
    - Well-understood algorithms (topological sort, shortest path, etc.)
    - Easy visualization
    - Standard format for many frameworks (TensorFlow, PyTorch graphs)
    
    Disadvantages:
    - Multi-input operations require operation nodes
    - Less expressive than hypergraph for complex operations
    - Can be more verbose (more nodes needed)
    """

    def __init__(self):
        """Initialize empty graph."""
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.metadata: Dict[str, Any] = {}

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        if edge.source not in self.nodes:
            raise ValueError(f"Source node {edge.source} not found")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node {edge.target} not found")
        self.edges[edge.id] = edge

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get all predecessor nodes (inputs)."""
        return [e.source for e in self.edges.values() if e.target == node_id]

    def get_successors(self, node_id: str) -> List[str]:
        """Get all successor nodes (outputs)."""
        return [e.target for e in self.edges.values() if e.source == node_id]

    def topological_sort(self) -> List[str]:
        """
        Return nodes in topological order (dependency order).
        
        Useful for understanding execution order of operations.
        """
        in_degree = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges.values():
            in_degree[edge.target] += 1
        
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            for successor in self.get_successors(node_id):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "metadata": self.metadata,
            "nodes": {
                n_id: n.to_dict() for n_id, n in self.nodes.items()
            },
            "edges": {
                e_id: e.to_dict() for e_id, e in self.edges.items()
            },
            "statistics": self.get_statistics(),
            "topological_order": self.topological_sort(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Compute statistics about the graph."""
        node_types = {}
        for n in self.nodes.values():
            node_types[n.type] = node_types.get(n.type, 0) + 1

        in_degrees = {}
        out_degrees = {}
        for node_id in self.nodes:
            in_degrees[node_id] = len(self.get_predecessors(node_id))
            out_degrees[node_id] = len(self.get_successors(node_id))

        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_types": node_types,
            "average_in_degree": sum(in_degrees.values()) / len(in_degrees) if in_degrees else 0,
            "average_out_degree": sum(out_degrees.values()) / len(out_degrees) if out_degrees else 0,
            "max_in_degree": max(in_degrees.values()) if in_degrees else 0,
            "max_out_degree": max(out_degrees.values()) if out_degrees else 0,
        }

    def to_json(self, filepath: Optional[Path] = None, indent: int = 2) -> str:
        """Export graph to JSON format."""
        json_str = json.dumps(self.to_dict(), indent=indent)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_tiny_transformer(cls, model_path: Optional[Path] = None) -> 'GraphRepresentation':
        """
        Create DAG representation from tiny transformer model.
        
        In this representation, operations are explicit nodes.
        """
        g = cls()
        
        g.metadata = {
            "model_name": "TinyTransformer",
            "architecture": "transformer",
            "representation": "DAG",
            "vocab_size": 10,
            "embedding_dim": 5,
            "context_length": 5,
            "num_blocks": 1,
            "num_heads": 1,
            "feedforward_dim": 5,
        }

        # Input
        g.add_node(Node(
            id="input_tokens",
            type="tensor",
            shape=(None, 5),
            dtype="int64",
            properties={"description": "Input token IDs"}
        ))

        # Embedding
        g.add_node(Node(
            id="embedding_weights",
            type="parameter",
            shape=(10, 5),
            dtype="float32",
            properties={"parameter_count": 50}
        ))

        g.add_node(Node(
            id="embed_op",
            type="operation",
            properties={"operation": "embedding_lookup"}
        ))

        g.add_node(Node(
            id="embeddings",
            type="tensor",
            shape=(None, 5, 5),
            dtype="float32"
        ))

        g.add_edge(Edge(id="e1", source="input_tokens", target="embed_op"))
        g.add_edge(Edge(id="e2", source="embedding_weights", target="embed_op"))
        g.add_edge(Edge(id="e3", source="embed_op", target="embeddings"))

        # Attention Q, K, V projections
        prev_tensor = "embeddings"
        qkv_tensors = []
        
        for i, proj in enumerate(["query", "key", "value"]):
            weight_id = f"attn_{proj}_weights"
            op_id = f"attn_{proj}_op"
            tensor_id = f"attn_{proj}"
            
            g.add_node(Node(
                id=weight_id,
                type="parameter",
                shape=(5, 5),
                dtype="float32",
                properties={"parameter_count": 25}
            ))
            
            g.add_node(Node(
                id=op_id,
                type="operation",
                properties={"operation": "linear"}
            ))
            
            g.add_node(Node(
                id=tensor_id,
                type="tensor",
                shape=(None, 5, 5),
                dtype="float32"
            ))
            
            g.add_edge(Edge(id=f"e_attn_{proj}_1", source=prev_tensor, target=op_id))
            g.add_edge(Edge(id=f"e_attn_{proj}_2", source=weight_id, target=op_id))
            g.add_edge(Edge(id=f"e_attn_{proj}_3", source=op_id, target=tensor_id))
            
            qkv_tensors.append(tensor_id)

        # Attention scores
        g.add_node(Node(
            id="attn_scores_op",
            type="operation",
            properties={"operation": "matmul_scaled", "scale": 0.447}
        ))
        
        g.add_node(Node(
            id="attn_scores",
            type="tensor",
            shape=(None, 5, 5),
            dtype="float32"
        ))
        
        g.add_edge(Edge(id="e_scores_1", source="attn_query", target="attn_scores_op"))
        g.add_edge(Edge(id="e_scores_2", source="attn_key", target="attn_scores_op"))
        g.add_edge(Edge(id="e_scores_3", source="attn_scores_op", target="attn_scores"))

        # Softmax
        g.add_node(Node(
            id="softmax_op",
            type="operation",
            properties={"operation": "softmax", "axis": -1}
        ))
        
        g.add_node(Node(
            id="attn_weights",
            type="tensor",
            shape=(None, 5, 5),
            dtype="float32"
        ))
        
        g.add_edge(Edge(id="e_softmax_1", source="attn_scores", target="softmax_op"))
        g.add_edge(Edge(id="e_softmax_2", source="softmax_op", target="attn_weights"))

        # Apply attention
        g.add_node(Node(
            id="apply_attn_op",
            type="operation",
            properties={"operation": "matmul"}
        ))
        
        g.add_node(Node(
            id="attn_output_pre",
            type="tensor",
            shape=(None, 5, 5),
            dtype="float32"
        ))
        
        g.add_edge(Edge(id="e_apply_1", source="attn_weights", target="apply_attn_op"))
        g.add_edge(Edge(id="e_apply_2", source="attn_value", target="apply_attn_op"))
        g.add_edge(Edge(id="e_apply_3", source="apply_attn_op", target="attn_output_pre"))

        # Attention output projection
        g.add_node(Node(
            id="attn_output_weights",
            type="parameter",
            shape=(5, 5),
            dtype="float32",
            properties={"parameter_count": 25}
        ))
        
        g.add_node(Node(
            id="attn_proj_op",
            type="operation",
            properties={"operation": "linear"}
        ))
        
        g.add_node(Node(
            id="attn_output",
            type="tensor",
            shape=(None, 5, 5),
            dtype="float32"
        ))
        
        g.add_edge(Edge(id="e_attn_proj_1", source="attn_output_pre", target="attn_proj_op"))
        g.add_edge(Edge(id="e_attn_proj_2", source="attn_output_weights", target="attn_proj_op"))
        g.add_edge(Edge(id="e_attn_proj_3", source="attn_proj_op", target="attn_output"))

        # Residual connection
        g.add_node(Node(
            id="residual1_op",
            type="operation",
            properties={"operation": "add"}
        ))
        
        g.add_node(Node(
            id="attn_residual",
            type="tensor",
            shape=(None, 5, 5),
            dtype="float32"
        ))
        
        g.add_edge(Edge(id="e_res1_1", source="embeddings", target="residual1_op"))
        g.add_edge(Edge(id="e_res1_2", source="attn_output", target="residual1_op"))
        g.add_edge(Edge(id="e_res1_3", source="residual1_op", target="attn_residual"))

        # Layer norm 1
        g.add_node(Node(id="ln1_weight", type="parameter", shape=(5,), dtype="float32"))
        g.add_node(Node(id="ln1_bias", type="parameter", shape=(5,), dtype="float32"))
        g.add_node(Node(id="ln1_op", type="operation", properties={"operation": "layer_norm"}))
        g.add_node(Node(id="ln1_output", type="tensor", shape=(None, 5, 5), dtype="float32"))
        
        g.add_edge(Edge(id="e_ln1_1", source="attn_residual", target="ln1_op"))
        g.add_edge(Edge(id="e_ln1_2", source="ln1_weight", target="ln1_op"))
        g.add_edge(Edge(id="e_ln1_3", source="ln1_bias", target="ln1_op"))
        g.add_edge(Edge(id="e_ln1_4", source="ln1_op", target="ln1_output"))

        # FFN layer 1
        g.add_node(Node(id="ffn_w1", type="parameter", shape=(5, 5), dtype="float32"))
        g.add_node(Node(id="ffn1_op", type="operation", properties={"operation": "linear_gelu"}))
        g.add_node(Node(id="ffn_hidden", type="tensor", shape=(None, 5, 5), dtype="float32"))
        
        g.add_edge(Edge(id="e_ffn1_1", source="ln1_output", target="ffn1_op"))
        g.add_edge(Edge(id="e_ffn1_2", source="ffn_w1", target="ffn1_op"))
        g.add_edge(Edge(id="e_ffn1_3", source="ffn1_op", target="ffn_hidden"))

        # FFN layer 2
        g.add_node(Node(id="ffn_w2", type="parameter", shape=(5, 5), dtype="float32"))
        g.add_node(Node(id="ffn2_op", type="operation", properties={"operation": "linear"}))
        g.add_node(Node(id="ffn_output", type="tensor", shape=(None, 5, 5), dtype="float32"))
        
        g.add_edge(Edge(id="e_ffn2_1", source="ffn_hidden", target="ffn2_op"))
        g.add_edge(Edge(id="e_ffn2_2", source="ffn_w2", target="ffn2_op"))
        g.add_edge(Edge(id="e_ffn2_3", source="ffn2_op", target="ffn_output"))

        # FFN residual
        g.add_node(Node(id="residual2_op", type="operation", properties={"operation": "add"}))
        g.add_node(Node(id="ffn_residual", type="tensor", shape=(None, 5, 5), dtype="float32"))
        
        g.add_edge(Edge(id="e_res2_1", source="ln1_output", target="residual2_op"))
        g.add_edge(Edge(id="e_res2_2", source="ffn_output", target="residual2_op"))
        g.add_edge(Edge(id="e_res2_3", source="residual2_op", target="ffn_residual"))

        # Output projection
        g.add_node(Node(id="output_weights", type="parameter", shape=(5, 10), dtype="float32"))
        g.add_node(Node(id="output_op", type="operation", properties={"operation": "linear"}))
        g.add_node(Node(id="logits", type="tensor", shape=(None, 5, 10), dtype="float32"))
        
        g.add_edge(Edge(id="e_out_1", source="ffn_residual", target="output_op"))
        g.add_edge(Edge(id="e_out_2", source="output_weights", target="output_op"))
        g.add_edge(Edge(id="e_out_3", source="output_op", target="logits"))

        return g

    def export_graphviz(self, filepath: Optional[Path] = None) -> str:
        """Export graph to Graphviz DOT format."""
        lines = ["digraph TinyTransformer {"]
        lines.append("  rankdir=TB;")
        lines.append("")
        
        for n_id, node in self.nodes.items():
            label = f"{n_id}"
            if node.shape:
                label += f"\\nshape: {node.shape}"
            
            if node.type == "tensor":
                shape, color = "box", "lightblue"
            elif node.type == "parameter":
                shape, color = "box", "lightgreen"
            else:  # operation
                shape, color = "ellipse", "lightyellow"
            
            lines.append(f'  "{n_id}" [label="{label}", shape={shape}, fillcolor="{color}", style=filled];')
        
        lines.append("")
        
        for edge in self.edges.values():
            label = f' [label="{edge.label}"]' if edge.label else ""
            lines.append(f'  "{edge.source}" -> "{edge.target}"{label};')
        
        lines.append("}")
        dot_str = "\n".join(lines)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(dot_str)
        
        return dot_str
