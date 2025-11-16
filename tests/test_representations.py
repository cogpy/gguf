"""
Tests for model representation modules.
"""

import json
import pytest
from pathlib import Path

from gguf_workbench.representations import (
    HypergraphRepresentation,
    GraphRepresentation,
    SymbolicRepresentation,
    RepresentationComparator,
)
from gguf_workbench.representations.hypergraph import Vertex, Hyperedge
from gguf_workbench.representations.graph import Node, Edge
from gguf_workbench.representations.symbolic import SymbolicExpression, Parameter


class TestHypergraphRepresentation:
    """Test hypergraph representation."""
    
    def test_create_empty_hypergraph(self):
        """Test creating empty hypergraph."""
        hg = HypergraphRepresentation()
        assert len(hg.vertices) == 0
        assert len(hg.hyperedges) == 0
    
    def test_add_vertex(self):
        """Test adding vertices."""
        hg = HypergraphRepresentation()
        v = Vertex(id="v1", type="tensor", shape=(5, 10))
        hg.add_vertex(v)
        assert "v1" in hg.vertices
        assert hg.vertices["v1"].shape == (5, 10)
    
    def test_add_hyperedge(self):
        """Test adding hyperedges."""
        hg = HypergraphRepresentation()
        v1 = Vertex(id="v1", type="tensor")
        v2 = Vertex(id="v2", type="tensor")
        v3 = Vertex(id="v3", type="tensor")
        hg.add_vertex(v1)
        hg.add_vertex(v2)
        hg.add_vertex(v3)
        
        edge = Hyperedge(
            id="e1",
            sources=["v1", "v2"],
            targets=["v3"],
            operation="add"
        )
        hg.add_hyperedge(edge)
        assert "e1" in hg.hyperedges
    
    def test_hyperedge_validation(self):
        """Test that hyperedge validates vertex existence."""
        hg = HypergraphRepresentation()
        edge = Hyperedge(
            id="e1",
            sources=["nonexistent"],
            targets=["also_nonexistent"],
            operation="add"
        )
        with pytest.raises(ValueError):
            hg.add_hyperedge(edge)
    
    def test_get_incident_hyperedges(self):
        """Test getting incident hyperedges."""
        hg = HypergraphRepresentation()
        for i in range(3):
            hg.add_vertex(Vertex(id=f"v{i}", type="tensor"))
        
        hg.add_hyperedge(Hyperedge(
            id="e1", sources=["v0"], targets=["v1"], operation="op1"
        ))
        hg.add_hyperedge(Hyperedge(
            id="e2", sources=["v1"], targets=["v2"], operation="op2"
        ))
        
        # v1 should be incident to both edges
        incident = hg.get_incident_hyperedges("v1")
        assert len(incident) == 2
    
    def test_from_tiny_transformer(self):
        """Test creating hypergraph from tiny transformer."""
        hg = HypergraphRepresentation.from_tiny_transformer()
        
        # Check that model was created
        assert len(hg.vertices) > 0
        assert len(hg.hyperedges) > 0
        assert hg.metadata["model_name"] == "TinyTransformer"
        
        # Check some key components exist
        assert "input_tokens" in hg.vertices
        assert "embedding_weights" in hg.vertices
        assert "logits" in hg.vertices
    
    def test_statistics(self):
        """Test statistics computation."""
        hg = HypergraphRepresentation.from_tiny_transformer()
        stats = hg.get_statistics()
        
        assert "vertex_count" in stats
        assert "hyperedge_count" in stats
        assert stats["vertex_count"] > 0
        assert stats["hyperedge_count"] > 0
    
    def test_to_json(self):
        """Test JSON export."""
        hg = HypergraphRepresentation.from_tiny_transformer()
        json_str = hg.to_json()
        
        # Should be valid JSON
        data = json.loads(json_str)
        assert "metadata" in data
        assert "vertices" in data
        assert "hyperedges" in data
    
    def test_export_graphviz(self):
        """Test Graphviz export."""
        hg = HypergraphRepresentation.from_tiny_transformer()
        dot_str = hg.export_graphviz()
        
        assert "digraph TinyTransformer" in dot_str
        assert "rankdir=TB" in dot_str


class TestGraphRepresentation:
    """Test DAG representation."""
    
    def test_create_empty_graph(self):
        """Test creating empty graph."""
        g = GraphRepresentation()
        assert len(g.nodes) == 0
        assert len(g.edges) == 0
    
    def test_add_node(self):
        """Test adding nodes."""
        g = GraphRepresentation()
        n = Node(id="n1", type="tensor", shape=(5, 10))
        g.add_node(n)
        assert "n1" in g.nodes
    
    def test_add_edge(self):
        """Test adding edges."""
        g = GraphRepresentation()
        g.add_node(Node(id="n1", type="tensor"))
        g.add_node(Node(id="n2", type="tensor"))
        
        e = Edge(id="e1", source="n1", target="n2")
        g.add_edge(e)
        assert "e1" in g.edges
    
    def test_edge_validation(self):
        """Test edge validation."""
        g = GraphRepresentation()
        e = Edge(id="e1", source="nonexistent", target="also_nonexistent")
        with pytest.raises(ValueError):
            g.add_edge(e)
    
    def test_get_predecessors_successors(self):
        """Test predecessor and successor queries."""
        g = GraphRepresentation()
        for i in range(3):
            g.add_node(Node(id=f"n{i}", type="tensor"))
        
        g.add_edge(Edge(id="e1", source="n0", target="n1"))
        g.add_edge(Edge(id="e2", source="n1", target="n2"))
        
        assert g.get_predecessors("n1") == ["n0"]
        assert g.get_successors("n1") == ["n2"]
    
    def test_topological_sort(self):
        """Test topological sorting."""
        g = GraphRepresentation.from_tiny_transformer()
        topo_order = g.topological_sort()
        
        # Should have all nodes
        assert len(topo_order) == len(g.nodes)
        
        # Input should come before output
        assert topo_order.index("input_tokens") < topo_order.index("logits")
    
    def test_from_tiny_transformer(self):
        """Test creating DAG from tiny transformer."""
        g = GraphRepresentation.from_tiny_transformer()
        
        assert len(g.nodes) > 0
        assert len(g.edges) > 0
        assert g.metadata["model_name"] == "TinyTransformer"
    
    def test_to_json(self):
        """Test JSON export."""
        g = GraphRepresentation.from_tiny_transformer()
        json_str = g.to_json()
        
        data = json.loads(json_str)
        assert "metadata" in data
        assert "nodes" in data
        assert "edges" in data
        assert "topological_order" in data


class TestSymbolicRepresentation:
    """Test symbolic representation."""
    
    def test_create_empty_symbolic(self):
        """Test creating empty symbolic representation."""
        sr = SymbolicRepresentation()
        assert len(sr.parameters) == 0
        assert len(sr.expressions) == 0
    
    def test_add_parameter(self):
        """Test adding parameters."""
        sr = SymbolicRepresentation()
        p = Parameter(name="W", shape=(5, 10), dtype="float32", description="Weight matrix")
        sr.add_parameter(p)
        assert "W" in sr.parameters
    
    def test_add_expression(self):
        """Test adding expressions."""
        sr = SymbolicRepresentation()
        expr = SymbolicExpression(
            name="y",
            expression="Wx + b",
            dependencies=["W", "x", "b"]
        )
        sr.add_expression(expr)
        assert "y" in sr.expressions
    
    def test_from_tiny_transformer(self):
        """Test creating symbolic representation from tiny transformer."""
        sr = SymbolicRepresentation.from_tiny_transformer()
        
        assert len(sr.parameters) > 0
        assert len(sr.expressions) > 0
        assert sr.metadata["model_name"] == "TinyTransformer"
        
        # Check key components
        assert "E" in sr.parameters  # Embedding
        assert "W^Q" in sr.parameters  # Query projection
        assert "logits" in sr.expressions  # Output
    
    def test_to_json(self):
        """Test JSON export."""
        sr = SymbolicRepresentation.from_tiny_transformer()
        json_str = sr.to_json()
        
        data = json.loads(json_str)
        assert "metadata" in data
        assert "parameters" in data
        assert "expressions" in data
        assert "notation" in data
    
    def test_export_markdown(self):
        """Test Markdown export."""
        sr = SymbolicRepresentation.from_tiny_transformer()
        md_str = sr.export_markdown()
        
        assert "# TinyTransformer" in md_str
        assert "## Parameters" in md_str
        assert "## Forward Pass" in md_str
    
    def test_to_latex(self):
        """Test LaTeX export."""
        sr = SymbolicRepresentation.from_tiny_transformer()
        latex_str = sr.to_latex()
        
        assert "\\documentclass{article}" in latex_str
        assert "\\begin{align}" in latex_str
        assert "\\end{document}" in latex_str


class TestRepresentationComparator:
    """Test representation comparator."""
    
    def test_create_empty_comparator(self):
        """Test creating empty comparator."""
        comp = RepresentationComparator()
        assert len(comp.metrics) == 0
    
    def test_create_default_comparison(self):
        """Test creating default comparison."""
        comp = RepresentationComparator.create_default_comparison()
        
        # Should have metrics for common formats
        assert "Hypergraph" in comp.metrics
        assert "DAG (Directed Graph)" in comp.metrics
        assert "Symbolic/Algebraic" in comp.metrics
        assert "GGUF (Binary)" in comp.metrics
    
    def test_compare_all(self):
        """Test comparison generation."""
        comp = RepresentationComparator.create_default_comparison()
        comparison = comp.compare_all()
        
        assert "representations" in comparison
        assert "rankings" in comparison
        assert "summary" in comparison
        
        # Check rankings exist for all dimensions
        assert "completeness_score" in comparison["rankings"]
        assert "transparency_score" in comparison["rankings"]
        assert "efficiency_score" in comparison["rankings"]
    
    def test_to_json(self):
        """Test JSON export."""
        comp = RepresentationComparator.create_default_comparison()
        json_str = comp.to_json()
        
        data = json.loads(json_str)
        assert "representations" in data
        assert "summary" in data
    
    def test_to_markdown(self):
        """Test Markdown export."""
        comp = RepresentationComparator.create_default_comparison()
        md_str = comp.to_markdown()
        
        assert "# Model Representation Comparison" in md_str
        assert "## Score Summary" in md_str
        assert "## Rankings" in md_str


class TestIntegration:
    """Integration tests."""
    
    def test_all_representations_consistency(self):
        """Test that all representations agree on model structure."""
        hg = HypergraphRepresentation.from_tiny_transformer()
        g = GraphRepresentation.from_tiny_transformer()
        sr = SymbolicRepresentation.from_tiny_transformer()
        
        # All should represent the same model
        assert hg.metadata["model_name"] == g.metadata["model_name"] == sr.metadata["model_name"]
        assert hg.metadata["vocab_size"] == g.metadata["vocab_size"] == sr.metadata["vocab_size"]
        assert hg.metadata["embedding_dim"] == g.metadata["embedding_dim"]
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow of generating and exporting representations."""
        # Create representations
        hg = HypergraphRepresentation.from_tiny_transformer()
        g = GraphRepresentation.from_tiny_transformer()
        sr = SymbolicRepresentation.from_tiny_transformer()
        comp = RepresentationComparator.create_default_comparison()
        
        # Export to files
        hg.to_json(tmp_path / "hypergraph.json")
        g.to_json(tmp_path / "graph.json")
        sr.to_json(tmp_path / "symbolic.json")
        comp.to_json(tmp_path / "comparison.json")
        
        # Verify files were created
        assert (tmp_path / "hypergraph.json").exists()
        assert (tmp_path / "graph.json").exists()
        assert (tmp_path / "symbolic.json").exists()
        assert (tmp_path / "comparison.json").exists()
        
        # Verify they can be read back
        with open(tmp_path / "hypergraph.json") as f:
            hg_data = json.load(f)
            assert hg_data["metadata"]["model_name"] == "TinyTransformer"
