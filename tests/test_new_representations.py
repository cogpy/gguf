"""
Tests for new representation formats: AIML, OpenCog, and TOML Hypergraph.
"""

import pytest
from pathlib import Path
import json

from gguf_workbench.representations import (
    AIMLRepresentation,
    OpenCogAtomSpaceRepresentation,
    TOMLHypergraphRepresentation,
)


class TestAIMLRepresentation:
    """Tests for AIML representation."""
    
    def test_create_empty_aiml(self):
        """Test creating an empty AIML representation."""
        aiml = AIMLRepresentation()
        assert len(aiml.categories) == 0
        assert aiml.topic == "tinytransformer"
        assert isinstance(aiml.metadata, dict)
    
    def test_add_category(self):
        """Test adding categories."""
        aiml = AIMLRepresentation()
        aiml.add_category("HELLO", "Hi there!")
        
        assert len(aiml.categories) == 1
        assert aiml.categories[0]["pattern"] == "HELLO"
        assert aiml.categories[0]["template"] == "Hi there!"
    
    def test_add_category_with_that(self):
        """Test adding category with context (that)."""
        aiml = AIMLRepresentation()
        aiml.add_category("YES", "Great!", that="DO YOU LIKE IT")
        
        assert len(aiml.categories) == 1
        assert aiml.categories[0]["that"] == "DO YOU LIKE IT"
    
    def test_from_tiny_transformer(self):
        """Test creating AIML from tiny transformer."""
        aiml = AIMLRepresentation.from_tiny_transformer()
        
        # Should have multiple categories
        assert len(aiml.categories) > 0
        
        # Should have metadata
        assert aiml.metadata["model_name"] == "TinyTransformer"
        assert aiml.metadata["vocab_size"] == 10
        assert aiml.metadata["embedding_dim"] == 5
    
    def test_to_xml(self):
        """Test converting to XML."""
        aiml = AIMLRepresentation()
        aiml.add_category("TEST", "Response")
        
        xml = aiml.to_xml()
        
        # Check XML structure
        assert '<?xml version="1.0" ?>' in xml
        assert '<aiml version="2.0">' in xml
        assert '<category>' in xml
        assert '<pattern>TEST</pattern>' in xml
        assert '<template>Response</template>' in xml
    
    def test_to_json(self):
        """Test converting to JSON."""
        aiml = AIMLRepresentation.from_tiny_transformer()
        
        json_data = aiml.to_json()
        
        assert "metadata" in json_data
        assert "categories" in json_data
        assert "category_count" in json_data
        assert json_data["category_count"] == len(aiml.categories)
    
    def test_save_xml(self, tmp_path):
        """Test saving to XML file."""
        aiml = AIMLRepresentation.from_tiny_transformer()
        output_file = tmp_path / "test.aiml"
        
        aiml.save_xml(output_file)
        
        assert output_file.exists()
        content = output_file.read_text()
        assert '<aiml version="2.0">' in content
    
    def test_save_json(self, tmp_path):
        """Test saving to JSON file."""
        aiml = AIMLRepresentation.from_tiny_transformer()
        output_file = tmp_path / "test.json"
        
        aiml.save_json(output_file)
        
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert "categories" in data


class TestOpenCogAtomSpaceRepresentation:
    """Tests for OpenCog AtomSpace representation."""
    
    def test_create_empty_atomspace(self):
        """Test creating an empty AtomSpace."""
        atomspace = OpenCogAtomSpaceRepresentation()
        assert len(atomspace.atoms) == 0
        assert isinstance(atomspace.metadata, dict)
        assert isinstance(atomspace.knowledge_base, list)
    
    def test_add_concept_node(self):
        """Test adding concept nodes."""
        atomspace = OpenCogAtomSpaceRepresentation()
        
        # Without truth value
        atom1 = atomspace.add_concept_node("Test")
        assert '(ConceptNode "Test")' in atom1
        
        # With truth value
        atom2 = atomspace.add_concept_node("Test2", tv=(0.9, 0.8))
        assert '(ConceptNode "Test2" (stv 0.9 0.8))' in atom2
        
        assert len(atomspace.atoms) == 2
    
    def test_add_predicate_node(self):
        """Test adding predicate nodes."""
        atomspace = OpenCogAtomSpaceRepresentation()
        
        atom = atomspace.add_predicate_node("hasProperty", tv=(1.0, 1.0))
        assert '(PredicateNode "hasProperty"' in atom
        assert len(atomspace.atoms) == 1
    
    def test_add_number_node(self):
        """Test adding number nodes."""
        atomspace = OpenCogAtomSpaceRepresentation()
        
        atom = atomspace.add_number_node(42.5)
        assert '(NumberNode "42.5")' in atom
    
    def test_add_inheritance_link(self):
        """Test adding inheritance links."""
        atomspace = OpenCogAtomSpaceRepresentation()
        
        link = atomspace.add_inheritance_link("Dog", "Animal", tv=(0.95, 0.9))
        
        assert "InheritanceLink" in link
        assert '"Dog"' in link
        assert '"Animal"' in link
        assert "stv 0.95 0.9" in link
    
    def test_add_evaluation_link(self):
        """Test adding evaluation links."""
        atomspace = OpenCogAtomSpaceRepresentation()
        
        link = atomspace.add_evaluation_link("hasColor", "Sky", "Blue", tv=(0.8, 0.7))
        
        assert "EvaluationLink" in link
        assert "hasColor" in link
        assert "Sky" in link
        assert "Blue" in link
    
    def test_add_member_link(self):
        """Test adding member links."""
        atomspace = OpenCogAtomSpaceRepresentation()
        
        link = atomspace.add_member_link("Element", "Set")
        
        assert "MemberLink" in link
        assert "Element" in link
        assert "Set" in link
    
    def test_add_execution_link(self):
        """Test adding execution links."""
        atomspace = OpenCogAtomSpaceRepresentation()
        
        link = atomspace.add_execution_link("Add", "X", "Y")
        
        assert "ExecutionLink" in link
        assert "SchemaNode" in link
        assert "Add" in link
    
    def test_from_tiny_transformer(self):
        """Test creating AtomSpace from tiny transformer."""
        atomspace = OpenCogAtomSpaceRepresentation.from_tiny_transformer()
        
        # Should have atoms
        assert len(atomspace.atoms) > 0
        
        # Should have metadata
        assert atomspace.metadata["model_name"] == "TinyTransformer"
        assert atomspace.metadata["vocab_size"] == 10
        
        # Should have knowledge base rules
        assert len(atomspace.knowledge_base) > 0
    
    def test_to_scheme(self):
        """Test converting to Scheme format."""
        atomspace = OpenCogAtomSpaceRepresentation.from_tiny_transformer()
        
        scheme = atomspace.to_scheme()
        
        # Check Scheme structure
        assert "(use-modules (opencog))" in scheme
        assert "(use-modules (opencog ure))" in scheme
        assert "(use-modules (opencog pln))" in scheme
        assert "ConceptNode" in scheme
        assert "InheritanceLink" in scheme
    
    def test_to_json(self):
        """Test converting to JSON."""
        atomspace = OpenCogAtomSpaceRepresentation.from_tiny_transformer()
        
        json_data = atomspace.to_json()
        
        assert "metadata" in json_data
        assert "atoms" in json_data
        assert "atom_count" in json_data
        assert json_data["atom_count"] == len(atomspace.atoms)
    
    def test_save_scheme(self, tmp_path):
        """Test saving to Scheme file."""
        atomspace = OpenCogAtomSpaceRepresentation.from_tiny_transformer()
        output_file = tmp_path / "test.scm"
        
        atomspace.save_scheme(output_file)
        
        assert output_file.exists()
        content = output_file.read_text()
        assert "(use-modules (opencog))" in content
    
    def test_save_json(self, tmp_path):
        """Test saving to JSON file."""
        atomspace = OpenCogAtomSpaceRepresentation.from_tiny_transformer()
        output_file = tmp_path / "test.json"
        
        atomspace.save_json(output_file)
        
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert "atoms" in data


class TestTOMLHypergraphRepresentation:
    """Tests for TOML Hypergraph representation."""
    
    def test_create_empty_toml_hypergraph(self):
        """Test creating an empty TOML hypergraph."""
        toml_hg = TOMLHypergraphRepresentation()
        assert len(toml_hg.vertices) == 0
        assert len(toml_hg.hyperedges) == 0
        assert isinstance(toml_hg.metadata, dict)
    
    def test_add_vertex(self):
        """Test adding vertices."""
        toml_hg = TOMLHypergraphRepresentation()
        
        toml_hg.add_vertex(
            "test_tensor",
            "tensor",
            shape=(None, 5),
            dtype="float32",
            properties={"description": "Test tensor"}
        )
        
        assert "test_tensor" in toml_hg.vertices
        vertex = toml_hg.vertices["test_tensor"]
        assert vertex["type"] == "tensor"
        assert vertex["dtype"] == "float32"
        assert vertex["shape"] == ["batch", 5]
    
    def test_add_hyperedge(self):
        """Test adding hyperedges."""
        toml_hg = TOMLHypergraphRepresentation()
        
        # Add vertices first
        toml_hg.add_vertex("v1", "tensor")
        toml_hg.add_vertex("v2", "tensor")
        toml_hg.add_vertex("v3", "tensor")
        
        # Add hyperedge
        toml_hg.add_hyperedge(
            "edge1",
            "matmul",
            sources=["v1", "v2"],
            targets=["v3"],
            properties={"description": "Matrix multiplication"}
        )
        
        assert "edge1" in toml_hg.hyperedges
        edge = toml_hg.hyperedges["edge1"]
        assert edge["operation"] == "matmul"
        assert edge["sources"] == ["v1", "v2"]
        assert edge["targets"] == ["v3"]
    
    def test_add_weights(self):
        """Test adding weights."""
        toml_hg = TOMLHypergraphRepresentation()
        toml_hg.add_vertex("weights", "parameter")
        
        weights = [[1.0, 2.0], [3.0, 4.0]]
        toml_hg.add_weights("weights", weights)
        
        assert "weights" in toml_hg.weights
        assert toml_hg.weights["weights"] == weights
    
    def test_from_tiny_transformer(self):
        """Test creating TOML hypergraph from tiny transformer."""
        toml_hg = TOMLHypergraphRepresentation.from_tiny_transformer(include_weights=True)
        
        # Should have vertices
        assert len(toml_hg.vertices) > 0
        
        # Should have hyperedges
        assert len(toml_hg.hyperedges) > 0
        
        # Should have metadata
        assert toml_hg.metadata["model_name"] == "TinyTransformer"
        assert toml_hg.metadata["representation"] == "hypergraph"
        
        # Should have weights
        assert len(toml_hg.weights) > 0
    
    def test_from_tiny_transformer_no_weights(self):
        """Test creating without weights."""
        toml_hg = TOMLHypergraphRepresentation.from_tiny_transformer(include_weights=False)
        
        assert len(toml_hg.vertices) > 0
        assert len(toml_hg.hyperedges) > 0
        assert len(toml_hg.weights) == 0
    
    def test_to_toml(self):
        """Test converting to TOML format."""
        toml_hg = TOMLHypergraphRepresentation.from_tiny_transformer(include_weights=False)
        
        toml_str = toml_hg.to_toml()
        
        # Check TOML structure
        assert "[metadata]" in toml_str
        assert "[vertices." in toml_str
        assert "[hyperedges." in toml_str
        assert "model_name = \"TinyTransformer\"" in toml_str
    
    def test_to_json(self):
        """Test converting to JSON."""
        toml_hg = TOMLHypergraphRepresentation.from_tiny_transformer()
        
        json_data = toml_hg.to_json()
        
        assert "metadata" in json_data
        assert "vertices" in json_data
        assert "hyperedges" in json_data
        assert "statistics" in json_data
        assert json_data["statistics"]["vertex_count"] == len(toml_hg.vertices)
    
    def test_save_toml(self, tmp_path):
        """Test saving to TOML file."""
        toml_hg = TOMLHypergraphRepresentation.from_tiny_transformer()
        output_file = tmp_path / "test.toml"
        
        toml_hg.save_toml(output_file)
        
        assert output_file.exists()
        content = output_file.read_text()
        assert "[metadata]" in content
    
    def test_save_json(self, tmp_path):
        """Test saving to JSON file."""
        toml_hg = TOMLHypergraphRepresentation.from_tiny_transformer()
        output_file = tmp_path / "test.json"
        
        toml_hg.save_json(output_file)
        
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert "vertices" in data


class TestIntegrationNewFormats:
    """Integration tests for new format representations."""
    
    def test_all_formats_have_consistent_metadata(self):
        """Test that all formats represent the same model metadata."""
        aiml = AIMLRepresentation.from_tiny_transformer()
        atomspace = OpenCogAtomSpaceRepresentation.from_tiny_transformer()
        toml_hg = TOMLHypergraphRepresentation.from_tiny_transformer()
        
        # All should have same model name
        assert aiml.metadata["model_name"] == "TinyTransformer"
        assert atomspace.metadata["model_name"] == "TinyTransformer"
        assert toml_hg.metadata["model_name"] == "TinyTransformer"
        
        # All should have same vocab size
        assert aiml.metadata["vocab_size"] == 10
        assert atomspace.metadata["vocab_size"] == 10
        assert toml_hg.metadata["vocab_size"] == 10
    
    def test_can_save_all_formats(self, tmp_path):
        """Test that all formats can be saved."""
        aiml = AIMLRepresentation.from_tiny_transformer()
        atomspace = OpenCogAtomSpaceRepresentation.from_tiny_transformer()
        toml_hg = TOMLHypergraphRepresentation.from_tiny_transformer()
        
        # Save AIML
        aiml.save_xml(tmp_path / "test.aiml")
        aiml.save_json(tmp_path / "aiml.json")
        
        # Save OpenCog
        atomspace.save_scheme(tmp_path / "test.scm")
        atomspace.save_json(tmp_path / "atomspace.json")
        
        # Save TOML Hypergraph
        toml_hg.save_toml(tmp_path / "test.toml")
        toml_hg.save_json(tmp_path / "toml_hg.json")
        
        # All files should exist
        assert (tmp_path / "test.aiml").exists()
        assert (tmp_path / "aiml.json").exists()
        assert (tmp_path / "test.scm").exists()
        assert (tmp_path / "atomspace.json").exists()
        assert (tmp_path / "test.toml").exists()
        assert (tmp_path / "toml_hg.json").exists()
