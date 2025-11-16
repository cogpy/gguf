"""
Tests for the generalized GGUF converter.
"""

import json
import pytest
from pathlib import Path
from gguf_workbench import GGUFConverter


class TestGGUFConverter:
    """Tests for GGUFConverter class."""
    
    @pytest.fixture
    def converter(self):
        """Create a converter for the tiny model."""
        return GGUFConverter("tinytf/tiny_model.gguf")
    
    def test_load_gguf(self, converter):
        """Test that GGUF file is loaded correctly."""
        assert converter.architecture is not None
        assert converter.model_info is not None
        assert "embedding_dim" in converter.model_info or "vocab_size" in converter.model_info
    
    def test_to_hypergraph(self, converter):
        """Test conversion to hypergraph."""
        hg = converter.to_hypergraph(include_weights=False)
        
        assert hg is not None
        assert len(hg.vertices) > 0
        assert len(hg.hyperedges) > 0
        assert hg.metadata is not None
    
    def test_to_dag(self, converter):
        """Test conversion to DAG."""
        dag = converter.to_dag()
        
        assert dag is not None
        assert len(dag.nodes) > 0
        assert len(dag.edges) > 0
        assert dag.metadata is not None
    
    def test_to_symbolic(self, converter):
        """Test conversion to symbolic representation."""
        symbolic = converter.to_symbolic()
        
        assert symbolic is not None
        assert len(symbolic.parameters) > 0
        assert len(symbolic.expressions) > 0
        assert symbolic.metadata is not None
    
    def test_to_aiml(self, converter):
        """Test conversion to AIML."""
        aiml = converter.to_aiml()
        
        assert aiml is not None
        assert len(aiml.categories) > 0
    
    def test_to_atomspace(self, converter):
        """Test conversion to OpenCog AtomSpace."""
        atomspace = converter.to_atomspace()
        
        assert atomspace is not None
        json_data = atomspace.to_json()
        assert json_data["atom_count"] > 0
    
    def test_to_toml_hypergraph(self, converter):
        """Test conversion to TOML hypergraph."""
        toml_hg = converter.to_toml_hypergraph(include_weights=False)
        
        assert toml_hg is not None
        stats = toml_hg.to_json()["statistics"]
        assert stats["vertex_count"] > 0
        assert stats["hyperedge_count"] > 0
    
    def test_export_all(self, converter, tmp_path):
        """Test exporting all formats."""
        output_dir = tmp_path / "converted"
        
        results = converter.export_all(
            str(output_dir),
            include_weights=False
        )
        
        # Check that all expected formats were created
        assert "hypergraph" in results
        assert "dag" in results
        assert "symbolic" in results
        assert "aiml" in results
        assert "atomspace" in results
        assert "toml" in results
        
        # Check that files exist
        for fmt, path in results.items():
            if isinstance(path, dict):
                for subpath in path.values():
                    assert Path(subpath).exists()
            else:
                assert Path(path).exists()
    
    def test_export_specific_format(self, converter, tmp_path):
        """Test exporting a specific format."""
        output_dir = tmp_path / "converted_specific"
        
        results = converter.export_all(
            str(output_dir),
            include_weights=False,
            formats=["hypergraph"]
        )
        
        assert "hypergraph" in results
        assert len(results) == 1  # Only hypergraph should be exported


class TestHypergraphToDAG:
    """Test conversion from hypergraph to DAG."""
    
    def test_hypergraph_to_dag(self):
        """Test that hypergraph can be converted to DAG."""
        converter = GGUFConverter("tinytf/tiny_model.gguf")
        hg = converter.to_hypergraph(include_weights=False)
        dag = hg.to_dag()
        
        # DAG should have more nodes than hypergraph vertices
        # because hyperedges become operation nodes
        assert len(dag.nodes) >= len(hg.vertices)
        
        # Check that metadata is preserved
        assert dag.metadata == hg.metadata


class TestModelInfoExtraction:
    """Test extraction of model information from different architectures."""
    
    def test_extract_tiny_transformer_info(self):
        """Test info extraction from tiny transformer."""
        converter = GGUFConverter("tinytf/tiny_model.gguf")
        info = converter.model_info
        
        # Should have extracted some information
        assert len(info) > 0
        assert "architecture" in info


class TestFileSize:
    """Test that structure-only exports are reasonably sized."""
    
    def test_structure_only_is_small(self, tmp_path):
        """Test that structure-only exports are much smaller than weights."""
        converter = GGUFConverter("tinytf/tiny_model.gguf")
        output_dir = tmp_path / "size_test"
        
        results = converter.export_all(
            str(output_dir),
            include_weights=False
        )
        
        # Get hypergraph file size
        hg_path = Path(results["hypergraph"])
        hg_size = hg_path.stat().st_size
        
        # Should be relatively small (< 1 MB for tiny model structure)
        assert hg_size < 1_000_000, f"Hypergraph too large: {hg_size} bytes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
