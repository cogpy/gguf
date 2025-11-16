"""
Tests for GGUF Workbench.
"""

import struct
import tempfile
from pathlib import Path

import pytest

from gguf_workbench import GGUFReader, GGUFWriter, GGUFMetadata
from gguf_workbench.constants import GGUF_MAGIC, GGUFValueType


def create_minimal_gguf(filepath: Path) -> None:
    """Create a minimal valid GGUF file for testing."""
    with open(filepath, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))  # Magic
        f.write(struct.pack("<I", 3))  # Version
        f.write(struct.pack("<Q", 0))  # Tensor count
        f.write(struct.pack("<Q", 3))  # KV count
        
        # Metadata KV pairs
        # Key 1: general.name (string)
        name_key = "general.name"
        f.write(struct.pack("<Q", len(name_key)))
        f.write(name_key.encode("utf-8"))
        f.write(struct.pack("<I", GGUFValueType.STRING))
        name_value = "Test Model"
        f.write(struct.pack("<Q", len(name_value)))
        f.write(name_value.encode("utf-8"))
        
        # Key 2: general.version (int)
        version_key = "general.version"
        f.write(struct.pack("<Q", len(version_key)))
        f.write(version_key.encode("utf-8"))
        f.write(struct.pack("<I", GGUFValueType.INT32))
        f.write(struct.pack("<i", 1))
        
        # Key 3: general.temperature (float)
        temp_key = "general.temperature"
        f.write(struct.pack("<Q", len(temp_key)))
        f.write(temp_key.encode("utf-8"))
        f.write(struct.pack("<I", GGUFValueType.FLOAT32))
        f.write(struct.pack("<f", 0.7))


def test_read_gguf():
    """Test reading a GGUF file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.gguf"
        create_minimal_gguf(filepath)
        
        with GGUFReader(filepath) as reader:
            metadata = reader.get_metadata()
            
            assert metadata.version == 3
            assert metadata.tensor_count == 0
            assert len(metadata.metadata_kv) == 3
            assert metadata.get("general.name") == "Test Model"
            assert metadata.get("general.version") == 1
            assert abs(metadata.get("general.temperature") - 0.7) < 0.001


def test_write_gguf():
    """Test writing a GGUF file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.gguf"
        
        # Create metadata
        metadata = GGUFMetadata()
        metadata.version = 3
        metadata.set("general.name", "Test Model")
        metadata.set("general.version", 1)
        metadata.set("test.bool", True)
        
        # Write file
        with GGUFWriter(filepath, metadata) as writer:
            writer.write()
        
        # Read it back
        with GGUFReader(filepath) as reader:
            read_metadata = reader.get_metadata()
            
            assert read_metadata.get("general.name") == "Test Model"
            assert read_metadata.get("general.version") == 1
            assert read_metadata.get("test.bool") == True


def test_modify_metadata():
    """Test modifying metadata."""
    metadata = GGUFMetadata()
    
    # Test set
    metadata.set("key1", "value1")
    assert metadata.get("key1") == "value1"
    
    # Test update
    metadata.set("key1", "value2")
    assert metadata.get("key1") == "value2"
    
    # Test delete
    assert metadata.delete("key1") == True
    assert metadata.get("key1") is None
    assert metadata.delete("key1") == False


def test_metadata_keys():
    """Test metadata keys listing."""
    metadata = GGUFMetadata()
    metadata.set("key1", "value1")
    metadata.set("key2", "value2")
    metadata.set("key3", "value3")
    
    keys = metadata.keys()
    assert len(keys) == 3
    assert "key1" in keys
    assert "key2" in keys
    assert "key3" in keys


def test_metadata_to_dict():
    """Test converting metadata to dictionary."""
    metadata = GGUFMetadata()
    metadata.version = 3
    metadata.tensor_count = 5
    metadata.set("key1", "value1")
    
    metadata_dict = metadata.to_dict()
    assert metadata_dict["version"] == 3
    assert metadata_dict["tensor_count"] == 5
    assert metadata_dict["metadata"]["key1"] == "value1"


def test_array_handling():
    """Test array value handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.gguf"
        
        # Create metadata with array
        metadata = GGUFMetadata()
        metadata.version = 3
        metadata.set("test.array", [1, 2, 3, 4, 5])
        
        # Write file
        with GGUFWriter(filepath, metadata) as writer:
            writer.write()
        
        # Read it back
        with GGUFReader(filepath) as reader:
            read_metadata = reader.get_metadata()
            array_value = read_metadata.get("test.array")
            assert array_value == [1, 2, 3, 4, 5]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
