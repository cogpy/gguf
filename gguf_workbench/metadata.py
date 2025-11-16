"""
GGUF metadata handling.
"""

from typing import Any, Dict, List


class GGUFMetadata:
    """Represents GGUF file metadata."""

    def __init__(self):
        self.version: int = 3
        self.tensor_count: int = 0
        self.metadata_kv: Dict[str, Any] = {}
        self.tensors: List[Dict[str, Any]] = []

    def get(self, key: str, default: Any = None) -> Any:
        """Get a metadata value by key."""
        return self.metadata_kv.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a metadata value."""
        self.metadata_kv[key] = value

    def delete(self, key: str) -> bool:
        """Delete a metadata key. Returns True if the key existed."""
        if key in self.metadata_kv:
            del self.metadata_kv[key]
            return True
        return False

    def keys(self) -> List[str]:
        """Get all metadata keys."""
        return list(self.metadata_kv.keys())

    def items(self):
        """Get all metadata items."""
        return self.metadata_kv.items()

    def __repr__(self) -> str:
        return (
            f"GGUFMetadata(version={self.version}, "
            f"kv_count={len(self.metadata_kv)}, "
            f"tensor_count={self.tensor_count})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary."""
        return {
            "version": self.version,
            "tensor_count": self.tensor_count,
            "metadata": dict(self.metadata_kv),
            "tensors": self.tensors,
        }

    def print_summary(self) -> None:
        """Print a summary of the metadata."""
        print(f"GGUF Version: {self.version}")
        print(f"Tensor Count: {self.tensor_count}")
        print(f"\nMetadata ({len(self.metadata_kv)} entries):")
        for key, value in sorted(self.metadata_kv.items()):
            # Truncate long values for display
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:97] + "..."
            print(f"  {key}: {value_str}")

        if self.tensors:
            print(f"\nTensors ({len(self.tensors)} tensors):")
            for i, tensor in enumerate(self.tensors[:10]):  # Show first 10
                tensor_name = tensor.get("name", "unknown")
                tensor_shape = tensor.get("shape", [])
                tensor_type = tensor.get("type", "unknown")
                print(f"  [{i}] {tensor_name}: shape={tensor_shape}, type={tensor_type}")
            if len(self.tensors) > 10:
                print(f"  ... and {len(self.tensors) - 10} more tensors")
