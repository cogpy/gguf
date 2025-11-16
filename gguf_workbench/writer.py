"""
GGUF file writer implementation.
"""

import struct
from pathlib import Path
from typing import Any, BinaryIO, List, Union

from .constants import GGUF_MAGIC, GGUFValueType, TYPE_FORMATS
from .metadata import GGUFMetadata


class GGUFWriter:
    """Writer for GGUF files."""

    def __init__(self, filepath: Union[str, Path], metadata: GGUFMetadata):
        """
        Initialize the GGUF writer.

        Args:
            filepath: Path to write the GGUF file
            metadata: Metadata to write
        """
        self.filepath = Path(filepath)
        self.metadata = metadata
        self._file: BinaryIO = None

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def open(self) -> None:
        """Open the file for writing."""
        self._file = open(self.filepath, "wb")

    def close(self) -> None:
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None

    def _write_string(self, s: str) -> None:
        """Write a GGUF string (length-prefixed UTF-8)."""
        encoded = s.encode("utf-8")
        self._file.write(struct.pack("<Q", len(encoded)))
        if len(encoded) > 0:
            self._file.write(encoded)

    def _infer_type(self, value: Any) -> int:
        """Infer the GGUF type from a Python value."""
        if isinstance(value, bool):
            return GGUFValueType.BOOL
        elif isinstance(value, int):
            if -128 <= value <= 127:
                return GGUFValueType.INT8
            elif 0 <= value <= 255:
                return GGUFValueType.UINT8
            elif -32768 <= value <= 32767:
                return GGUFValueType.INT16
            elif 0 <= value <= 65535:
                return GGUFValueType.UINT16
            elif -2147483648 <= value <= 2147483647:
                return GGUFValueType.INT32
            elif 0 <= value <= 4294967295:
                return GGUFValueType.UINT32
            else:
                return GGUFValueType.INT64
        elif isinstance(value, float):
            return GGUFValueType.FLOAT32
        elif isinstance(value, str):
            return GGUFValueType.STRING
        elif isinstance(value, list):
            return GGUFValueType.ARRAY
        else:
            raise ValueError(f"Cannot infer GGUF type for value: {value}")

    def _write_value(self, value: Any, value_type: int = None) -> None:
        """Write a value of the specified type."""
        if value_type is None:
            value_type = self._infer_type(value)

        if value_type == GGUFValueType.STRING:
            self._write_string(value)
        elif value_type == GGUFValueType.ARRAY:
            self._write_array(value)
        elif value_type in TYPE_FORMATS:
            fmt = TYPE_FORMATS[value_type]
            self._file.write(struct.pack(f"<{fmt}", value))
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def _write_array(self, array: List[Any]) -> None:
        """Write a GGUF array."""
        if not array:
            # Empty array - use UINT8 as default
            self._file.write(struct.pack("<I", GGUFValueType.UINT8))
            self._file.write(struct.pack("<Q", 0))
            return

        # Infer array element type from first element
        element_type = self._infer_type(array[0])
        self._file.write(struct.pack("<I", element_type))
        self._file.write(struct.pack("<Q", len(array)))

        for item in array:
            self._write_value(item, element_type)

    def write_header(self) -> None:
        """Write the GGUF header."""
        # Magic
        self._file.write(struct.pack("<I", GGUF_MAGIC))
        # Version
        self._file.write(struct.pack("<I", self.metadata.version))
        # Tensor count
        self._file.write(struct.pack("<Q", self.metadata.tensor_count))
        # KV count
        self._file.write(struct.pack("<Q", len(self.metadata.metadata_kv)))

    def write_metadata(self) -> None:
        """Write all metadata key-value pairs."""
        for key, value in self.metadata.metadata_kv.items():
            # Write key
            self._write_string(key)
            # Write value type
            value_type = self._infer_type(value)
            self._file.write(struct.pack("<I", value_type))
            # Write value
            self._write_value(value, value_type)

    def write_tensor_info(self) -> None:
        """Write tensor information."""
        for tensor in self.metadata.tensors:
            # Write tensor name
            self._write_string(tensor["name"])

            # Write number of dimensions
            shape = tensor["shape"]
            self._file.write(struct.pack("<I", len(shape)))

            # Write shape
            for dim in shape:
                self._file.write(struct.pack("<Q", dim))

            # Write tensor type
            self._file.write(struct.pack("<I", tensor["type"]))

            # Write offset
            self._file.write(struct.pack("<Q", tensor["offset"]))

    def write(self) -> None:
        """Write the complete GGUF file (metadata only, no tensor data)."""
        self.write_header()
        self.write_metadata()
        self.write_tensor_info()
