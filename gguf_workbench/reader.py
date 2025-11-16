"""
GGUF file reader implementation.
"""

import struct
from pathlib import Path
from typing import Any, BinaryIO, List, Union

from .constants import GGUF_MAGIC, GGUFValueType, TYPE_FORMATS
from .metadata import GGUFMetadata


class GGUFReader:
    """Reader for GGUF files."""

    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize the GGUF reader.

        Args:
            filepath: Path to the GGUF file
        """
        self.filepath = Path(filepath)
        self.metadata = GGUFMetadata()
        self._file: BinaryIO = None

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def open(self) -> None:
        """Open the GGUF file and read metadata."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        self._file = open(self.filepath, "rb")
        self._read_header()
        self._read_metadata()
        self._read_tensor_info()

    def close(self) -> None:
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None

    def _read_header(self) -> None:
        """Read and validate the GGUF header."""
        magic = struct.unpack("<I", self._file.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic number: 0x{magic:08x}")

        version = struct.unpack("<I", self._file.read(4))[0]
        self.metadata.version = version

        tensor_count = struct.unpack("<Q", self._file.read(8))[0]
        self.metadata.tensor_count = tensor_count

        kv_count = struct.unpack("<Q", self._file.read(8))[0]
        self._kv_count = kv_count

    def _read_string(self) -> str:
        """Read a GGUF string (length-prefixed UTF-8)."""
        length = struct.unpack("<Q", self._file.read(8))[0]
        if length == 0:
            return ""
        string_bytes = self._file.read(length)
        return string_bytes.decode("utf-8")

    def _read_value(self, value_type: int) -> Any:
        """Read a value of the specified type."""
        if value_type == GGUFValueType.STRING:
            return self._read_string()
        elif value_type == GGUFValueType.ARRAY:
            return self._read_array()
        elif value_type in TYPE_FORMATS:
            fmt = TYPE_FORMATS[value_type]
            size = struct.calcsize(fmt)
            data = self._file.read(size)
            return struct.unpack(f"<{fmt}", data)[0]
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def _read_array(self) -> List[Any]:
        """Read a GGUF array."""
        array_type = struct.unpack("<I", self._file.read(4))[0]
        array_len = struct.unpack("<Q", self._file.read(8))[0]

        result = []
        for _ in range(array_len):
            result.append(self._read_value(array_type))
        return result

    def _read_metadata(self) -> None:
        """Read all metadata key-value pairs."""
        for _ in range(self._kv_count):
            key = self._read_string()
            value_type = struct.unpack("<I", self._file.read(4))[0]
            value = self._read_value(value_type)
            self.metadata.metadata_kv[key] = value

    def _read_tensor_info(self) -> None:
        """Read tensor information."""
        for _ in range(self.metadata.tensor_count):
            tensor_name = self._read_string()

            # Read number of dimensions
            n_dims = struct.unpack("<I", self._file.read(4))[0]

            # Read shape
            shape = []
            for _ in range(n_dims):
                dim = struct.unpack("<Q", self._file.read(8))[0]
                shape.append(dim)

            # Read tensor type
            tensor_type = struct.unpack("<I", self._file.read(4))[0]

            # Read offset
            offset = struct.unpack("<Q", self._file.read(8))[0]

            tensor_info = {
                "name": tensor_name,
                "shape": shape,
                "type": tensor_type,
                "offset": offset,
            }
            self.metadata.tensors.append(tensor_info)

    def get_metadata(self) -> GGUFMetadata:
        """Get the metadata object."""
        return self.metadata

    def inspect(self) -> None:
        """Print a detailed inspection of the GGUF file."""
        self.metadata.print_summary()
