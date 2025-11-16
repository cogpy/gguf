"""
GGUF file format constants and type definitions.
"""

import struct
from enum import IntEnum
from typing import Dict

# GGUF Magic number
GGUF_MAGIC = 0x46554747  # "GGUF" in ASCII (little-endian)
GGUF_VERSION = 3


# GGUF value types
class GGMLType(IntEnum):
    """GGML tensor types."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    I8 = 16
    I16 = 17
    I32 = 18


class GGUFValueType(IntEnum):
    """GGUF metadata value types."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# Type format strings for struct module
TYPE_FORMATS: Dict[int, str] = {
    GGUFValueType.UINT8: "B",
    GGUFValueType.INT8: "b",
    GGUFValueType.UINT16: "H",
    GGUFValueType.INT16: "h",
    GGUFValueType.UINT32: "I",
    GGUFValueType.INT32: "i",
    GGUFValueType.FLOAT32: "f",
    GGUFValueType.BOOL: "?",
    GGUFValueType.UINT64: "Q",
    GGUFValueType.INT64: "q",
    GGUFValueType.FLOAT64: "d",
}


def get_type_size(value_type: int) -> int:
    """Get the size in bytes of a GGUF value type."""
    if value_type in TYPE_FORMATS:
        return struct.calcsize(TYPE_FORMATS[value_type])
    elif value_type == GGUFValueType.STRING:
        return -1  # Variable size
    elif value_type == GGUFValueType.ARRAY:
        return -1  # Variable size
    else:
        raise ValueError(f"Unknown value type: {value_type}")
