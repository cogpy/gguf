"""
GGUF Workbench - A tool for inspecting and modifying GGUF model files.
"""

__version__ = "0.1.0"

from .reader import GGUFReader
from .writer import GGUFWriter
from .metadata import GGUFMetadata

__all__ = ["GGUFReader", "GGUFWriter", "GGUFMetadata", "__version__"]

# Inference module is available but not imported by default
# Use: from gguf_workbench import inference
# Or:  from gguf_workbench.inference import TinyTransformerListBased
