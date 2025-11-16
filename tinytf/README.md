# TinyTF - Tiny Transformer Test Models

This directory contains a minimal transformer model in various formats, used for testing and demonstrating the GGUF Workbench functionality. The tiny transformer is intentionally small to make it easy to inspect, validate, and test file format conversions.

## Overview

The tiny transformer is a minimal working transformer model with:
- **Architecture**: TinyTransformer
- **Embedding dimension**: 5
- **Context length**: 5
- **Number of blocks**: 1
- **Feed-forward layer size**: 5
- **Attention heads**: 1
- **Vocabulary size**: 10 tokens
- **Total parameters**: Approximately 200 parameters

## Files and Formats

### 1. GGUF Format (Binary)

#### `tiny_model.gguf`
The primary binary GGUF file containing the complete model including:
- Model metadata (architecture, dimensions, hyperparameters)
- Vocabulary (10 tokens: token_0 through token_9)
- Model weights for all layers (embedding, attention, feedforward, output)

This is the main file format used by llama.cpp and compatible inference engines.

**Structure:**
- Header: Magic number, version, tensor count, metadata count
- Metadata: Model configuration and hyperparameters
- Tensor Info: Names, shapes, types, and offsets for all weight tensors
- Tensor Data: Binary weight data

### 2. JSON Representations

#### `tiny_model_gguf.json`
Complete JSON representation of the model including all weights.

**Contents:**
```json
{
  "nodes": {
    "embedding": {
      "type": "embedding",
      "input_dim": 10,
      "output_dim": 5,
      "weights": [[...]]  // Full weight matrices
    },
    "attention": { ... },
    "output": { ... }
  },
  "edges": {
    "embedding_to_attention": { ... },
    "attention_to_output": { ... }
  }
}
```

This format is useful for:
- Human-readable inspection of model architecture
- Debugging weight values
- Understanding model connectivity
- JSON-based tooling and analysis

#### `tiny_model_gguf_parsed.json`
Simplified metadata-only representation extracted from the GGUF file.

**Contents:**
```json
{
  "Model_Architecture": "TinyTransformer",
  "Context_Length": 5,
  "Embedding_Length": 5,
  "Block_Count": 1,
  "Feed_Forward_Layer_Size": 5,
  "RoPE_Dimension_Count": 5,
  "Attention_Head_Count": 1,
  "Layer_Norm_Epsilon": 0.00001,
  "RoPE_Frequency_Base": 10000
}
```

Use this for:
- Quick inspection of model configuration
- Extracting hyperparameters
- Model documentation

#### `tiny_model_gguf_full.json`
Intermediate format showing metadata and vocabulary with placeholders for weight data.

**Contents:**
```json
{
  "metadata": { ... },
  "vocab": {
    "vocab": {
      "0": "token_0",
      "1": "token_1",
      ...
    }
  },
  "nodes": {
    "embedding": {
      "weights": "binary: tiny_model_weights.bin"
    },
    ...
  }
}
```

#### `tiny_model_gguf_full_parsed.json`
Complete parsed GGUF structure with metadata, vocabulary, and partial weight data.

**Contents:**
```json
{
  "metadata": { ... },
  "vocabulary": {},
  "weights": {
    "embedding": [[...], ...],
    "attention": [],
    "feedforward": [],
    "output": []
  }
}
```

### 3. TOML Configuration Files

#### `tiny_model_gguf_test.toml`
Test configuration showing model structure without weights.

**Contents:**
```toml
[nodes.embedding]
type = "embedding"
input_dim = 10
output_dim = 5
weights = []

[nodes.attention]
type = "self_attention"
input_dim = 5
num_heads = 1
weights = []

[nodes.output]
type = "linear"
input_dim = 5
output_dim = 10
weights = []

[edges.embedding_to_attention]
from = "embedding"
to = "attention"
```

Use this for:
- Testing model architecture parsing
- Validating graph structure
- Configuration templates

#### `tiny_model_gguf_with_weights.toml`
Complete TOML representation including full weight matrices.

**Features:**
- Human-readable weight values
- TOML array format for easy editing
- Complete layer definitions
- Edge connectivity specifications

Use this for:
- Manual weight inspection
- Small-scale weight editing
- TOML-based tooling

#### `tiny_model_gguf_with_weights_null.toml`
Similar to above but with null/empty weight placeholders (if weights are stored externally).

### 4. Binary Weight File

#### `tiny_model_weights.bin`
Raw binary file containing just the model weights.

**Size:** 800 bytes (approximately 200 float32 values)

**Structure:**
- All weights concatenated in sequential order
- 32-bit floating-point format (little-endian)
- No metadata or headers

Use this for:
- Separate weight storage
- Weight-only operations
- Binary diff comparisons
- Memory-mapped weight loading

### 5. PyTorch Format

#### `tiny_transformer.pth`
PyTorch serialized model state dict.

**Contents:**
```python
{
  'embedding.weight': Tensor(...),
  'attention.q_proj.weight': Tensor(...),
  'attention.k_proj.weight': Tensor(...),
  'attention.v_proj.weight': Tensor(...),
  'attention.o_proj.weight': Tensor(...),
  'feedforward.w1.weight': Tensor(...),
  'feedforward.w2.weight': Tensor(...),
  'output.weight': Tensor(...)
}
```

Use this for:
- PyTorch model loading
- Training or fine-tuning
- PyTorch-based inference
- Model conversion to other formats

**Loading:**
```python
import torch
state_dict = torch.load('tiny_transformer.pth')
```

### 6. ONNX Format

#### `tiny_transformer.onnx`
ONNX (Open Neural Network Exchange) format for interoperability.

**Features:**
- Platform-independent representation
- Optimized computation graph
- Compatible with ONNX Runtime
- Supports various deployment targets

Use this for:
- Cross-framework compatibility
- Production deployment
- ONNX Runtime inference
- Model optimization tools

**Loading:**
```python
import onnx
model = onnx.load('tiny_transformer.onnx')
```

### 7. PyTorch State Dict Directory

#### `tiny_transformer/`
Directory containing PyTorch's saved state dict in filesystem format.

**Structure:**
```
tiny_transformer/
├── version          # PyTorch version (3)
├── byteorder        # System byte order (little)
├── data.pkl         # Pickled metadata
└── data/            # Individual tensor files
    ├── 0            # Embedding weights
    ├── 1            # Query projection
    ├── 2            # Key projection
    ├── 3            # Value projection
    ├── 4            # Output projection
    ├── 5            # Feedforward W1
    ├── 6            # Feedforward W2
    ├── 7            # Layer norm weights
    ├── 8            # Layer norm bias
    ├── 9            # Additional parameters
    ├── 10-14        # Other model components
    └── ...
```

**Files:**
- `version`: PyTorch save format version (ASCII: "3")
- `byteorder`: Byte order for binary data (ASCII: "little")
- `data.pkl`: Pickled Python objects (tensor metadata, structure)
- `data/N`: Individual binary tensor data files

**Advantages:**
- Efficient for large models (individual tensor files)
- Easy to inspect individual tensors
- Supports memory mapping
- PyTorch's default format for large models

**Loading:**
```python
import torch
state_dict = torch.load('tiny_transformer', map_location='cpu')
```

## Model Architecture

The tiny transformer implements a minimal but complete transformer architecture:

```
Input (tokens: 0-9)
    ↓
Embedding Layer (10 → 5)
    ↓
Self-Attention (1 head, dim 5)
    ↓
Feed-Forward Network (5 → 5)
    ↓
Output Layer (5 → 10)
    ↓
Output (logits over 10 tokens)
```

### Layer Details

1. **Embedding Layer**
   - Input: Token IDs (0-9)
   - Output: 5-dimensional embeddings
   - Weights: 10 × 5 = 50 parameters

2. **Self-Attention Layer**
   - Type: Multi-head attention (1 head)
   - Dimensions: 5 (Q, K, V)
   - Weights: 5 × 15 = 75 parameters (Q, K, V projections + output)

3. **Feed-Forward Network**
   - Hidden size: 5
   - Activation: Usually GELU or ReLU
   - Weights: 5 × 5 × 2 = 50 parameters (two linear layers)

4. **Output Layer**
   - Input: 5-dimensional vectors
   - Output: 10-dimensional logits (vocabulary size)
   - Weights: 5 × 10 = 50 parameters

### Hyperparameters

- **Context Length**: 5 tokens (maximum sequence length)
- **Embedding Length**: 5 (dimensionality of embeddings)
- **Block Count**: 1 (number of transformer blocks)
- **Attention Heads**: 1 (single-head attention)
- **Layer Norm Epsilon**: 1e-5 (numerical stability)
- **RoPE Frequency Base**: 10000 (rotary position embeddings)
- **RoPE Dimensions**: 5 (all dimensions use RoPE)

## Use Cases

### 1. Testing GGUF Workbench
```bash
# Inspect the tiny model
gguf-workbench inspect tiny_model.gguf

# Get specific metadata
gguf-workbench get tiny_model.gguf Model_Architecture

# Modify metadata
gguf-workbench set tiny_model.gguf general.name "My Tiny Model"

# Export to JSON
gguf-workbench export tiny_model.gguf -o output.json
```

### 2. Format Conversion Examples

**GGUF → JSON:**
```python
from gguf_workbench import GGUFReader
import json

with GGUFReader('tiny_model.gguf') as reader:
    metadata = reader.get_metadata()
    with open('output.json', 'w') as f:
        json.dump(metadata.to_dict(), f, indent=2)
```

**PyTorch → GGUF:**
```python
import torch
# Load PyTorch model
state_dict = torch.load('tiny_transformer.pth')
# Convert to GGUF (requires conversion logic)
# ...
```

### 3. Model Validation

Verify that all formats represent the same model:
```python
import torch
import json
import numpy as np

# Load from different formats
pth_model = torch.load('tiny_transformer.pth')
with open('tiny_model_gguf.json') as f:
    json_model = json.load(f)

# Compare weights
# ...
```

### 4. Testing File Readers

The small size makes it perfect for:
- Unit testing GGUF readers/writers
- Validating format parsers
- Benchmarking conversion tools
- Debugging file format issues

## Format Comparison

| Format | Size | Human-Readable | Complete Weights | Use Case |
|--------|------|----------------|------------------|----------|
| `.gguf` | ~2KB | No | Yes | Production inference |
| `.json` (full) | ~4KB | Yes | Yes | Debugging, inspection |
| `.json` (parsed) | ~300B | Yes | No | Metadata only |
| `.toml` (with weights) | ~5KB | Yes | Yes | Manual editing |
| `.toml` (test) | ~400B | Yes | No | Structure testing |
| `.bin` | 800B | No | Yes | Weight storage only |
| `.pth` | ~6KB | No | Yes | PyTorch training/inference |
| `.onnx` | ~15KB | No | Yes | Cross-platform deployment |
| `tiny_transformer/` | ~2KB | No | Yes | PyTorch large model format |

## Vocabulary

The model uses a minimal 10-token vocabulary for testing:

```
Token ID  │  Token String
─────────────────────────
    0     │  token_0
    1     │  token_1
    2     │  token_2
    3     │  token_3
    4     │  token_4
    5     │  token_5
    6     │  token_6
    7     │  token_7
    8     │  token_8
    9     │  token_9
```

## Notes

- **Purpose**: These files are for testing and demonstration only, not for actual inference
- **Consistency**: All formats should represent the same model architecture and (where applicable) weights
- **Tiny Size**: The model is intentionally minimal to facilitate testing and inspection
- **File Types**: Multiple formats demonstrate interoperability between different ML frameworks
- **Documentation**: This folder serves as a reference for GGUF file structure and format conversion

## Related Documentation

- [GGUF Format Specification](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md)
- [GGUF Workbench README](../README.md)
- [PyTorch Save/Load](https://pytorch.org/docs/stable/notes/serialization.html)
- [ONNX Documentation](https://onnx.ai/onnx/)

## Maintenance

When updating these test files:
1. Ensure all formats remain synchronized (same architecture/weights)
2. Update this README if adding new formats
3. Validate files can be read by their respective loaders
4. Keep the model minimal and simple for testing purposes
