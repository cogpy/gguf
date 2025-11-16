# GGUF Workbench

A comprehensive tool for inspecting, modifying, and customizing GGUF (GPT-Generated Unified Format) model files. GGUF is the file format used by llama.cpp and other tools for storing large language models.

## Features

- 🔍 **Inspect** GGUF files and view their metadata
- ✏️ **Modify** metadata values in GGUF files
- 📊 **Export** metadata to JSON for analysis
- 🔑 **List** all metadata keys in a file
- 🛠️ **CLI** Easy-to-use command-line interface
- 📚 **Python API** for programmatic access
- 🧠 **Pure Python Inference** - Multiple implementations showing how transformers work (NEW!)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/cogpy/gguf.git
cd gguf

# Install with pip
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Using pip (once published)

```bash
pip install gguf-workbench
```

## Quick Start

### Command Line Usage

#### Inspect a GGUF file

```bash
gguf-workbench inspect model.gguf
```

This will display:
- GGUF version
- Number of tensors
- All metadata key-value pairs
- Tensor information (names, shapes, types)

#### Get a specific metadata value

```bash
gguf-workbench get model.gguf general.name
```

#### Set a metadata value

```bash
# Set a string value
gguf-workbench set model.gguf general.name "My Custom Model"

# Set an integer value
gguf-workbench set model.gguf general.parameter_count 7000000000 --type int

# Set a float value
gguf-workbench set model.gguf general.temperature 0.8 --type float

# Set a boolean value
gguf-workbench set model.gguf general.chat_template_enabled true --type bool

# Save to a different file
gguf-workbench set model.gguf general.name "Custom Model" -o custom_model.gguf
```

#### Delete a metadata key

```bash
gguf-workbench delete model.gguf custom.key
```

#### List all metadata keys

```bash
# List keys only
gguf-workbench list model.gguf

# List keys with values
gguf-workbench list model.gguf --verbose
```

#### Export metadata to JSON

```bash
# Print to stdout
gguf-workbench export model.gguf

# Save to file
gguf-workbench export model.gguf -o metadata.json
```

### Python API Usage

```python
from gguf_workbench import GGUFReader, GGUFWriter, GGUFMetadata

# Read a GGUF file
with GGUFReader("model.gguf") as reader:
    metadata = reader.get_metadata()
    
    # Print summary
    reader.inspect()
    
    # Get specific values
    model_name = metadata.get("general.name")
    print(f"Model name: {model_name}")
    
    # List all keys
    for key in metadata.keys():
        print(f"{key}: {metadata.get(key)}")

# Modify metadata
with GGUFReader("model.gguf") as reader:
    metadata = reader.get_metadata()

# Update values
metadata.set("general.name", "My Custom Model")
metadata.set("general.version", "1.0")
metadata.delete("unwanted.key")

# Write to a new file
with GGUFWriter("custom_model.gguf", metadata) as writer:
    writer.write()

# Export to dictionary
metadata_dict = metadata.to_dict()
import json
with open("metadata.json", "w") as f:
    json.dump(metadata_dict, f, indent=2)
```

## GGUF Format Overview

GGUF (GPT-Generated Unified Format) is a binary format for storing large language models. It consists of:

1. **Header**: Magic number, version, tensor count, metadata count
2. **Metadata**: Key-value pairs containing model information (name, architecture, parameters, etc.)
3. **Tensor Info**: Information about each tensor (name, shape, type, offset)
4. **Tensor Data**: The actual tensor weights (not modified by this tool)

### Common Metadata Keys

- `general.name` - Model name
- `general.architecture` - Model architecture (e.g., "llama", "falcon")
- `general.file_type` - Quantization type
- `general.quantization_version` - Quantization version
- `tokenizer.ggml.model` - Tokenizer model type
- `tokenizer.ggml.tokens` - Tokenizer vocabulary
- And many more architecture-specific keys

## Pure Python Inference (NEW!)

This repository now includes **pure Python implementations** of transformer inference, showing exactly how weights are applied step-by-step. Perfect for learning and understanding transformers!

```python
from gguf_workbench.inference import TinyTransformerListBased

# Load model
model = TinyTransformerListBased.from_json('tinytf/tiny_model_gguf.json')

# Forward pass with detailed trace showing every operation
model.forward([0, 1, 2], trace=True)
```

**Four different implementations:**
1. **List-Based** - Most transparent, shows every indexed weight application
2. **Dict-Based** - Best for inspection, returns structured results
3. **Class-Based** - Production-ready with type safety
4. **Functional** - Pure functions for testing and verification

See [Inference README](gguf_workbench/inference/README.md) and [Language Comparison](LANGUAGE_COMPARISON.md) for details.

**Run the demo:**
```bash
python examples/demonstrate_inference.py
```

## Use Cases

- **Model Customization**: Change model names, descriptions, or other metadata
- **Model Analysis**: Inspect model architecture and configuration
- **Metadata Export**: Extract metadata for documentation or analysis
- **Model Preparation**: Prepare models for specific deployment scenarios
- **Debugging**: Investigate model format issues
- **Learning**: Understand how transformers work with transparent inference code (NEW!)

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black gguf_workbench/
```

### Linting

```bash
flake8 gguf_workbench/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- GGUF format specification from the llama.cpp project
- Inspired by the need for better GGUF file manipulation tools

## Related Projects

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Original GGUF implementation
- [gguf-parser](https://github.com/ggerganov/llama.cpp/tree/master/gguf-py) - Official GGUF parser