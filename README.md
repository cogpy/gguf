# GGUF Workbench

A comprehensive tool for inspecting, modifying, and customizing GGUF (GPT-Generated Unified Format) model files. GGUF is the file format used by llama.cpp and other tools for storing large language models.

## Features

- 🔍 **Inspect** GGUF files and view their metadata
- ✏️ **Modify** metadata values in GGUF files
- 📊 **Export** metadata to JSON for analysis
- 🔑 **List** all metadata keys in a file
- 🔄 **Convert** GGUF to multiple representation formats (NEW!)
- 🛠️ **CLI** Easy-to-use command-line interface
- 📚 **Python API** for programmatic access
- 🧠 **Pure Python Inference** - Multiple implementations showing how transformers work
- 🎭 **Persona Engineering** - Tools for targeted training and semantic analysis (NEW!)

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

#### Convert to representation formats (NEW!)

```bash
# Convert to all formats (Hypergraph, DAG, Symbolic, AIML, OpenCog, TOML)
gguf-workbench convert model.gguf output_dir/

# Convert to specific format
gguf-workbench convert model.gguf output_dir/ --format hypergraph

# Include weights (only for small models)
gguf-workbench convert model.gguf output_dir/ --weights
```

See [CONVERSION_GUIDE.md](CONVERSION_GUIDE.md) for detailed usage and [SCALING_ANALYSIS.md](SCALING_ANALYSIS.md) for scalability analysis.

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

#### Convert GGUF to Multiple Formats (NEW!)

```python
from gguf_workbench import GGUFConverter

# Create converter for any GGUF file
converter = GGUFConverter("model.gguf")

# Check extracted model info
print(converter.architecture)  # e.g., "llama", "gpt2", "falcon"
print(converter.model_info)    # Dict with vocab_size, num_layers, etc.

# Convert to specific formats
hypergraph = converter.to_hypergraph()  # Multi-way relationship graph
dag = converter.to_dag()                # Directed acyclic graph
symbolic = converter.to_symbolic()      # Mathematical equations
aiml = converter.to_aiml()              # Chatbot format
atomspace = converter.to_atomspace()    # OpenCog cognitive AI
toml_hg = converter.to_toml_hypergraph() # Human-editable config

# Export all formats at once
results = converter.export_all(
    output_dir="representations/",
    include_weights=False  # Structure only (recommended for large models)
)
```

**Key Features**:
- Works with ANY GGUF file (auto-detects architecture)
- Structure/weight separation for efficient large model handling
- Enables persona engineering and targeted training
- See [CONVERSION_GUIDE.md](CONVERSION_GUIDE.md) for complete documentation

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

**Multiple representation formats:**
- **Hypergraph** - Multi-way edges for complex tensor operations
- **DAG** - Directed acyclic graph of computations
- **Symbolic** - Mathematical/algebraic equations
- **AIML** - Pandorabot chatbot integration format
- **OpenCog AtomSpace** - Typed hypergraph for symbolic AI (URE, PLN, ECAN, MOSES)
- **TOML Hypergraph** - Configuration format with explicit hypergraph tuples

See [Inference README](gguf_workbench/inference/README.md) and [Language Comparison](LANGUAGE_COMPARISON.md) for details.

**Run the demo:**
```bash
python examples/demonstrate_inference.py
```

## Generalized GGUF Conversion (NEW!)

Convert ANY GGUF file to multiple representation formats for analysis, documentation, and persona engineering.

### Key Features

- **Automatic Architecture Detection**: Works with LLaMA, GPT-2, Falcon, and any transformer
- **Structure/Weight Separation**: Analyze 700B models with < 200 MB overhead
- **Multiple Output Formats**: Hypergraph, DAG, Symbolic, AIML, OpenCog, TOML
- **Persona Engineering Support**: Identify and target specific semantic components

### Example: Converting a 7B Model

```python
from gguf_workbench import GGUFConverter

# Automatic architecture detection
converter = GGUFConverter("llama-7b.gguf")

# Generate all representations (structure only: ~10 MB total)
results = converter.export_all("analysis/", include_weights=False)

# Result:
# - Hypergraph: 1.5 MB (multi-way operation graph)
# - DAG: 4 MB (execution flow)  
# - Symbolic: 100 KB (mathematical equations)
# - OpenCog: 200 KB (cognitive AI integration)
# - AIML: 10 KB (chatbot interface)
# - TOML: 100 KB (human-editable config)
```

### Documentation

- **[CONVERSION_GUIDE.md](CONVERSION_GUIDE.md)** - Complete conversion guide with examples
- **[SCALING_ANALYSIS.md](SCALING_ANALYSIS.md)** - File sizes and performance for 120M to 700B models
- **[GENERALIZED_CONVERSION_SUMMARY.md](GENERALIZED_CONVERSION_SUMMARY.md)** - Implementation details

### Use Cases

- **Research**: Analyze transformer architectures at any scale
- **Interpretability**: Map semantic functions to specific components
- **Persona Engineering**: Target emotion/style processing for fine-tuning
- **Documentation**: Generate academic papers, chatbots, or interactive docs
- **Cognitive AI**: Integration with OpenCog for neuro-symbolic systems

**Run the conversion demo:**
```bash
python examples/convert_gguf.py
```

## Use Cases

- **Model Customization**: Change model names, descriptions, or other metadata
- **Model Analysis**: Inspect model architecture and configuration
- **Representation Conversion**: Convert to Hypergraph, DAG, Symbolic, and other formats (NEW!)
- **Persona Engineering**: Map and target emotion/style processing components (NEW!)
- **Metadata Export**: Extract metadata for documentation or analysis
- **Model Preparation**: Prepare models for specific deployment scenarios
- **Interpretability Research**: Analyze semantic functions of model components (NEW!)
- **Cognitive AI Integration**: Export to OpenCog for neuro-symbolic systems (NEW!)
- **Debugging**: Investigate model format issues
- **Learning**: Understand how transformers work with transparent inference code

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