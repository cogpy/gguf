# GGUF Workbench Quick Start Guide

This guide will help you get started with the GGUF Workbench in just a few minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/cogpy/gguf.git
cd gguf

# Install the package
pip install -e .
```

## Basic Usage

### 1. Inspect a GGUF File

The quickest way to see what's inside a GGUF file:

```bash
gguf-workbench inspect your-model.gguf
```

This shows:
- GGUF version
- Number of tensors
- All metadata keys and values
- Tensor information

### 2. Get a Specific Value

```bash
gguf-workbench get your-model.gguf general.name
```

### 3. List All Keys

```bash
# List just the keys
gguf-workbench list your-model.gguf

# List keys with values
gguf-workbench list your-model.gguf --verbose
```

### 4. Modify Metadata

```bash
# Change the model name
gguf-workbench set your-model.gguf general.name "My Custom Model"

# Save to a different file
gguf-workbench set your-model.gguf general.name "Custom" -o custom.gguf
```

### 5. Export to JSON

```bash
# Print to screen
gguf-workbench export your-model.gguf

# Save to file
gguf-workbench export your-model.gguf -o metadata.json
```

## Python API

For programmatic access:

```python
from gguf_workbench import GGUFReader, GGUFWriter

# Read a file
with GGUFReader("model.gguf") as reader:
    metadata = reader.get_metadata()
    
    # Get values
    name = metadata.get("general.name")
    print(f"Model: {name}")
    
    # List all keys
    for key in metadata.keys():
        print(f"{key}: {metadata.get(key)}")

# Modify and write
metadata.set("general.name", "New Name")
metadata.set("custom.field", "Custom Value")

with GGUFWriter("modified.gguf", metadata) as writer:
    writer.write()
```

## Common Use Cases

### Rename a Model

```bash
gguf-workbench set model.gguf general.name "My Model" -o renamed.gguf
```

### Add Custom Metadata

```bash
gguf-workbench set model.gguf custom.author "Your Name"
gguf-workbench set model.gguf custom.date "2025-11-16"
```

### Clean Up Metadata

```bash
gguf-workbench delete model.gguf unwanted.key
```

### Inspect Model Details

```bash
# Get model architecture
gguf-workbench get model.gguf general.architecture

# Get parameter count
gguf-workbench get model.gguf general.parameter_count

# Get context length
gguf-workbench get model.gguf llama.context_length
```

## Examples

Check the `examples/` directory for complete examples:

- `inspect_file.py` - Basic file inspection
- `modify_metadata.py` - Modifying metadata
- `export_to_json.py` - Exporting to JSON
- `complete_demo.py` - Full feature demonstration

Run them with:

```bash
python examples/complete_demo.py
```

## Tips

1. **Always backup**: Make a copy of your GGUF file before modifying
2. **Use `-o` flag**: Save to a new file instead of modifying in-place
3. **Check types**: Use `--type` flag when setting values (int, float, bool, json)
4. **Verify changes**: Use `inspect` or `get` to verify your modifications

## Getting Help

```bash
# General help
gguf-workbench --help

# Command-specific help
gguf-workbench inspect --help
gguf-workbench set --help
```

## Next Steps

- Read the full [README.md](../README.md) for detailed documentation
- Check out the [examples](../examples/) directory
- Explore the Python API in [gguf_workbench/](../gguf_workbench/)
- Run the tests: `pytest tests/`
