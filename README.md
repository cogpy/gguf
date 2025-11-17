# GGUF Workbench

A comprehensive tool for inspecting, modifying, and customizing GGUF (GPT-Generated Unified Format) model files. GGUF is the file format used by llama.cpp and other tools for storing large language models.

## Features

- 🔍 **Inspect** GGUF files and view their metadata
- ✏️ **Modify** metadata values in GGUF files
- 📊 **Export** metadata to JSON for analysis
- 🔑 **List** all metadata keys in a file
- 🔄 **Convert** GGUF to multiple representation formats
- 💬 **Conversation Analysis** - Analyze what can be learned from query-response datasets
- 📖 **Vocabulary Analysis** - Analyze vocabulary usage and layer activations (NEW!)
- 🛠️ **CLI** Easy-to-use command-line interface
- 📚 **Python API** for programmatic access
- 🧠 **Pure Python Inference** - Multiple implementations showing how transformers work
- 🎭 **Persona Engineering** - Tools for targeted training and semantic analysis

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

#### Analyze conversation datasets (NEW!)

```bash
# Analyze what can be learned from conversation data
gguf-workbench analyze-conversations conversations.json

# With model information for better analysis
gguf-workbench analyze-conversations conversations.json \
  --vocab-size 50000 \
  --embedding-dim 768

# Save detailed analysis to JSON
gguf-workbench analyze-conversations conversations.json \
  -o analysis_results.json
```

This command helps answer fundamental questions:
- What can we learn about model weights from query-response pairs?
- How does sequential ordering enhance our understanding?
- What additional insights come from character prompts and parameters?
- What are the theoretical and practical limits of inference?

#### Analyze vocabulary usage and layer activations (NEW!)

```bash
# Analyze vocabulary coverage and layer activations
gguf-workbench analyze-vocabulary conversations.json \
  --vocab-size 50257 \
  --layers 12 \
  --embedding-dim 768 \
  --architecture gpt2

# Save detailed analysis to JSON
gguf-workbench analyze-vocabulary conversations.json \
  --vocab-size 50257 \
  --layers 12 \
  -o vocab_analysis.json
```

This command provides deep insights into:
- **Vocabulary enumeration**: Complete word counts from conversations with model metadata
- **Coverage analysis**: How expressed vocabulary relates to total available vocabulary
- **Layer activation inference**: Which transformer layers are likely active based on patterns
- **Echo state reservoir dynamics**: How reservoir computing interacts with transformers
- **Dynamic persona variables**: What persona dimensions are reflected in vocabulary

See [VOCABULARY_ANALYSIS_GUIDE.md](VOCABULARY_ANALYSIS_GUIDE.md) for complete documentation.

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

#### Analyze Conversation Datasets (NEW!)

```python
from gguf_workbench import (
    ConversationPair,
    ConversationDataset,
    ConversationAnalyzer,
)

# Create a conversation dataset
pairs = [
    ConversationPair(
        query="What is machine learning?",
        response="Machine learning is a subset of AI...",
        position=0
    ),
    ConversationPair(
        query="Can you explain more?",
        response="It enables systems to learn from data...",
        position=1
    ),
]

dataset = ConversationDataset(
    pairs=pairs,
    is_sequential=True,  # Preserves conversation order
    global_metadata={'model': 'gpt-4', 'temperature': 0.7}
)

# Analyze the dataset
analyzer = ConversationAnalyzer(
    model_vocab_size=50000,  # Model vocabulary size (if known)
    embedding_dim=768,       # Embedding dimension (if known)
)

result = analyzer.analyze(dataset)

# Get human-readable summary
print(result.summary())

# Save detailed analysis
result.to_json_file("analysis_results.json")

# Access specific insights
print(f"Unique tokens: {result.token_analysis['unique_tokens']}")
print(f"Vocabulary coverage: {result.token_analysis['vocab_coverage']:.2%}")

# Understand what can be learned
for aspect in result.unordered_insights['learnable_aspects']:
    print(f"✓ {aspect['aspect']}: {aspect['description']}")

# Understand fundamental limits
for limit in result.inference_limits['theoretical_limits']:
    print(f"✗ {limit['limit']}: {limit['explanation']}")
```

**Key Questions Answered**:
- What can we learn from unordered query-response pairs?
- How does sequential conversation order enhance analysis?
- What insights come from character prompts and parameters?
- What are the theoretical limits of inferring weights and activations?
- How under-determined is the system given the data?

**Use Cases**:
- Understanding data requirements for model analysis
- Evaluating conversation datasets for research
- Planning interpretability studies
- Educational tool for understanding model internals

#### Analyze Vocabulary Usage and Layer Activations (NEW!)

```python
from gguf_workbench import VocabularyAnalyzer, load_conversations_from_json

# Load conversations (supports JSON, JSONL, messages format)
conversations = load_conversations_from_json('conversations.json')

# Create analyzer with model configuration
analyzer = VocabularyAnalyzer(
    total_vocab_size=50257,  # Total model vocabulary size
    model_layers=12,          # Number of transformer layers
    embedding_dim=768,        # Embedding dimension
    architecture="gpt2"       # Model architecture
)

# Generate comprehensive report
report = analyzer.generate_comprehensive_report(conversations)

# Access vocabulary analysis
vocab = report['vocabulary_analysis']
print(f"Unique words: {vocab['total_unique_words']}")
print(f"Coverage: {vocab['vocabulary_coverage']['coverage_percentage']:.2f}%")

# Access layer activation insights
layers = report['layer_activation_analysis']
for estimate in layers['layer_activation_estimates']:
    print(f"Layers {estimate['layer_range']}: {estimate['activation_level']}")

# Access echo reservoir analysis
echo = report['echo_reservoir_analysis']
feedback = echo['feedback_loops']
print(f"Feedback strength: {feedback['feedback_strength']}")
print(f"Echo ratio: {feedback['echo_ratio']:.1%}")

# Get recommendations
for rec in echo['recommendations']:
    print(f"• {rec}")

# Save detailed report
import json
with open('vocab_report.json', 'w') as f:
    json.dump(report, f, indent=2)
```

**Key Insights Provided**:
- Complete vocabulary enumeration with word counts
- Vocabulary coverage metrics (expressed vs. total available)
- Layer activation inference based on vocabulary patterns
- Echo state reservoir computing dynamics
- Dynamic persona variable detection
- Temporal vocabulary evolution patterns

**Use Cases**:
- Understanding model vocabulary utilization
- Targeting interpretability at active layers
- Analyzing reservoir-transformer interactions
- Detecting persona dimensions in conversations
- Evaluating dataset vocabulary diversity

See [VOCABULARY_ANALYSIS_GUIDE.md](VOCABULARY_ANALYSIS_GUIDE.md) for complete documentation and examples.

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
- **Representation Conversion**: Convert to Hypergraph, DAG, Symbolic, and other formats
- **Conversation Data Analysis**: Understand what can be learned from query-response datasets (NEW!)
- **Interpretability Research**: Analyze semantic functions of model components and data requirements
- **Persona Engineering**: Map and target emotion/style processing components
- **Metadata Export**: Extract metadata for documentation or analysis
- **Model Preparation**: Prepare models for specific deployment scenarios
- **Cognitive AI Integration**: Export to OpenCog for neuro-symbolic systems
- **Educational Tool**: Learn about model internals, inference limits, and data analysis
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