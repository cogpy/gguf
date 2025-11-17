# GGUF Conversion Guide

## Overview

This guide explains how to convert any GGUF file to and from various representation formats. The GGUF Workbench provides a generalized conversion framework that works with models of any size, from tiny test models to frontier-scale LLMs.

## Quick Start

### Command Line

Convert a GGUF file to all representation formats:

```bash
gguf-workbench convert model.gguf output_dir/
```

Convert to a specific format:

```bash
# Hypergraph only
gguf-workbench convert model.gguf output_dir/ --format hypergraph

# DAG only
gguf-workbench convert model.gguf output_dir/ --format dag

# Symbolic representation
gguf-workbench convert model.gguf output_dir/ --format symbolic
```

Include weight values (warning: creates large files):

```bash
gguf-workbench convert model.gguf output_dir/ --weights
```

### Python API

```python
from gguf_workbench import GGUFConverter

# Create converter
converter = GGUFConverter("model.gguf")

# Check model info
print(converter.model_info)

# Convert to specific formats
hypergraph = converter.to_hypergraph(include_weights=False)
dag = converter.to_dag()
symbolic = converter.to_symbolic()
aiml = converter.to_aiml()
atomspace = converter.to_atomspace()
toml_hg = converter.to_toml_hypergraph()

# Export all formats at once
results = converter.export_all("output_dir/", include_weights=False)
```

## Supported Formats

| Format | Purpose | File Size (120M model) | Best For |
|--------|---------|------------------------|----------|
| **Hypergraph** | Multi-way relationship analysis | ~200 KB (structure) | Research, interpretability |
| **DAG** | Sequential computation flow | ~500 KB (structure) | Execution planning, teaching |
| **Symbolic** | Mathematical equations | ~50 KB | Academic papers, theory |
| **AIML** | Chatbot integration | ~5 KB | Interactive documentation |
| **OpenCog** | Neuro-symbolic AI | ~10 KB | Cognitive architectures, PLN |
| **TOML** | Human-editable config | ~15 KB (structure) | Collaborative research |

## Conversion Principles

### 1. Structure vs. Weights Separation

For models larger than ~1B parameters, it's recommended to separate structure from weights:

```python
# Structure only (small file)
converter.export_all("output/", include_weights=False)

# With weights (large file, only for small models)
converter.export_all("output/", include_weights=True)
```

**Why separate?**
- Structure scales sub-linearly with model size (~O(layers × operations))
- Weights scale linearly (~O(parameters))
- A 70B model has ~70 GB of weights but only ~40 MB of structure

### 2. Automatic Architecture Detection

The converter automatically detects model architecture from GGUF metadata:

```python
converter = GGUFConverter("model.gguf")

print(converter.architecture)  # e.g., "llama", "falcon", "gpt2"
print(converter.model_info)    # Extracted parameters
```

Supported architectures:
- LLaMA / LLaMA 2
- GPT-2 / GPT-Neo
- Falcon
- MPT
- Any transformer-based architecture

### 3. Metadata Preservation

All representations preserve model metadata:

```python
hypergraph = converter.to_hypergraph()
print(hypergraph.metadata)  # Contains vocab_size, embedding_dim, etc.
```

## Representation Details

### Hypergraph Representation

**Purpose**: Explicitly represent multi-way relationships (e.g., attention combining Q, K, V).

**Structure**:
- **Vertices**: Tensors and parameters
- **Hyperedges**: Operations connecting multiple inputs to multiple outputs

**Use Cases**:
- Analyzing attention patterns
- Identifying semantic modules
- Targeted fine-tuning
- Persona engineering

**Example**:
```python
hg = converter.to_hypergraph()

# Get statistics
stats = hg.get_statistics()
print(f"Vertices: {stats['vertex_count']}")
print(f"Hyperedges: {stats['hyperedge_count']}")

# Find attention operations
for edge_id, edge in hg.hyperedges.items():
    if "attention" in edge_id:
        print(f"{edge_id}: {edge.sources} → {edge.targets}")

# Export to JSON
hg.to_json("model_hypergraph.json")

# Export to Graphviz DOT for visualization
hg.export_graphviz("model_hypergraph.dot")
```

### DAG Representation

**Purpose**: Standard directed graph showing computation flow.

**Structure**:
- **Nodes**: Tensors, parameters, AND operation nodes
- **Edges**: Pairwise data flow connections

**Use Cases**:
- Understanding execution order
- Dependency tracking
- Standard visualization
- Integration with graph tools

**Example**:
```python
dag = converter.to_dag()

# Topological sort for execution order
order = dag.topological_sort()
print("Execution order:", order)

# Get predecessors/successors
preds = dag.get_predecessors("layer_5_attn_output")
succs = dag.get_successors("layer_5_attn_output")

# Export
dag.to_json("model_dag.json")
```

### Symbolic Representation

**Purpose**: Mathematical/algebraic representation.

**Structure**:
- **Parameters**: Matrix symbols (W^Q, W^K, etc.)
- **Expressions**: Equations showing computations

**Use Cases**:
- Academic papers
- Theoretical analysis
- Teaching concepts
- Documentation

**Example**:
```python
symbolic = converter.to_symbolic()

# View parameters
for param in symbolic.parameters:
    print(f"{param['symbol']}: {param['description']}")

# View expressions
for expr in symbolic.expressions:
    print(f"{expr['expression']}: {expr['description']}")

# Export to different formats
symbolic.to_json("model_symbolic.json")
symbolic.export_markdown("model_symbolic.md")
symbolic.to_latex("model_symbolic.tex")
```

### AIML Representation

**Purpose**: Chatbot integration (Pandorabots).

**Structure**:
- **Categories**: Pattern-response pairs about the model

**Use Cases**:
- Interactive documentation
- Educational chatbots
- Q&A interfaces

**Example**:
```python
aiml = converter.to_aiml()

# View categories
for cat in aiml.categories[:5]:
    print(f"Q: {cat['pattern']}")
    print(f"A: {cat['template']}\n")

# Export
aiml.save_xml("model.aiml")
aiml.save_json("model_aiml.json")
```

### OpenCog AtomSpace Representation

**Purpose**: Neuro-symbolic AI integration.

**Structure**:
- **ConceptNodes**: Model components
- **PredicateNodes**: Properties and relationships
- **Links**: Typed relationships (Inheritance, Evaluation, Execution)

**Use Cases**:
- Symbolic reasoning about neural models
- PLN (Probabilistic Logic Networks)
- Cognitive architectures
- Explainable AI

**Example**:
```python
atomspace = converter.to_atomspace()

# View atoms
json_data = atomspace.to_json()
print(f"Total atoms: {json_data['atom_count']}")
print(f"KB rules: {json_data['knowledge_base_rules']}")

# Export
atomspace.save_scheme("model_atomspace.scm")

# Use in OpenCog:
# (load "model_atomspace.scm")
# (pln-fc (Concept "ModelName"))
```

### TOML Hypergraph Representation

**Purpose**: Human-editable hypergraph configuration.

**Structure**:
- **[metadata]**: Model configuration
- **[vertices.*]**: Tensor and parameter definitions
- **[hyperedges.*]**: Operation specifications
- **[weights.*]**: Optional weight values

**Use Cases**:
- Version control friendly
- Collaborative research
- Manual annotation
- Configuration-based models

**Example**:
```python
toml_hg = converter.to_toml_hypergraph()

# View statistics
stats = toml_hg.to_json()['statistics']
print(f"Vertices: {stats['vertex_count']}")
print(f"Hyperedges: {stats['hyperedge_count']}")

# Export
toml_hg.save_toml("model_hypergraph.toml")
toml_hg.save_json("model_hypergraph.json")
```

## Scaling Considerations

### Small Models (< 1B parameters)

Can include weights in all representations:

```python
converter.export_all("output/", include_weights=True)
```

**File sizes for 120M model**:
- Hypergraph JSON: ~1.2 GB (with weights)
- DAG JSON: ~1.5 GB (with weights)
- TOML: ~1.2 GB (with weights)

### Medium Models (1B - 10B parameters)

Use structure-only representations:

```python
converter.export_all("output/", include_weights=False)
```

**File sizes for 7B model**:
- Hypergraph JSON: ~1.5 MB (structure only)
- DAG JSON: ~4 MB (structure only)
- All text formats combined: < 10 MB

### Large Models (10B - 100B parameters)

Structure-only mandatory:

```python
# Only export lightweight formats
converter.export_all(
    "output/",
    include_weights=False,
    formats=["hypergraph", "symbolic", "aiml"]
)
```

### Frontier Models (> 100B parameters)

Use minimal representations:

```python
# Only the most compact formats
results = converter.export_all(
    "output/",
    include_weights=False,
    formats=["symbolic", "aiml"]
)
```

## Persona and Emotive Mapping

### Using Hypergraph for Targeted Training

**Step 1: Discover semantic functions**

```python
hg = converter.to_hypergraph()

# Run interpretability tools (e.g., activation probing)
# Annotate hyperedges with discovered functions
for edge_id in ["layer_12_attn", "layer_15_ffn"]:
    edge = hg.hyperedges[edge_id]
    edge.properties["semantic_role"] = "emotion_processing"
    edge.properties["primary_emotions"] = ["sadness", "empathy"]
```

**Step 2: Export annotated hypergraph**

```python
hg.to_json("model_annotated.json")
```

**Step 3: Use for targeted fine-tuning**

```python
# Load annotated hypergraph
import json
with open("model_annotated.json") as f:
    data = json.load(f)

# Find components to target
target_components = [
    edge_id for edge_id, edge in data["hyperedges"].items()
    if "emotion_processing" in edge.get("properties", {}).get("semantic_role", "")
]

# Apply to PyTorch model (pseudo-code)
# for component in target_components:
#     unfreeze_parameters(component)
# train_with_frozen_layers(model, target_components)
```

### Using OpenCog for Persona Engineering

**Step 1: Define persona rules**

```python
atomspace = converter.to_atomspace()

# Add semantic annotations (manually or programmatically)
# This would typically be done in Scheme or via Python API
```

**Step 2: Export with rules**

```python
atomspace.save_scheme("model_with_personas.scm")
```

**Step 3: Use PLN for inference**

```scheme
; Load in OpenCog
(load "model_with_personas.scm")

; Define persona
(DefineLink
  (DefinedSchemaNode "Persona_Empathetic")
  (SequentialAnd
    ; Amplify emotion processing
    (Evaluation (stv 1.5 0.9)
      (Predicate "activate")
      (Concept "EmotionLayers"))))

; Apply persona at inference time
(cog-execute! (DefinedSchema "Persona_Empathetic"))
```

## Conversion Best Practices

### 1. Always separate structure from weights for large models

```python
# Good for models > 1B
converter.export_all("output/", include_weights=False)

# Only for tiny models
converter.export_all("output/", include_weights=True)
```

### 2. Use appropriate formats for your task

```python
# Research & interpretability
formats = ["hypergraph", "atomspace"]

# Documentation
formats = ["symbolic", "aiml"]

# All purposes (small models only)
formats = None  # defaults to all
```

### 3. Check file sizes before exporting with weights

```python
import os

# Check GGUF size first
gguf_size = os.path.getsize("model.gguf")
print(f"GGUF size: {gguf_size / 1e9:.2f} GB")

# Estimate representation sizes
# JSON with weights ≈ 5x GGUF size
# TOML with weights ≈ 5x GGUF size

if gguf_size > 1e9:  # > 1 GB
    print("Model too large for weight inclusion")
    include_weights = False
else:
    include_weights = True

converter.export_all("output/", include_weights=include_weights)
```

### 4. Use version control for lightweight representations

```python
# These are small enough for Git
formats = ["symbolic", "aiml"]
converter.export_all("models/configs/", formats=formats)

# Commit to version control
# git add models/configs/
# git commit -m "Add model architecture representations"
```

## Advanced Usage

### Custom Annotations

Add custom semantic annotations to hypergraph:

```python
hg = converter.to_hypergraph()

# Annotate specific components
hg.hyperedges["layer_10_attn"].properties.update({
    "semantic_function": "coreference_resolution",
    "confidence": 0.85,
    "discovered_by": "activation_probing_v1"
})

# Save annotated version
hg.to_json("model_annotated.json")
```

### Partial Conversions

Convert only specific layers:

```python
# This would require extending the converter
# Future feature: layer-wise conversion for very large models
```

### Integration with Interpretability Tools

```python
from gguf_workbench import GGUFConverter
# from your_interpretability_tool import probe_attention_heads

converter = GGUFConverter("model.gguf")
hg = converter.to_hypergraph()

# Run interpretability analysis
# results = probe_attention_heads(model, hg)

# Annotate hypergraph with results
# for edge_id, semantic_info in results.items():
#     hg.hyperedges[edge_id].properties.update(semantic_info)

# Save enriched representation
hg.to_json("model_interpreted.json")
```

## Troubleshooting

### "Architecture not recognized"

The converter may not recognize all architectures. Check metadata:

```python
converter = GGUFConverter("model.gguf")
print(converter.metadata)  # See all available metadata

# Manually specify architecture if needed
# (Future feature)
```

### "File too large"

Don't include weights for large models:

```python
converter.export_all("output/", include_weights=False)
```

### "Missing metadata keys"

Some GGUF files may have non-standard metadata. The converter uses fallback values:

```python
print(converter.model_info)  # Check what was extracted
```

## Related Documentation

- [SCALING_ANALYSIS.md](SCALING_ANALYSIS.md) - Detailed analysis of file sizes and inference speeds
- [tinytf/REPRESENTATION_ANALYSIS.md](tinytf/REPRESENTATION_ANALYSIS.md) - Comparison of representation methods
- [tinytf/representations/FORMATS_GUIDE.md](tinytf/representations/FORMATS_GUIDE.md) - Complete format reference

## Examples

See `examples/convert_gguf.py` for a complete working example.
