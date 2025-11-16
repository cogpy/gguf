# TinyTransformer Model Representations

This directory contains multiple representations of the TinyTransformer model, demonstrating different approaches to model structure representation.

## Files

### Hypergraph Representation
- **tiny_transformer_hypergraph.json** - Complete hypergraph representation
- **tiny_transformer_hypergraph.dot** - Graphviz visualization

**Statistics:**
- 26 vertices (16 tensors, 10 parameters)
- 14 hyperedges (operations)
- Average hyperedge size: 3.0 vertices
- Maximum hyperedge size: 4 vertices

**Operations:** embedding_lookup, linear (6x), matmul_scaled, softmax, matmul, add (2x), layer_norm, linear_gelu

### DAG (Directed Graph) Representation
- **tiny_transformer_dag.json** - Complete DAG representation
- **tiny_transformer_dag.dot** - Graphviz visualization

**Statistics:**
- 39 nodes (tensors, parameters, and operation nodes)
- 42 edges (data flow)
- Average in-degree: 1.08
- Average out-degree: 1.08

### Symbolic/Algebraic Representation
- **tiny_transformer_symbolic.json** - JSON format
- **tiny_transformer_symbolic.md** - Human-readable Markdown
- **tiny_transformer_symbolic.tex** - LaTeX for academic papers

**Statistics:**
- 10 parameters (E, W^Q, W^K, W^V, W^O, γ₁, β₁, W^ff₁, W^ff₂, W^out)
- 12 forward pass expressions
- 260 total parameter values (counting all matrix elements)

### Comparison Analysis
- **representation_comparison.json** - Quantitative comparison
- **representation_comparison.md** - Detailed analysis report

**Rankings (Overall Score):**
1. Symbolic/Algebraic (0.88)
2. DAG (0.87)
3. Hypergraph (0.81)
4. ONNX (0.81)
5. JSON (0.78)
6. TOML (0.77)
7. PyTorch (0.77)
8. GGUF (0.72)

## Visualization

To visualize the graph representations:

```bash
# Hypergraph (with operation nodes as diamonds)
dot -Tpng tiny_transformer_hypergraph.dot -o hypergraph.png

# DAG (all as boxes/circles)
dot -Tpng tiny_transformer_dag.dot -o dag.png

# SVG format for web
dot -Tsvg tiny_transformer_hypergraph.dot -o hypergraph.svg
dot -Tsvg tiny_transformer_dag.dot -o dag.svg
```

## Usage Examples

### Python API

```python
from gguf_workbench.representations import (
    HypergraphRepresentation,
    GraphRepresentation,
    SymbolicRepresentation,
    RepresentationComparator
)

# Load or create representations
hg = HypergraphRepresentation.from_tiny_transformer()
dag = GraphRepresentation.from_tiny_transformer()
sym = SymbolicRepresentation.from_tiny_transformer()

# Get statistics
print(hg.get_statistics())
print(dag.get_statistics())
print(sym.get_statistics())

# Export to files
hg.to_json("my_hypergraph.json")
dag.export_graphviz("my_dag.dot")
sym.export_markdown("my_symbolic.md")

# Run comparison
comp = RepresentationComparator.create_default_comparison()
comp.to_markdown("my_comparison.md")
```

### Command Line

```bash
# Generate all representations
python ../examples/generate_representations.py

# View specific representation
cat tiny_transformer_symbolic.md

# View comparison
cat representation_comparison.md
```

## Key Differences

### Hypergraph vs DAG

**Hypergraph advantages:**
- Single edge for attention operation (connects Q, K, V → scores)
- More compact for multi-input operations
- Natural for neural network analysis

**DAG advantages:**
- Standard graph algorithms (topological sort)
- Better tool support
- Easier to visualize

**Example - Attention Scores:**

Hypergraph (1 hyperedge):
```
Hyperedge: compute_attention_scores
  Sources: [attn_query, attn_key]
  Targets: [attn_scores]
  Operation: matmul_scaled
```

DAG (3 nodes, 3 edges):
```
attn_query → attn_scores_op
attn_key → attn_scores_op
attn_scores_op → attn_scores
```

### Symbolic vs Graph Representations

**Symbolic advantages:**
- Human-readable equations
- Compact notation
- Best for mathematical analysis

**Graph advantages:**
- Implementation details
- Execution order
- Tool-based analysis

**Example - Attention:**

Symbolic:
```
Q = h_0 W^Q
K = h_0 W^K
V = h_0 W^V
S = QK^T / sqrt(d_k)
A = softmax(S)
h_attn = (AV)W^O
```

Graph: Shows individual tensor nodes and operation nodes for each step.

## Detailed Analysis

See the parent directory's [`REPRESENTATION_ANALYSIS.md`](../REPRESENTATION_ANALYSIS.md) for:
- Complete comparison of all 8 representation formats
- Use case recommendations
- Implementation details
- Academic context

## Regenerating

To regenerate all representations:

```bash
cd /path/to/gguf
python examples/generate_representations.py
```

This will:
1. Create hypergraph representation (JSON, DOT)
2. Create DAG representation (JSON, DOT)
3. Create symbolic representation (JSON, MD, LaTeX)
4. Generate comparison analysis (JSON, MD)

## Related Files

- **Source code:** `gguf_workbench/representations/`
- **Tests:** `tests/test_representations.py`
- **Examples:** `examples/generate_representations.py`
- **Analysis:** `../REPRESENTATION_ANALYSIS.md`

## References

- [GGUF Format](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md)
- [Hypergraph Theory](https://en.wikipedia.org/wiki/Hypergraph)
- [Graphviz DOT Language](https://graphviz.org/doc/info/lang.html)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
