# Pure Python Inference Implementation for TinyTransformer

## Overview

This directory contains multiple pure Python implementations of the TinyTransformer model inference, demonstrating different programming paradigms and showing exactly how weights are applied to inputs at each step.

## Problem Statement

The goal is to provide the **best representation method** to show:
1. How indexed weights are applied to each element of the input
2. How all tokenized features relate to the vocabulary
3. How to determine the output through step-by-step computation
4. Various implementation approaches using different Python data structures and patterns

## Implementations

We provide **four different implementations**, each with distinct advantages:

### 1. List-Based Implementation (`list_based.py`)

**Approach:** Uses only Python lists and basic operations.

**Key Features:**
- Most explicit and transparent
- Shows direct weight indexing: `embedding_weights[token_id]`
- Manual loop handling for all operations
- No external dependencies (except json, math)

**Best For:**
- Learning how transformers work
- Understanding weight application step-by-step
- Teaching and educational purposes
- Debugging specific computations

**Example:**
```python
from gguf_workbench.inference import TinyTransformerListBased

model = TinyTransformerListBased.from_json('tinytf/tiny_model_gguf.json')

# Forward pass with detailed trace
logits = model.forward([0, 1, 2], trace=True)
# Shows: Token embedding lookup, attention computation, output projection

# Predict next token
next_token = model.predict_next_token([0, 1, 2])
print(f"Next token: token_{next_token}")

# Generate sequence
sequence = model.generate([0, 1], max_new_tokens=3)
print(f"Generated: {sequence}")
```

**How Weights Are Applied:**
```
Embedding: embedding_weights[token_id] → embedding_vector
  - Token 0: Select row 0 from 10×5 matrix
  - Token 1: Select row 1 from 10×5 matrix

Attention Q: sum(query_weights[i] * embedding[j] for j in range(5))
  - Each output[i] = dot(query_weights[i], embedding)

Attention Scores: dot(query, key) / sqrt(5)
  - Similarity between query and each key

Softmax: exp(score_i) / sum(exp(scores))
  - Normalize scores to probabilities

Output: sum(output_weights[i][j] * hidden[j])
  - Project hidden state to vocabulary
```

### 2. Dict-Based Implementation (`dict_based.py`)

**Approach:** Uses dictionaries with named keys for structured data.

**Key Features:**
- Named access to all components
- Easy to inspect intermediate states
- Returns comprehensive result dictionaries
- Good for debugging and analysis

**Best For:**
- Inspecting specific computation steps
- Understanding data flow
- JSON-compatible results
- Interactive exploration

**Example:**
```python
from gguf_workbench.inference import TinyTransformerDictBased

model = TinyTransformerDictBased.from_json('tinytf/tiny_model_gguf.json')

# Get full result with all intermediate states
result = model.forward([0, 1, 2], trace=False)

# Access specific components
print("Embeddings:", result['embedding']['embeddings'])
print("Attention weights:", result['attention']['attention_weights'])
print("Predictions:", result['predictions'])

# Generate detailed report
report = model.inspect_computation([0, 1, 2])
print(report)
```

**Result Structure:**
```python
{
    'input': {
        'token_ids': [0, 1, 2],
        'token_names': ['token_0', 'token_1', 'token_2']
    },
    'embedding': {
        'embeddings': [[...], [...], [...]],
        'embedding_indices': [0, 1, 2]
    },
    'attention': {
        'queries': [...],
        'keys': [...],
        'values': [...],
        'attention_scores': [...],
        'attention_weights': [...],
        'outputs': [...]
    },
    'output': {
        'logits': [...],
        'predictions': [2, 7, 2],
        'prediction_names': ['token_2', 'token_7', 'token_2']
    }
}
```

### 3. Class-Based Implementation (`class_based.py`)

**Approach:** Uses dataclasses and OOP principles with type hints.

**Key Features:**
- Type-safe with full type annotations
- Clean separation of concerns
- Dataclasses for structured data (`Token`, `Embedding`, `AttentionHead`, etc.)
- Production-ready code quality

**Best For:**
- Production use
- Large codebases
- Type checking with mypy
- Maintainable code

**Example:**
```python
from gguf_workbench.inference import TinyTransformerClassBased

model = TinyTransformerClassBased.from_json('tinytf/tiny_model_gguf.json')

# Forward pass returns list of LogitPrediction objects
predictions = model.forward([0, 1, 2], trace=False)

for pred in predictions:
    print(f"Position {pred.position}: {pred.predicted_token_name}")
    print(f"  Logits: {pred.logits}")
    print(f"  Confidence: {pred.logits[pred.predicted_token_id]:.4f}")

# Get attention matrix for visualization
attn_matrix = model.get_attention_matrix([0, 1, 2])

# Explain specific prediction
explanation = model.explain_prediction([0, 1, 2], position=1)
print(explanation)
```

**Type System:**
```python
@dataclass
class Token:
    id: int
    position: int
    name: str

@dataclass
class Embedding:
    token: Token
    vector: List[float]

@dataclass
class AttentionHead:
    position: int
    query: List[float]
    key: List[float]
    value: List[float]
    attention_scores: List[float]
    attention_weights: List[float]
    output: List[float]

@dataclass
class LogitPrediction:
    position: int
    logits: List[float]
    predicted_token_id: int
    predicted_token_name: str
```

### 4. Functional Implementation (`functional.py`)

**Approach:** Pure functions with no side effects.

**Key Features:**
- No mutable state
- All functions are pure (same input → same output)
- Composable and testable
- Functional programming paradigm

**Best For:**
- Testing and verification
- Mathematical correctness
- Parallel processing
- Composition and pipelines

**Example:**
```python
from gguf_workbench.inference.functional import (
    tiny_transformer_functional,
    load_model_weights_functional,
    create_inference_function,
    trace_inference_functional,
)

# Load weights (returns tuple)
weights = load_model_weights_functional('tinytf/tiny_model_gguf.json')

# Simple inference
predictions = tiny_transformer_functional(
    [0, 1, 2],
    *weights,
    return_details=False
)

# Get detailed trace
trace = trace_inference_functional([0, 1, 2], 'tinytf/tiny_model_gguf.json')
print(trace)

# Create specialized inference function (closure)
infer = create_inference_function('tinytf/tiny_model_gguf.json')
predictions = infer([5, 6, 7])
```

**Pure Functions:**
```python
# All operations are pure functions
def dot_product(vec1: List[float], vec2: List[float]) -> float:
    return sum(a * b for a, b in zip(vec1, vec2))

def softmax(values: List[float]) -> List[float]:
    max_val = max(values)
    exp_values = [math.exp(v - max_val) for v in values]
    sum_exp = sum(exp_values)
    return [v / sum_exp for v in exp_values]

def embed_tokens_functional(
    token_ids: List[int],
    embedding_matrix: List[List[float]],
) -> List[List[float]]:
    return [embedding_matrix[token_id] for token_id in token_ids]
```

## Tracing and Visualization

### InferenceTracer (`trace.py`)

Provides detailed step-by-step execution traces showing exactly how weights are applied.

**Example:**
```python
from gguf_workbench.inference import InferenceTracer

tracer = InferenceTracer()

# Record embedding lookup
tracer.trace_embedding_lookup([0, 1, 2], embedding_weights)

# Record matrix operations
tracer.trace_matrix_vector_multiply(
    "query_projection",
    query_weights,
    embedding,
    query_result
)

# Generate report
report = tracer.get_report(verbose=True)
print(report)

# Visualize attention
viz = tracer.visualize_attention_flow(
    attention_weights,
    ['token_0', 'token_1', 'token_2']
)
print(viz)
```

## How Weights Are Applied

### 1. Token Embedding (Indexed Lookup)

```
Input: token_id = 2
Operation: embedding = embedding_weights[2]
Result: [0.648, -0.176, -1.841, 0.797] (5-dimensional vector)

Visualization:
embedding_weights = [
    [row 0: token_0 embedding],
    [row 1: token_1 embedding],
    [row 2: token_2 embedding],  ← SELECT THIS ROW
    [row 3: token_3 embedding],
    ...
]
```

### 2. Matrix-Vector Multiplication (Query/Key/Value Projection)

```
Query = query_weights @ embedding

For each output dimension i:
    query[i] = sum(query_weights[i][j] * embedding[j] for j in 0..4)

Example:
query[0] = -0.0418 * 0.648 + 0.529 * (-0.176) + ... + (-0.144) * 0.797
         = -0.027 + (-0.093) + ... + (-0.115)
         = -0.235 (approximately)

This applies learned weights to transform the embedding.
```

### 3. Attention Scores (Dot Product + Scaling)

```
score[i,j] = dot(query[i], key[j]) / sqrt(d_k)

For position 1 attending to position 0:
    score[1,0] = (q1[0]*k0[0] + q1[1]*k0[1] + ... + q1[4]*k0[4]) / sqrt(5)

This measures similarity between query and key vectors.
```

### 4. Softmax (Normalization)

```
weight[i] = exp(score[i]) / sum(exp(score[j]) for all j)

Converts scores to probabilities that sum to 1:
    scores     = [-0.5, 0.2, -0.1]
    exp(scores) = [0.607, 1.221, 0.905]
    weights    = [0.224, 0.451, 0.334]  (sum = 1.0)

These weights determine how much to attend to each position.
```

### 5. Weighted Sum (Value Aggregation)

```
output = sum(attention_weight[j] * value[j] for all j)

For dimension d:
    output[d] = 0.224 * value[0][d] + 0.451 * value[1][d] + 0.334 * value[2][d]

Combines values from all positions using learned attention patterns.
```

### 6. Output Projection (Vocabulary Prediction)

```
logits = output_weights @ hidden_state

For each token t in vocabulary:
    logits[t] = sum(output_weights[t][d] * hidden_state[d] for d in 0..4)

Final prediction = argmax(logits)
```

## Vocabulary Mapping

The TinyTransformer uses a simple 10-token vocabulary:

```
Token ID  ←→  Token String  ←→  Embedding Row
   0      ←→    "token_0"   ←→  embedding_weights[0]
   1      ←→    "token_1"   ←→  embedding_weights[1]
   2      ←→    "token_2"   ←→  embedding_weights[2]
   ...
   9      ←→    "token_9"   ←→  embedding_weights[9]
```

**Forward (Tokenization → Embedding):**
1. Input text: "token_2" → Token ID: 2
2. Token ID: 2 → Embedding: embedding_weights[2]

**Backward (Logits → Prediction):**
1. Logits: [0.2, -0.1, 0.8, ...] (10 values)
2. argmax: position 2 has highest value
3. Predicted token ID: 2 → "token_2"

## Comparison of Approaches

| Feature | List | Dict | Class | Functional |
|---------|------|------|-------|------------|
| **Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Transparency** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Type Safety** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Maintainability** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Testability** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Performance** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Inspection** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Production Ready** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## Language Effectiveness

### Python (Current Implementation)
**Advantages:**
- ✓ Excellent for educational purposes
- ✓ Clear, readable syntax
- ✓ Rich ecosystem for ML
- ✓ Interactive development (REPL)
- ✓ Multiple paradigms supported

**Disadvantages:**
- ✗ Slower than compiled languages
- ✗ Manual list operations less efficient than NumPy

### Other Effective Languages

**JavaScript/TypeScript:**
- Good for web-based visualization
- Similar expressiveness to Python
- Runs in browser for interactive demos

**Julia:**
- Fast as C, easy as Python
- Excellent for numerical computing
- Multiple dispatch for clean code

**Rust:**
- Maximum performance
- Type safety
- Good for production inference

**C++:**
- Best performance
- Used in llama.cpp
- More complex but fastest

## Running Examples

### Quick Start

```bash
# Run comprehensive demonstration
python examples/demonstrate_inference.py

# The demo will show:
# 1. List-based implementation with trace
# 2. Dict-based with structured output
# 3. Class-based with type safety
# 4. Functional with pure functions
# 5. Comparison of all implementations
# 6. Attention visualization
```

### Individual Tests

```python
# Test list-based
from gguf_workbench.inference import TinyTransformerListBased
model = TinyTransformerListBased.from_json('tinytf/tiny_model_gguf.json')
model.forward([0, 1, 2], trace=True)

# Test dict-based
from gguf_workbench.inference import TinyTransformerDictBased
model = TinyTransformerDictBased.from_json('tinytf/tiny_model_gguf.json')
print(model.inspect_computation([0, 1, 2]))

# Test class-based
from gguf_workbench.inference import TinyTransformerClassBased
model = TinyTransformerClassBased.from_json('tinytf/tiny_model_gguf.json')
print(model.explain_prediction([0, 1, 2], position=1))

# Test functional
from gguf_workbench.inference.functional import trace_inference_functional
print(trace_inference_functional([0, 1, 2], 'tinytf/tiny_model_gguf.json'))
```

## Files

- `__init__.py` - Module exports
- `list_based.py` - List-based implementation (13KB)
- `dict_based.py` - Dict-based implementation (12KB)
- `class_based.py` - Class-based implementation (14KB)
- `functional.py` - Functional implementation (14KB)
- `trace.py` - Inference tracer (13KB)

## Summary

This implementation provides:

1. ✅ **Multiple representations** showing weight application at each step
2. ✅ **Complete transparency** into how transformers process input
3. ✅ **Vocabulary mapping** clearly demonstrated
4. ✅ **Four different paradigms** (List, Dict, Class, Functional)
5. ✅ **Pure Python** with no ML framework dependencies
6. ✅ **Educational focus** with extensive documentation
7. ✅ **Production-ready** class-based option

**Recommended Approach:**
- **Learning:** Start with list-based for transparency
- **Debugging:** Use dict-based for inspection
- **Production:** Use class-based for maintainability
- **Testing:** Use functional for correctness verification

All implementations produce identical results and can be used interchangeably.
