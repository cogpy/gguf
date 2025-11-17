# Vocabulary and Layer Activation Analysis Guide

## Overview

The vocabulary and layer activation analysis feature provides deep insights into how vocabulary usage in conversations relates to the total available vocabulary of a model, and which layers and activations are likely involved in processing that vocabulary.

This analysis directly addresses questions about:
- **Vocabulary enumeration**: What words are actually used vs. available?
- **Coverage analysis**: How much of the model's vocabulary is expressed?
- **Layer activation inference**: Which transformer layers are most active?
- **Echo state reservoir dynamics**: How does the reservoir computing framework interact with transformer layers?
- **Dynamic persona variables**: What persona dimensions are reflected in vocabulary patterns?

## Quick Start

### Command Line

```bash
# Basic vocabulary analysis
gguf-workbench analyze-vocabulary conversations.json \
  --vocab-size 50257 \
  --layers 12 \
  --embedding-dim 768 \
  --architecture gpt2

# Save detailed report to JSON
gguf-workbench analyze-vocabulary conversations.json \
  --vocab-size 50257 \
  --layers 12 \
  -o vocab_analysis_report.json
```

### Python API

```python
from gguf_workbench import VocabularyAnalyzer, load_conversations_from_json

# Load conversations
conversations = load_conversations_from_json('conversations.json')

# Create analyzer with model configuration
analyzer = VocabularyAnalyzer(
    total_vocab_size=50257,  # GPT-2 vocabulary size
    model_layers=12,          # Number of transformer layers
    embedding_dim=768,        # Embedding dimension
    architecture="gpt2"
)

# Generate comprehensive report
report = analyzer.generate_comprehensive_report(conversations)

# Access specific analyses
vocab_analysis = report['vocabulary_analysis']
layer_insights = report['layer_activation_analysis']
echo_analysis = report['echo_reservoir_analysis']

# Print summary
print(report['summary'])
```

## Input Format

The analyzer accepts conversations in multiple formats:

### JSON Array Format

```json
[
  {
    "query": "What is machine learning?",
    "response": "Machine learning is a subset of AI...",
    "model": "gpt-4"
  },
  {
    "query": "How does it work?",
    "response": "It learns from data and patterns...",
    "model": "gpt-4"
  }
]
```

### JSONL Format

```jsonl
{"query": "Question 1", "response": "Answer 1", "model": "gpt-4"}
{"query": "Question 2", "response": "Answer 2", "model": "gpt-3.5"}
```

### Messages Format (ChatGPT style)

```jsonl
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}], "model": "gpt-4"}
```

## Analysis Components

### 1. Vocabulary Enumeration

Complete enumeration of all words used in conversations with frequency counts.

**Output includes:**
- Total unique words
- Total word instances
- Vocabulary diversity (unique/total ratio)
- Top N most frequent words
- Query vs. response vocabulary breakdown
- Per-model vocabulary statistics (if model metadata present)

### 2. Coverage Analysis

Relates expressed vocabulary to total available vocabulary.

**Metrics:**
- Expressed vocabulary size
- Total vocabulary size
- Coverage ratio (0.0 to 1.0)
- Coverage percentage
- Unexpressed vocabulary size

**Interpretation:**
- < 1%: Very sparse activation
- 1-5%: Sparse activation
- 5-20%: Moderate activation
- 20-50%: High activation
- > 50%: Very high activation

### 3. Layer Activation Inference

Estimates which transformer layers are most active based on vocabulary patterns.

**Analyzes:**
- **Early layers** (first 25%): Embedding lookup and basic pattern recognition
- **Middle layers** (25-75%): Contextual composition and semantic processing
- **Late layers** (75-100%): Task-specific processing and output generation

**Evidence considered:**
- Vocabulary complexity
- Word frequency distribution
- Response generation patterns

### 4. Attention Pattern Inference

Infers which attention mechanisms are likely active.

**Patterns detected:**
- Cross-attention (query to response)
- Self-attention (within query)
- Self-attention (within response, causal)
- Multi-head diversity

### 5. Echo State Reservoir Analysis

Analyzes how the echo state reservoir computing framework interacts with transformer layers.

**Components analyzed:**
- **Reservoir dynamics**: State space dimensionality, recurrent connections
- **Feedback loops**: Echo ratio, feedback strength
- **Temporal memory**: Context encoding in reservoir state
- **State update mechanism**: How tokens update reservoir state

**Feedback strength levels:**
- Weak: < 10% echo ratio
- Moderate: 10-30% echo ratio
- Strong: > 30% echo ratio

### 6. Dynamic Persona Variables

Detects persona dimensions reflected in vocabulary.

**Dimensions tracked:**
- Formal language (please, kindly, respectfully, etc.)
- Casual language (hey, yeah, cool, etc.)
- Technical language (algorithm, function, implementation, etc.)
- Emotional language (feel, believe, think, etc.)
- Cognitive language (understand, analyze, consider, etc.)

### 7. Temporal Vocabulary Patterns

Analyzes how vocabulary evolves over the conversation sequence.

**Patterns:**
- Expanding: Vocabulary grows over time (increasing complexity)
- Stable: Vocabulary remains consistent
- Contracting: Vocabulary narrows (focus/convergence)

## Use Cases

### 1. Model Capability Assessment

Understand what fraction of a model's vocabulary is being exercised:

```python
analyzer = VocabularyAnalyzer(total_vocab_size=50257)
report = analyzer.generate_comprehensive_report(conversations)

coverage = report['vocabulary_analysis']['vocabulary_coverage']
print(f"Using {coverage['coverage_percentage']:.1f}% of vocabulary")
```

### 2. Layer-Specific Interpretability

Target interpretability efforts at the most active layers:

```python
layer_insights = report['layer_activation_analysis']
for estimate in layer_insights['layer_activation_estimates']:
    if estimate['activation_level'] == 'very high':
        print(f"Focus on layers {estimate['layer_range']}: {estimate['primary_function']}")
```

### 3. Reservoir Computing Research

Understand echo state dynamics:

```python
echo = report['echo_reservoir_analysis']
feedback = echo['feedback_loops']

if feedback['feedback_strength'] == 'strong':
    print("Strong echo effects detected - reservoir maintains context well")
```

### 4. Persona Engineering

Identify active persona dimensions for targeted training:

```python
persona = echo['persona_variables']
print(f"Dominant persona: {persona['dominant_persona']}")
print(f"Active dimensions: {persona['persona_diversity']}")
```

### 5. Dataset Evaluation

Assess whether your conversation dataset is diverse enough:

```python
recommendations = echo['recommendations']
for rec in recommendations:
    print(f"• {rec}")
```

## Advanced Features

### Per-Model Analysis

When conversations include model metadata, get per-model vocabulary statistics:

```python
vocab_stats = analyzer.enumerate_vocabulary(conversations, include_metadata=True)

for model, stats in vocab_stats['per_model_vocabulary'].items():
    print(f"\n{model}:")
    print(f"  Unique words: {stats['unique_words']}")
    print(f"  Top words: {stats['top_words']}")
```

### Custom Model Configurations

Provide detailed model configuration for more accurate analysis:

```python
analyzer = VocabularyAnalyzer(
    total_vocab_size=128000,  # Custom vocabulary
    model_layers=32,          # Deeper model
    embedding_dim=4096,       # Larger embeddings
    architecture="llama"      # Specific architecture
)
```

### Loading from Different Formats

The `load_conversations_from_json` function handles multiple formats automatically:

```python
# Handles .json, .jsonl, messages format, and mapping format
conversations = load_conversations_from_json('any_format.json')
```

## Output Reference

### Complete Report Structure

```python
{
    'vocabulary_analysis': {
        'total_unique_words': int,
        'total_word_instances': int,
        'word_counts': dict,
        'vocabulary_diversity': float,
        'vocabulary_coverage': {
            'expressed_vocab_size': int,
            'total_vocab_size': int,
            'coverage_ratio': float,
            'coverage_percentage': float,
            'unexpressed_vocab_size': int
        },
        'per_model_vocabulary': dict  # if metadata present
    },
    'layer_activation_analysis': {
        'embedding_layer_analysis': dict,
        'layer_activation_estimates': list,
        'attention_head_insights': list,
        'output_layer_analysis': dict
    },
    'echo_reservoir_analysis': {
        'reservoir_dynamics': dict,
        'feedback_loops': dict,
        'persona_variables': dict,
        'temporal_patterns': dict,
        'recommendations': list
    },
    'model_configuration': {
        'total_vocab_size': int,
        'model_layers': int,
        'embedding_dim': int,
        'architecture': str
    },
    'summary': str
}
```

## Best Practices

1. **Provide model configuration**: For the most accurate analysis, provide all model parameters (vocab size, layers, embedding dim).

2. **Use model metadata**: Include model information in your conversation data for per-model analysis.

3. **Analyze sufficient data**: Larger conversation datasets provide more reliable insights.

4. **Iterate on recommendations**: Use the generated recommendations to improve your dataset or analysis.

5. **Save detailed reports**: Use `-o` flag or `to_json_file()` to save complete analysis for later reference.

6. **Combine with other analyses**: Use alongside the standard conversation analysis (`analyze-conversations`) for complete understanding.

## Examples

See `examples/analyze_vocabulary_layers.py` for a complete demonstration with sample data.

## Integration with Deep Tree Echo Framework

This analysis is specifically designed to work with the deep tree echo state reservoir computing framework:

- **Reservoir state space** is estimated from vocabulary dimensionality
- **Feedback loops** measure how query information echoes into responses
- **Temporal patterns** show how reservoir state evolves
- **Layer activation inference** shows transformer-reservoir interaction points

The combination of reservoir computing and transformer architecture creates a powerful hybrid where:
- Reservoir provides rich temporal dynamics
- Transformers perform learned readout and composition
- Vocabulary patterns reveal which components are most active

## Theoretical Background

### Under-Determined Systems

With N model parameters and M conversation pairs where M << N, the system is vastly under-determined. This analysis helps understand:
- What CAN be inferred from vocabulary patterns
- What CANNOT be directly recovered (exact weights, activations)
- How to make the most of available data

### Vocabulary as a Probe

Vocabulary usage serves as a natural probe into model internals:
- Sparse vocabulary → limited activation
- Diverse vocabulary → complex processing
- Echo patterns → feedback strength
- Temporal evolution → context integration

This makes vocabulary analysis a practical, non-invasive method for understanding model behavior.

## Related Documentation

- `CONVERSATION_ANALYSIS_GUIDE.md` - Standard conversation analysis
- `README.md` - General GGUF Workbench documentation
- `examples/analyze_vocabulary_layers.py` - Complete example code
