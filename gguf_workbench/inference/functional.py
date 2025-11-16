"""
Functional implementation of TinyTransformer inference.

This implementation uses pure functions with no side effects,
emphasizing functional programming principles.

Advantages:
- No mutable state
- Easy to test
- Clear data flow
- Composable functions

Disadvantages:
- Can be less efficient (copying data)
- May be less intuitive for some
- More function parameters
"""

import json
import math
from typing import List, Tuple, Dict, Any, Callable


# ============================================================================
# Pure helper functions
# ============================================================================

def dot_product(vec1: List[float], vec2: List[float]) -> float:
    """Compute dot product of two vectors (pure function)."""
    return sum(a * b for a, b in zip(vec1, vec2))


def matrix_vector_multiply(
    matrix: List[List[float]], 
    vector: List[float]
) -> List[float]:
    """Multiply matrix by vector (pure function)."""
    return [dot_product(row, vector) for row in matrix]


def softmax(values: List[float]) -> List[float]:
    """Apply softmax to values (pure function)."""
    if not values:
        return []
    
    max_val = max(values)
    exp_values = [math.exp(v - max_val) for v in values]
    sum_exp = sum(exp_values)
    return [v / sum_exp for v in exp_values]


def weighted_sum(
    vectors: List[List[float]], 
    weights: List[float]
) -> List[float]:
    """Compute weighted sum of vectors (pure function)."""
    if not vectors:
        return []
    
    dim = len(vectors[0])
    result = [0.0] * dim
    
    for vector, weight in zip(vectors, weights):
        for i in range(dim):
            result[i] += weight * vector[i]
    
    return result


# ============================================================================
# Model loading
# ============================================================================

def load_model_weights_functional(json_path: str) -> Tuple[
    List[List[float]],  # embedding_weights
    List[List[float]],  # query_weights
    List[List[float]],  # key_weights
    List[List[float]],  # value_weights
    List[List[float]],  # output_weights
    int,                 # embed_dim
]:
    """
    Load model weights from JSON file.
    
    Returns a tuple of all weights needed for inference (pure function).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    embedding_weights = data['nodes']['embedding']['weights']
    attention_weights = data['nodes']['attention']['weights']
    output_weights = data['nodes']['output']['weights']
    embed_dim = data['nodes']['embedding']['output_dim']
    
    # Split attention weights
    n_rows = len(attention_weights) // 3
    query_weights = attention_weights[:n_rows]
    key_weights = attention_weights[n_rows:2*n_rows]
    value_weights = attention_weights[2*n_rows:]
    
    return (
        embedding_weights,
        query_weights,
        key_weights,
        value_weights,
        output_weights,
        embed_dim,
    )


# ============================================================================
# Core transformer operations (pure functions)
# ============================================================================

def embed_tokens_functional(
    token_ids: List[int],
    embedding_matrix: List[List[float]],
) -> List[List[float]]:
    """
    Convert token IDs to embeddings (pure function).
    
    Args:
        token_ids: List of token IDs
        embedding_matrix: [vocab_size, embed_dim] weight matrix
        
    Returns:
        List of embedding vectors
    """
    return [embedding_matrix[token_id] for token_id in token_ids]


def project_to_qkv(
    embeddings: List[List[float]],
    query_weights: List[List[float]],
    key_weights: List[List[float]],
    value_weights: List[List[float]],
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """
    Project embeddings to Query, Key, Value (pure function).
    
    Args:
        embeddings: List of embedding vectors
        query_weights: [embed_dim, embed_dim] projection matrix
        key_weights: [embed_dim, embed_dim] projection matrix
        value_weights: [embed_dim, embed_dim] projection matrix
        
    Returns:
        Tuple of (queries, keys, values)
    """
    queries = [matrix_vector_multiply(query_weights, emb) for emb in embeddings]
    keys = [matrix_vector_multiply(key_weights, emb) for emb in embeddings]
    values = [matrix_vector_multiply(value_weights, emb) for emb in embeddings]
    
    return queries, keys, values


def compute_attention_scores(
    query: List[float],
    keys: List[List[float]],
    scale: float,
) -> List[float]:
    """
    Compute attention scores for one query (pure function).
    
    Args:
        query: Query vector
        keys: List of key vectors
        scale: Scaling factor (1/sqrt(d_k))
        
    Returns:
        List of attention scores
    """
    return [dot_product(query, key) * scale for key in keys]


def apply_attention(
    queries: List[List[float]],
    keys: List[List[float]],
    values: List[List[float]],
    embed_dim: int,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Apply self-attention mechanism (pure function).
    
    Args:
        queries: List of query vectors
        keys: List of key vectors
        values: List of value vectors
        embed_dim: Embedding dimension
        
    Returns:
        Tuple of (attention_weights, outputs)
    """
    scale = 1.0 / math.sqrt(embed_dim)
    
    attention_weights = []
    outputs = []
    
    for query in queries:
        # Compute scores
        scores = compute_attention_scores(query, keys, scale)
        
        # Apply softmax
        weights = softmax(scores)
        attention_weights.append(weights)
        
        # Weighted sum of values
        output = weighted_sum(values, weights)
        outputs.append(output)
    
    return attention_weights, outputs


def project_to_vocab(
    hidden_states: List[List[float]],
    output_weights: List[List[float]],
) -> List[List[float]]:
    """
    Project hidden states to vocabulary space (pure function).
    
    Args:
        hidden_states: List of hidden vectors
        output_weights: [vocab_size, embed_dim] projection matrix
        
    Returns:
        List of logit vectors
    """
    return [
        matrix_vector_multiply(output_weights, hidden)
        for hidden in hidden_states
    ]


def get_predictions(logits: List[List[float]]) -> List[int]:
    """
    Get predicted token IDs from logits (pure function).
    
    Args:
        logits: List of logit vectors
        
    Returns:
        List of predicted token IDs
    """
    return [logit_vec.index(max(logit_vec)) for logit_vec in logits]


# ============================================================================
# Main inference function
# ============================================================================

def tiny_transformer_functional(
    token_ids: List[int],
    embedding_weights: List[List[float]],
    query_weights: List[List[float]],
    key_weights: List[List[float]],
    value_weights: List[List[float]],
    output_weights: List[List[float]],
    embed_dim: int,
    return_details: bool = False,
) -> Any:
    """
    Pure functional implementation of TinyTransformer forward pass.
    
    This is a completely pure function with no side effects.
    All operations are composable and testable in isolation.
    
    Args:
        token_ids: Input token IDs
        embedding_weights: Embedding matrix
        query_weights: Query projection matrix
        key_weights: Key projection matrix
        value_weights: Value projection matrix
        output_weights: Output projection matrix
        embed_dim: Embedding dimension
        return_details: If True, return all intermediate values
        
    Returns:
        If return_details=False: List of predicted token IDs
        If return_details=True: Dictionary with all intermediate states
    """
    # Step 1: Embed tokens
    embeddings = embed_tokens_functional(token_ids, embedding_weights)
    
    # Step 2: Project to Q, K, V
    queries, keys, values = project_to_qkv(
        embeddings,
        query_weights,
        key_weights,
        value_weights,
    )
    
    # Step 3: Apply attention
    attention_weights, attention_outputs = apply_attention(
        queries, keys, values, embed_dim
    )
    
    # Step 4: Project to vocabulary
    logits = project_to_vocab(attention_outputs, output_weights)
    
    # Step 5: Get predictions
    predictions = get_predictions(logits)
    
    if return_details:
        return {
            'token_ids': token_ids,
            'embeddings': embeddings,
            'queries': queries,
            'keys': keys,
            'values': values,
            'attention_weights': attention_weights,
            'attention_outputs': attention_outputs,
            'logits': logits,
            'predictions': predictions,
        }
    else:
        return predictions


# ============================================================================
# Convenience wrappers
# ============================================================================

def create_inference_function(json_path: str) -> Callable[[List[int]], List[int]]:
    """
    Create a specialized inference function for a specific model.
    
    This demonstrates currying/partial application in functional programming.
    
    Args:
        json_path: Path to model JSON file
        
    Returns:
        A function that takes token_ids and returns predictions
    """
    # Load weights once
    (
        embedding_weights,
        query_weights,
        key_weights,
        value_weights,
        output_weights,
        embed_dim,
    ) = load_model_weights_functional(json_path)
    
    # Return a closure that captures the weights
    def inference_fn(token_ids: List[int]) -> List[int]:
        return tiny_transformer_functional(
            token_ids,
            embedding_weights,
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            embed_dim,
            return_details=False,
        )
    
    return inference_fn


def trace_inference_functional(
    token_ids: List[int],
    json_path: str,
) -> str:
    """
    Generate a trace of functional inference.
    
    Args:
        token_ids: Input token IDs
        json_path: Path to model JSON
        
    Returns:
        Formatted trace string
    """
    # Load weights
    (
        embedding_weights,
        query_weights,
        key_weights,
        value_weights,
        output_weights,
        embed_dim,
    ) = load_model_weights_functional(json_path)
    
    # Run with details
    result = tiny_transformer_functional(
        token_ids,
        embedding_weights,
        query_weights,
        key_weights,
        value_weights,
        output_weights,
        embed_dim,
        return_details=True,
    )
    
    lines = []
    lines.append("=" * 80)
    lines.append("FUNCTIONAL TINY TRANSFORMER TRACE")
    lines.append("=" * 80)
    lines.append("")
    
    # Input
    lines.append(f"Input: {result['token_ids']}")
    lines.append(f"Token names: {[f'token_{t}' for t in result['token_ids']]}")
    lines.append("")
    
    # Embeddings
    lines.append("Embeddings:")
    for i, emb in enumerate(result['embeddings']):
        lines.append(f"  Position {i}: [{', '.join(f'{v:.4f}' for v in emb)}]")
    lines.append("")
    
    # Attention weights
    lines.append("Attention Weights:")
    for i, weights in enumerate(result['attention_weights']):
        lines.append(f"  Position {i}:")
        for j, w in enumerate(weights):
            lines.append(f"    → Position {j}: {w:.4f}")
    lines.append("")
    
    # Predictions
    lines.append("Predictions:")
    for i, (pred_id, logits) in enumerate(zip(
        result['predictions'], 
        result['logits']
    )):
        lines.append(f"  Position {i}: token_{pred_id}")
        lines.append(f"    Logits: [{', '.join(f'{v:.4f}' for v in logits)}]")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


# ============================================================================
# Functional composition examples
# ============================================================================

def compose(*functions: Callable) -> Callable:
    """
    Compose functions right-to-left.
    
    compose(f, g, h)(x) = f(g(h(x)))
    """
    def composed(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    return composed


def pipeline(*functions: Callable) -> Callable:
    """
    Compose functions left-to-right.
    
    pipeline(f, g, h)(x) = h(g(f(x)))
    """
    def piped(arg):
        result = arg
        for func in functions:
            result = func(result)
        return result
    return piped


# Example of building inference pipeline with composition
def create_inference_pipeline(
    embedding_weights: List[List[float]],
    query_weights: List[List[float]],
    key_weights: List[List[float]],
    value_weights: List[List[float]],
    output_weights: List[List[float]],
    embed_dim: int,
) -> Callable[[List[int]], List[int]]:
    """
    Create an inference pipeline using functional composition.
    
    This shows how the transformer can be built as a pipeline of functions.
    """
    # Create partially applied functions
    def embed(token_ids):
        return embed_tokens_functional(token_ids, embedding_weights)
    
    def qkv(embeddings):
        return project_to_qkv(
            embeddings, 
            query_weights, 
            key_weights, 
            value_weights
        )
    
    def attn(qkv_tuple):
        queries, keys, values = qkv_tuple
        _, outputs = apply_attention(queries, keys, values, embed_dim)
        return outputs
    
    def vocab(hidden_states):
        return project_to_vocab(hidden_states, output_weights)
    
    def predict(logits):
        return get_predictions(logits)
    
    # Compose into pipeline
    return pipeline(embed, qkv, attn, vocab, predict)
