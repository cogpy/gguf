"""
List-based implementation of TinyTransformer inference.

This is the simplest implementation using only Python lists and basic operations.
It shows step-by-step how weights are applied to inputs using list indexing.

Advantages:
- Simplest to understand
- No dependencies
- Clear indexing operations
- Explicit computation steps

Disadvantages:
- Manual loop handling
- Less efficient than numpy
- Verbose for matrix operations
"""

import json
import math
from typing import List, Tuple, Optional


def load_model_weights(json_path: str) -> dict:
    """Load model weights from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def softmax(values: List[float]) -> List[float]:
    """Apply softmax to a list of values."""
    # Subtract max for numerical stability
    max_val = max(values)
    exp_values = [math.exp(v - max_val) for v in values]
    sum_exp = sum(exp_values)
    return [v / sum_exp for v in exp_values]


def dot_product(vec1: List[float], vec2: List[float]) -> float:
    """Compute dot product of two vectors."""
    return sum(a * b for a, b in zip(vec1, vec2))


def matrix_vector_multiply(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Multiply matrix by vector (matrix is list of rows)."""
    return [dot_product(row, vector) for row in matrix]


def matrix_multiply(mat1: List[List[float]], mat2: List[List[float]]) -> List[List[float]]:
    """Multiply two matrices."""
    # Transpose mat2 for easier column access
    mat2_T = [[mat2[j][i] for j in range(len(mat2))] for i in range(len(mat2[0]))]
    
    result = []
    for row in mat1:
        result.append([dot_product(row, col) for col in mat2_T])
    return result


def transpose(matrix: List[List[float]]) -> List[List[float]]:
    """Transpose a matrix."""
    if not matrix or not matrix[0]:
        return []
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def scale_matrix(matrix: List[List[float]], scale: float) -> List[List[float]]:
    """Scale all elements in a matrix."""
    return [[val * scale for val in row] for row in matrix]


class TinyTransformerListBased:
    """
    Pure list-based implementation of TinyTransformer.
    
    This implementation uses only Python lists and basic arithmetic,
    making it easy to understand how each operation works.
    
    Model structure:
    - Vocabulary size: 10 tokens (token_0 to token_9)
    - Embedding dimension: 5
    - Single attention head
    - Output dimension: 10 (vocabulary size)
    
    Example:
        >>> model = TinyTransformerListBased.from_json('tinytf/tiny_model_gguf.json')
        >>> tokens = [0, 1, 2]  # Input token IDs
        >>> logits = model.forward(tokens, trace=True)
        >>> predicted_token = logits.index(max(logits))
    """
    
    def __init__(
        self,
        embedding_weights: List[List[float]],
        attention_weights: List[List[float]],
        output_weights: List[List[float]],
        vocab_size: int = 10,
        embed_dim: int = 5,
        num_heads: int = 1,
    ):
        """
        Initialize TinyTransformer with weights.
        
        Args:
            embedding_weights: [vocab_size, embed_dim] embedding matrix
            attention_weights: [3 * embed_dim, embed_dim] Q, K, V projections stacked
            output_weights: [vocab_size, embed_dim] output projection
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
        """
        self.embedding_weights = embedding_weights
        self.attention_weights = attention_weights
        self.output_weights = output_weights
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Split attention weights into Q, K, V projections
        # attention_weights is [15, 5] for TinyTransformer
        # First 5 rows are Q, next 5 are K, last 5 are V
        rows_per_proj = len(attention_weights) // 3
        self.query_weights = attention_weights[:rows_per_proj]
        self.key_weights = attention_weights[rows_per_proj:2*rows_per_proj]
        self.value_weights = attention_weights[2*rows_per_proj:]
        
    @classmethod
    def from_json(cls, json_path: str) -> 'TinyTransformerListBased':
        """Load model from JSON file."""
        data = load_model_weights(json_path)
        
        return cls(
            embedding_weights=data['nodes']['embedding']['weights'],
            attention_weights=data['nodes']['attention']['weights'],
            output_weights=data['nodes']['output']['weights'],
            vocab_size=data['nodes']['embedding']['input_dim'],
            embed_dim=data['nodes']['embedding']['output_dim'],
            num_heads=data['nodes']['attention']['num_heads'],
        )
    
    def embed_tokens(self, token_ids: List[int], trace: bool = False) -> List[List[float]]:
        """
        Convert token IDs to embeddings by looking up in embedding matrix.
        
        This demonstrates indexed weight application:
        - For each token ID, we select the corresponding row from embedding_weights
        - Token 0 gets row 0, token 1 gets row 1, etc.
        
        Args:
            token_ids: List of token IDs (0-9 for TinyTransformer)
            trace: Whether to print trace information
            
        Returns:
            List of embedding vectors, one per token
        """
        embeddings = []
        
        for i, token_id in enumerate(token_ids):
            if trace:
                print(f"\n  Token {i}: ID={token_id} (token_{token_id})")
                print(f"  Looking up row {token_id} in embedding matrix")
            
            # This is the key operation: indexed weight lookup
            embedding = self.embedding_weights[token_id]
            
            if trace:
                print(f"  Embedding: {[f'{v:.4f}' for v in embedding]}")
            
            embeddings.append(embedding)
        
        return embeddings
    
    def attention(
        self, 
        embeddings: List[List[float]], 
        trace: bool = False
    ) -> List[List[float]]:
        """
        Apply self-attention mechanism.
        
        Shows how attention weights combine multiple input embeddings:
        1. Project embeddings to Query, Key, Value using learned weights
        2. Compute attention scores from Q and K
        3. Apply softmax to get attention weights
        4. Weighted combination of Values
        
        Args:
            embeddings: List of embedding vectors [seq_len, embed_dim]
            trace: Whether to print trace information
            
        Returns:
            Attended embeddings [seq_len, embed_dim]
        """
        if trace:
            print("\n=== Attention Mechanism ===")
            print(f"Input: {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
        
        # Step 1: Project to Q, K, V
        queries = [matrix_vector_multiply(self.query_weights, emb) for emb in embeddings]
        keys = [matrix_vector_multiply(self.key_weights, emb) for emb in embeddings]
        values = [matrix_vector_multiply(self.value_weights, emb) for emb in embeddings]
        
        if trace:
            print(f"\nProjected to Q, K, V (each {len(embeddings)} vectors of dim {len(queries[0])})")
            print(f"Example Query[0]: {[f'{v:.4f}' for v in queries[0]]}")
        
        # Step 2: Compute attention scores (Q @ K^T / sqrt(d_k))
        scale = 1.0 / math.sqrt(self.embed_dim)
        
        attended = []
        for i, query in enumerate(queries):
            if trace:
                print(f"\n  Processing position {i}:")
            
            # Compute scores with all keys
            scores = [dot_product(query, key) * scale for key in keys]
            
            if trace:
                print(f"    Raw scores: {[f'{s:.4f}' for s in scores]}")
            
            # Apply softmax to get attention weights
            attn_weights = softmax(scores)
            
            if trace:
                print(f"    Attention weights: {[f'{w:.4f}' for w in attn_weights]}")
                print(f"    Sum of weights: {sum(attn_weights):.6f}")
            
            # Weighted combination of values
            output = [0.0] * self.embed_dim
            for j, weight in enumerate(attn_weights):
                for k in range(self.embed_dim):
                    output[k] += weight * values[j][k]
            
            if trace:
                print(f"    Output: {[f'{v:.4f}' for v in output]}")
            
            attended.append(output)
        
        return attended
    
    def output_projection(
        self, 
        hidden_states: List[List[float]], 
        trace: bool = False
    ) -> List[List[float]]:
        """
        Project hidden states to vocabulary size for final predictions.
        
        Args:
            hidden_states: List of hidden vectors [seq_len, embed_dim]
            trace: Whether to print trace information
            
        Returns:
            Logits for each position [seq_len, vocab_size]
        """
        if trace:
            print("\n=== Output Projection ===")
            print(f"Projecting {len(hidden_states)} vectors to vocab size {self.vocab_size}")
        
        logits = []
        for i, hidden in enumerate(hidden_states):
            # Project to vocabulary size
            output_logits = matrix_vector_multiply(self.output_weights, hidden)
            
            if trace:
                print(f"\nPosition {i}:")
                print(f"  Hidden: {[f'{v:.4f}' for v in hidden]}")
                print(f"  Logits: {[f'{v:.4f}' for v in output_logits]}")
                
                # Find top prediction
                max_idx = output_logits.index(max(output_logits))
                print(f"  Top prediction: token_{max_idx} (logit={output_logits[max_idx]:.4f})")
            
            logits.append(output_logits)
        
        return logits
    
    def forward(
        self, 
        token_ids: List[int], 
        trace: bool = False,
        return_last_only: bool = False
    ) -> List[float]:
        """
        Full forward pass through the model.
        
        Args:
            token_ids: List of input token IDs
            trace: Whether to print detailed trace
            return_last_only: If True, return only last position's logits
            
        Returns:
            Logits for each token position (or just last if return_last_only=True)
        """
        if trace:
            print("=" * 80)
            print("TINY TRANSFORMER FORWARD PASS")
            print("=" * 80)
            print(f"\nInput tokens: {token_ids}")
            print(f"Token strings: {[f'token_{t}' for t in token_ids]}")
        
        # Step 1: Embed tokens
        if trace:
            print("\n" + "=" * 80)
            print("STEP 1: Token Embedding")
            print("=" * 80)
        
        embeddings = self.embed_tokens(token_ids, trace=trace)
        
        # Step 2: Self-attention
        if trace:
            print("\n" + "=" * 80)
            print("STEP 2: Self-Attention")
            print("=" * 80)
        
        attended = self.attention(embeddings, trace=trace)
        
        # Step 3: Output projection
        if trace:
            print("\n" + "=" * 80)
            print("STEP 3: Output Projection")
            print("=" * 80)
        
        logits = self.output_projection(attended, trace=trace)
        
        if trace:
            print("\n" + "=" * 80)
            print("FINAL PREDICTIONS")
            print("=" * 80)
            for i, token_logits in enumerate(logits):
                max_idx = token_logits.index(max(token_logits))
                print(f"Position {i}: Predicted token_{max_idx}")
        
        if return_last_only:
            return logits[-1]
        return logits
    
    def predict_next_token(self, token_ids: List[int], trace: bool = False) -> int:
        """
        Predict the next token given a sequence.
        
        Args:
            token_ids: List of input token IDs
            trace: Whether to print trace
            
        Returns:
            Predicted next token ID
        """
        logits = self.forward(token_ids, trace=trace, return_last_only=True)
        return logits.index(max(logits))
    
    def generate(
        self, 
        initial_tokens: List[int], 
        max_new_tokens: int = 5,
        trace: bool = False
    ) -> List[int]:
        """
        Generate tokens autoregressively.
        
        Args:
            initial_tokens: Starting sequence
            max_new_tokens: How many tokens to generate
            trace: Whether to print trace
            
        Returns:
            Complete sequence including initial and generated tokens
        """
        tokens = initial_tokens.copy()
        
        for i in range(max_new_tokens):
            if trace:
                print(f"\n{'='*80}")
                print(f"Generation step {i+1}/{max_new_tokens}")
                print(f"{'='*80}")
            
            next_token = self.predict_next_token(tokens, trace=trace)
            tokens.append(next_token)
            
            if trace:
                print(f"\nGenerated token_{next_token}")
                print(f"Current sequence: {tokens}")
        
        return tokens
