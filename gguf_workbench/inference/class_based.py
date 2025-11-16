"""
Class-based implementation of TinyTransformer inference.

This implementation uses dataclasses to represent each component,
making the code more structured and type-safe.

Advantages:
- Type hints for clarity
- Structured data with validation
- Clean separation of concerns
- Easy to extend and maintain

Disadvantages:
- More boilerplate code
- Steeper learning curve
- Additional abstraction layers
"""

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Token:
    """Represents a single token."""
    id: int
    position: int
    name: str = field(init=False)
    
    def __post_init__(self):
        self.name = f"token_{self.id}"


@dataclass
class Embedding:
    """Represents an embedding vector."""
    token: Token
    vector: List[float]
    
    @property
    def dim(self) -> int:
        return len(self.vector)


@dataclass
class AttentionHead:
    """Represents attention computation for one position."""
    position: int
    query: List[float]
    key: List[float]
    value: List[float]
    
    attention_scores: List[float] = field(default_factory=list)
    attention_weights: List[float] = field(default_factory=list)
    output: List[float] = field(default_factory=list)


@dataclass
class LogitPrediction:
    """Represents output logits and prediction for one position."""
    position: int
    logits: List[float]
    predicted_token_id: int
    predicted_token_name: str = field(init=False)
    
    def __post_init__(self):
        self.predicted_token_name = f"token_{self.predicted_token_id}"


@dataclass
class ModelWeights:
    """Container for all model weights."""
    embedding_matrix: List[List[float]]
    query_projection: List[List[float]]
    key_projection: List[List[float]]
    value_projection: List[List[float]]
    output_projection: List[List[float]]
    
    vocab_size: int
    embed_dim: int
    num_heads: int


class MatrixOps:
    """Helper class for matrix operations."""
    
    @staticmethod
    def dot_product(vec1: List[float], vec2: List[float]) -> float:
        """Compute dot product of two vectors."""
        return sum(a * b for a, b in zip(vec1, vec2))
    
    @staticmethod
    def matrix_vector_multiply(
        matrix: List[List[float]], 
        vector: List[float]
    ) -> List[float]:
        """Multiply matrix by vector."""
        return [MatrixOps.dot_product(row, vector) for row in matrix]
    
    @staticmethod
    def softmax(values: List[float]) -> List[float]:
        """Apply softmax to values."""
        max_val = max(values)
        exp_values = [math.exp(v - max_val) for v in values]
        sum_exp = sum(exp_values)
        return [v / sum_exp for v in exp_values]
    
    @staticmethod
    def weighted_sum(
        vectors: List[List[float]], 
        weights: List[float]
    ) -> List[float]:
        """Compute weighted sum of vectors."""
        if not vectors:
            return []
        
        dim = len(vectors[0])
        result = [0.0] * dim
        
        for vector, weight in zip(vectors, weights):
            for i in range(dim):
                result[i] += weight * vector[i]
        
        return result


class TinyTransformerClassBased:
    """
    Class-based implementation of TinyTransformer using dataclasses.
    
    This implementation emphasizes clean code structure with type hints
    and separation of concerns.
    
    Example:
        >>> model = TinyTransformerClassBased.from_json('tinytf/tiny_model_gguf.json')
        >>> predictions = model.forward([0, 1, 2], trace=True)
        >>> for pred in predictions:
        ...     print(f"Position {pred.position}: {pred.predicted_token_name}")
    """
    
    def __init__(self, weights: ModelWeights):
        """Initialize with model weights."""
        self.weights = weights
        self.ops = MatrixOps()
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TinyTransformerClassBased':
        """Load model from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract weights
        embedding_weights = data['nodes']['embedding']['weights']
        attention_weights = data['nodes']['attention']['weights']
        output_weights = data['nodes']['output']['weights']
        
        # Split attention weights into Q, K, V
        n_rows = len(attention_weights) // 3
        query_weights = attention_weights[:n_rows]
        key_weights = attention_weights[n_rows:2*n_rows]
        value_weights = attention_weights[2*n_rows:]
        
        weights = ModelWeights(
            embedding_matrix=embedding_weights,
            query_projection=query_weights,
            key_projection=key_weights,
            value_projection=value_weights,
            output_projection=output_weights,
            vocab_size=data['nodes']['embedding']['input_dim'],
            embed_dim=data['nodes']['embedding']['output_dim'],
            num_heads=data['nodes']['attention']['num_heads'],
        )
        
        return cls(weights)
    
    def create_tokens(self, token_ids: List[int], trace: bool = False) -> List[Token]:
        """Create Token objects from IDs."""
        tokens = [Token(id=tid, position=i) for i, tid in enumerate(token_ids)]
        
        if trace:
            print("\n=== Input Tokens ===")
            for token in tokens:
                print(f"Position {token.position}: {token.name} (ID={token.id})")
        
        return tokens
    
    def embed_tokens(self, tokens: List[Token], trace: bool = False) -> List[Embedding]:
        """
        Convert tokens to embeddings.
        
        This demonstrates the indexed weight lookup:
        For each token, we select the corresponding row from the embedding matrix.
        """
        embeddings = []
        
        if trace:
            print("\n=== Token Embedding ===")
        
        for token in tokens:
            # Key operation: index into embedding matrix
            embedding_vector = self.weights.embedding_matrix[token.id]
            embedding = Embedding(token=token, vector=embedding_vector)
            embeddings.append(embedding)
            
            if trace:
                print(f"\n{token.name} (position {token.position}):")
                print(f"  Embedding matrix row {token.id}")
                print(f"  Vector: [{', '.join(f'{v:.4f}' for v in embedding.vector)}]")
        
        return embeddings
    
    def compute_attention(
        self, 
        embeddings: List[Embedding], 
        trace: bool = False
    ) -> List[AttentionHead]:
        """
        Compute self-attention for all positions.
        
        Shows how attention combines information from different positions
        using learned weight matrices and dynamic attention weights.
        """
        if trace:
            print("\n=== Self-Attention ===")
        
        # Project all embeddings to Q, K, V
        attention_heads = []
        
        for emb in embeddings:
            query = self.ops.matrix_vector_multiply(
                self.weights.query_projection, 
                emb.vector
            )
            key = self.ops.matrix_vector_multiply(
                self.weights.key_projection, 
                emb.vector
            )
            value = self.ops.matrix_vector_multiply(
                self.weights.value_projection, 
                emb.vector
            )
            
            head = AttentionHead(
                position=emb.token.position,
                query=query,
                key=key,
                value=value,
            )
            attention_heads.append(head)
        
        if trace:
            print(f"\nProjected {len(attention_heads)} positions to Q, K, V")
            print(f"Q/K/V dimension: {len(attention_heads[0].query)}")
        
        # Compute attention scores and outputs
        scale = 1.0 / math.sqrt(self.weights.embed_dim)
        
        for i, head in enumerate(attention_heads):
            if trace:
                print(f"\n--- Position {i} ({embeddings[i].token.name}) ---")
            
            # Compute scores with all keys
            scores = [
                self.ops.dot_product(head.query, other_head.key) * scale
                for other_head in attention_heads
            ]
            head.attention_scores = scores
            
            # Apply softmax
            weights = self.ops.softmax(scores)
            head.attention_weights = weights
            
            if trace:
                print("Attention weights:")
                for j, w in enumerate(weights):
                    print(f"  Position {j}: {w:.4f}")
            
            # Weighted combination of values
            values = [other_head.value for other_head in attention_heads]
            output = self.ops.weighted_sum(values, weights)
            head.output = output
            
            if trace:
                print(f"Output: [{', '.join(f'{v:.4f}' for v in output)}]")
        
        return attention_heads
    
    def predict_tokens(
        self, 
        attention_heads: List[AttentionHead], 
        trace: bool = False
    ) -> List[LogitPrediction]:
        """
        Project attention outputs to vocabulary space.
        
        For each position, computes logits over the vocabulary
        and determines the most likely next token.
        """
        predictions = []
        
        if trace:
            print("\n=== Output Projection ===")
        
        for head in attention_heads:
            # Project to vocabulary size
            logits = self.ops.matrix_vector_multiply(
                self.weights.output_projection,
                head.output
            )
            
            # Find prediction
            predicted_id = logits.index(max(logits))
            
            prediction = LogitPrediction(
                position=head.position,
                logits=logits,
                predicted_token_id=predicted_id,
            )
            predictions.append(prediction)
            
            if trace:
                print(f"\nPosition {head.position}:")
                print(f"  Logits: [{', '.join(f'{v:.4f}' for v in logits)}]")
                print(f"  Prediction: {prediction.predicted_token_name}")
                print(f"  Confidence: {logits[predicted_id]:.4f}")
        
        return predictions
    
    def forward(
        self, 
        token_ids: List[int], 
        trace: bool = False
    ) -> List[LogitPrediction]:
        """
        Full forward pass through the model.
        
        Returns a list of predictions, one for each input position.
        """
        if trace:
            print("=" * 80)
            print("CLASS-BASED TINY TRANSFORMER")
            print("=" * 80)
        
        # Step 1: Create token objects
        tokens = self.create_tokens(token_ids, trace=trace)
        
        # Step 2: Embed tokens
        embeddings = self.embed_tokens(tokens, trace=trace)
        
        # Step 3: Self-attention
        attention_heads = self.compute_attention(embeddings, trace=trace)
        
        # Step 4: Output prediction
        predictions = self.predict_tokens(attention_heads, trace=trace)
        
        if trace:
            print("\n" + "=" * 80)
            print("FINAL PREDICTIONS")
            print("=" * 80)
            for pred in predictions:
                print(f"Position {pred.position}: {pred.predicted_token_name}")
        
        return predictions
    
    def get_attention_matrix(self, token_ids: List[int]) -> List[List[float]]:
        """
        Get the attention weight matrix for visualization.
        
        Returns a matrix where element [i][j] is the attention weight
        from position i to position j.
        """
        tokens = self.create_tokens(token_ids)
        embeddings = self.embed_tokens(tokens)
        attention_heads = self.compute_attention(embeddings)
        
        return [head.attention_weights for head in attention_heads]
    
    def explain_prediction(self, token_ids: List[int], position: int) -> str:
        """
        Generate a detailed explanation of prediction for a specific position.
        
        Args:
            token_ids: Input token sequence
            position: Which position to explain
            
        Returns:
            Formatted explanation string
        """
        tokens = self.create_tokens(token_ids)
        embeddings = self.embed_tokens(tokens)
        attention_heads = self.compute_attention(embeddings)
        predictions = self.predict_tokens(attention_heads)
        
        if position >= len(predictions):
            return f"Invalid position {position}, sequence has {len(predictions)} tokens"
        
        pred = predictions[position]
        head = attention_heads[position]
        emb = embeddings[position]
        
        lines = []
        lines.append("=" * 80)
        lines.append(f"PREDICTION EXPLANATION - Position {position}")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"Input Token: {emb.token.name}")
        lines.append(f"Embedding: [{', '.join(f'{v:.4f}' for v in emb.vector)}]")
        lines.append("")
        
        lines.append("Attention Weights (what this position attends to):")
        for i, w in enumerate(head.attention_weights):
            lines.append(f"  Position {i} ({tokens[i].name}): {w:.4f}")
        lines.append("")
        
        lines.append(f"Attention Output: [{', '.join(f'{v:.4f}' for v in head.output)}]")
        lines.append("")
        
        lines.append("Output Logits:")
        for i, logit in enumerate(pred.logits):
            marker = " ← PREDICTED" if i == pred.predicted_token_id else ""
            lines.append(f"  token_{i}: {logit:.4f}{marker}")
        lines.append("")
        
        lines.append(f"Predicted Token: {pred.predicted_token_name}")
        lines.append("=" * 80)
        
        return "\n".join(lines)
