"""
Dict-based implementation of TinyTransformer inference.

This implementation uses dictionaries to structure data, making it easier
to understand the relationships between different components.

Advantages:
- Named access to components
- Clear data relationships
- Easy to inspect intermediate states
- Good for debugging

Disadvantages:
- More memory overhead
- Slower than arrays for computation
- Requires careful key management
"""

import json
import math
from typing import Dict, List, Any, Optional


def load_model_as_dict(json_path: str) -> Dict[str, Any]:
    """Load model weights into structured dictionaries."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Restructure into more intuitive format
    return {
        'embedding': {
            'weights': data['nodes']['embedding']['weights'],
            'vocab_size': data['nodes']['embedding']['input_dim'],
            'embed_dim': data['nodes']['embedding']['output_dim'],
        },
        'attention': {
            'weights': data['nodes']['attention']['weights'],
            'embed_dim': data['nodes']['attention']['input_dim'],
            'num_heads': data['nodes']['attention']['num_heads'],
        },
        'output': {
            'weights': data['nodes']['output']['weights'],
            'embed_dim': data['nodes']['output']['input_dim'],
            'vocab_size': data['nodes']['output']['output_dim'],
        }
    }


class TinyTransformerDictBased:
    """
    Dict-based implementation of TinyTransformer.
    
    Uses dictionaries with named keys for clarity and ease of understanding.
    Each intermediate state is stored as a dict with meaningful names.
    
    Example:
        >>> model = TinyTransformerDictBased.from_json('tinytf/tiny_model_gguf.json')
        >>> result = model.forward([0, 1, 2], trace=True)
        >>> print(result['predictions'])
    """
    
    def __init__(self, model_dict: Dict[str, Any]):
        """Initialize from structured model dictionary."""
        self.model = model_dict
        
        # Split attention weights
        attn_weights = model_dict['attention']['weights']
        n_rows = len(attn_weights) // 3
        
        self.model['attention']['query_weights'] = attn_weights[:n_rows]
        self.model['attention']['key_weights'] = attn_weights[n_rows:2*n_rows]
        self.model['attention']['value_weights'] = attn_weights[2*n_rows:]
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TinyTransformerDictBased':
        """Load model from JSON file."""
        model_dict = load_model_as_dict(json_path)
        return cls(model_dict)
    
    def _dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute dot product."""
        return sum(a * b for a, b in zip(vec1, vec2))
    
    def _matrix_vector_multiply(
        self, 
        matrix: List[List[float]], 
        vector: List[float]
    ) -> List[float]:
        """Multiply matrix by vector."""
        return [self._dot_product(row, vector) for row in matrix]
    
    def _softmax(self, values: List[float]) -> List[float]:
        """Apply softmax."""
        max_val = max(values)
        exp_values = [math.exp(v - max_val) for v in values]
        sum_exp = sum(exp_values)
        return [v / sum_exp for v in exp_values]
    
    def embed_tokens(
        self, 
        tokens: List[int], 
        trace: bool = False
    ) -> Dict[str, Any]:
        """
        Embed tokens into vectors.
        
        Returns a dictionary with:
        - 'token_ids': Input token IDs
        - 'token_names': Human-readable token names
        - 'embeddings': List of embedding vectors
        - 'embedding_indices': Which rows were selected
        """
        embedding_weights = self.model['embedding']['weights']
        
        result = {
            'token_ids': tokens,
            'token_names': [f'token_{t}' for t in tokens],
            'embeddings': [],
            'embedding_indices': tokens,
        }
        
        for i, token_id in enumerate(tokens):
            embedding = embedding_weights[token_id]
            result['embeddings'].append(embedding)
            
            if trace:
                print(f"\nToken {i}:")
                print(f"  ID: {token_id}")
                print(f"  Name: token_{token_id}")
                print(f"  Embedding (row {token_id}): {[f'{v:.4f}' for v in embedding]}")
        
        return result
    
    def attention(
        self, 
        embedding_result: Dict[str, Any], 
        trace: bool = False
    ) -> Dict[str, Any]:
        """
        Apply self-attention.
        
        Returns a dictionary with:
        - 'queries': Query vectors for each position
        - 'keys': Key vectors for each position
        - 'values': Value vectors for each position
        - 'attention_scores': Raw attention scores [seq_len, seq_len]
        - 'attention_weights': Softmaxed attention weights [seq_len, seq_len]
        - 'outputs': Attended output vectors
        """
        embeddings = embedding_result['embeddings']
        
        q_weights = self.model['attention']['query_weights']
        k_weights = self.model['attention']['key_weights']
        v_weights = self.model['attention']['value_weights']
        
        # Project to Q, K, V
        queries = [self._matrix_vector_multiply(q_weights, emb) for emb in embeddings]
        keys = [self._matrix_vector_multiply(k_weights, emb) for emb in embeddings]
        values = [self._matrix_vector_multiply(v_weights, emb) for emb in embeddings]
        
        if trace:
            print("\n=== Projections ===")
            print(f"Queries: {len(queries)} vectors")
            print(f"Keys: {len(keys)} vectors")
            print(f"Values: {len(values)} vectors")
        
        # Compute attention
        embed_dim = len(queries[0])
        scale = 1.0 / math.sqrt(embed_dim)
        
        attention_scores = []
        attention_weights = []
        outputs = []
        
        for i, query in enumerate(queries):
            # Scores with all keys
            scores = [self._dot_product(query, key) * scale for key in keys]
            attention_scores.append(scores)
            
            # Apply softmax
            weights = self._softmax(scores)
            attention_weights.append(weights)
            
            # Weighted sum of values
            output = [0.0] * embed_dim
            for j, weight in enumerate(weights):
                for k in range(embed_dim):
                    output[k] += weight * values[j][k]
            
            outputs.append(output)
            
            if trace:
                print(f"\nPosition {i} (token_{embedding_result['token_ids'][i]}):")
                print(f"  Attention to each position: {[f'{w:.4f}' for w in weights]}")
                print(f"  Output: {[f'{v:.4f}' for v in output]}")
        
        return {
            'queries': queries,
            'keys': keys,
            'values': values,
            'attention_scores': attention_scores,
            'attention_weights': attention_weights,
            'outputs': outputs,
        }
    
    def output_projection(
        self, 
        attention_result: Dict[str, Any], 
        trace: bool = False
    ) -> Dict[str, Any]:
        """
        Project to vocabulary space.
        
        Returns a dictionary with:
        - 'logits': Logits for each position [seq_len, vocab_size]
        - 'predictions': Predicted token ID for each position
        - 'prediction_names': Predicted token names
        """
        hidden_states = attention_result['outputs']
        output_weights = self.model['output']['weights']
        
        logits = []
        predictions = []
        prediction_names = []
        
        for i, hidden in enumerate(hidden_states):
            # Project to vocab size
            output_logits = self._matrix_vector_multiply(output_weights, hidden)
            logits.append(output_logits)
            
            # Get prediction
            pred_id = output_logits.index(max(output_logits))
            predictions.append(pred_id)
            prediction_names.append(f'token_{pred_id}')
            
            if trace:
                print(f"\nPosition {i}:")
                print(f"  Logits: {[f'{v:.4f}' for v in output_logits]}")
                print(f"  Prediction: token_{pred_id} (logit={output_logits[pred_id]:.4f})")
        
        return {
            'logits': logits,
            'predictions': predictions,
            'prediction_names': prediction_names,
        }
    
    def forward(
        self, 
        token_ids: List[int], 
        trace: bool = False
    ) -> Dict[str, Any]:
        """
        Full forward pass, returning all intermediate states.
        
        Returns a comprehensive dictionary with:
        - 'input': Input information
        - 'embedding': Embedding results
        - 'attention': Attention results
        - 'output': Output projection results
        - 'predictions': Final predictions
        """
        if trace:
            print("=" * 80)
            print("DICT-BASED TINY TRANSFORMER")
            print("=" * 80)
            print(f"\nInput: {token_ids}")
            print(f"Tokens: {[f'token_{t}' for t in token_ids]}")
        
        # Step 1: Embedding
        if trace:
            print("\n" + "=" * 80)
            print("STEP 1: Embedding")
            print("=" * 80)
        
        embedding_result = self.embed_tokens(token_ids, trace=trace)
        
        # Step 2: Attention
        if trace:
            print("\n" + "=" * 80)
            print("STEP 2: Self-Attention")
            print("=" * 80)
        
        attention_result = self.attention(embedding_result, trace=trace)
        
        # Step 3: Output
        if trace:
            print("\n" + "=" * 80)
            print("STEP 3: Output Projection")
            print("=" * 80)
        
        output_result = self.output_projection(attention_result, trace=trace)
        
        if trace:
            print("\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"Predictions: {output_result['prediction_names']}")
        
        return {
            'input': {
                'token_ids': token_ids,
                'token_names': [f'token_{t}' for t in token_ids],
            },
            'embedding': embedding_result,
            'attention': attention_result,
            'output': output_result,
            'predictions': output_result['predictions'],
        }
    
    def inspect_computation(self, token_ids: List[int]) -> str:
        """
        Generate a detailed report of the computation.
        
        Returns a formatted string showing all intermediate values.
        """
        result = self.forward(token_ids, trace=False)
        
        lines = []
        lines.append("=" * 80)
        lines.append("TINY TRANSFORMER COMPUTATION TRACE")
        lines.append("=" * 80)
        lines.append("")
        
        # Input
        lines.append("INPUT:")
        for i, (tid, tname) in enumerate(zip(
            result['input']['token_ids'], 
            result['input']['token_names']
        )):
            lines.append(f"  Position {i}: {tname} (ID={tid})")
        lines.append("")
        
        # Embeddings
        lines.append("EMBEDDINGS:")
        for i, emb in enumerate(result['embedding']['embeddings']):
            lines.append(f"  Position {i}: [{', '.join(f'{v:.4f}' for v in emb)}]")
        lines.append("")
        
        # Attention weights
        lines.append("ATTENTION WEIGHTS (who attends to whom):")
        for i, weights in enumerate(result['attention']['attention_weights']):
            lines.append(f"  Position {i} attends to:")
            for j, w in enumerate(weights):
                lines.append(f"    Position {j}: {w:.4f}")
        lines.append("")
        
        # Outputs
        lines.append("PREDICTIONS:")
        for i, (pred_id, pred_name) in enumerate(zip(
            result['output']['predictions'],
            result['output']['prediction_names']
        )):
            logits = result['output']['logits'][i]
            lines.append(f"  Position {i}: {pred_name}")
            lines.append(f"    Logits: [{', '.join(f'{v:.4f}' for v in logits)}]")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
