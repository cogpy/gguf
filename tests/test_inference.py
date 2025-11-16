"""
Tests for inference implementations.
"""

import pytest
from pathlib import Path

from gguf_workbench.inference import (
    TinyTransformerListBased,
    TinyTransformerDictBased,
    TinyTransformerClassBased,
)
from gguf_workbench.inference.functional import (
    tiny_transformer_functional,
    load_model_weights_functional,
    create_inference_function,
)


# Test data
MODEL_PATH = Path(__file__).parent.parent / "tinytf" / "tiny_model_gguf.json"
TEST_TOKENS = [0, 1, 2]
TEST_TOKENS_LONG = [0, 1, 2, 3, 4]


class TestListBasedInference:
    """Test list-based implementation."""
    
    def test_load_model(self):
        """Test model loading."""
        model = TinyTransformerListBased.from_json(str(MODEL_PATH))
        assert model.vocab_size == 10
        assert model.embed_dim == 5
        assert model.num_heads == 1
        assert len(model.embedding_weights) == 10
        assert len(model.embedding_weights[0]) == 5
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = TinyTransformerListBased.from_json(str(MODEL_PATH))
        logits = model.forward(TEST_TOKENS, trace=False)
        
        assert len(logits) == len(TEST_TOKENS)
        for logit_vec in logits:
            assert len(logit_vec) == 10
    
    def test_predict_next_token(self):
        """Test next token prediction."""
        model = TinyTransformerListBased.from_json(str(MODEL_PATH))
        next_token = model.predict_next_token(TEST_TOKENS, trace=False)
        
        assert isinstance(next_token, int)
        assert 0 <= next_token < 10
    
    def test_generate(self):
        """Test sequence generation."""
        model = TinyTransformerListBased.from_json(str(MODEL_PATH))
        sequence = model.generate([0, 1], max_new_tokens=3, trace=False)
        
        assert len(sequence) == 2 + 3
        assert sequence[:2] == [0, 1]
        for token in sequence:
            assert 0 <= token < 10
    
    def test_embed_tokens(self):
        """Test token embedding."""
        model = TinyTransformerListBased.from_json(str(MODEL_PATH))
        embeddings = model.embed_tokens(TEST_TOKENS, trace=False)
        
        assert len(embeddings) == len(TEST_TOKENS)
        for emb in embeddings:
            assert len(emb) == 5


class TestDictBasedInference:
    """Test dict-based implementation."""
    
    def test_load_model(self):
        """Test model loading."""
        model = TinyTransformerDictBased.from_json(str(MODEL_PATH))
        assert 'embedding' in model.model
        assert 'attention' in model.model
        assert 'output' in model.model
    
    def test_forward_pass(self):
        """Test forward pass returns structured dict."""
        model = TinyTransformerDictBased.from_json(str(MODEL_PATH))
        result = model.forward(TEST_TOKENS, trace=False)
        
        assert 'input' in result
        assert 'embedding' in result
        assert 'attention' in result
        assert 'output' in result
        assert 'predictions' in result
        
        assert result['input']['token_ids'] == TEST_TOKENS
        assert len(result['predictions']) == len(TEST_TOKENS)
    
    def test_inspect_computation(self):
        """Test computation inspection."""
        model = TinyTransformerDictBased.from_json(str(MODEL_PATH))
        report = model.inspect_computation(TEST_TOKENS)
        
        assert isinstance(report, str)
        assert 'INPUT' in report
        assert 'EMBEDDINGS' in report
        assert 'ATTENTION' in report
        assert 'PREDICTIONS' in report


class TestClassBasedInference:
    """Test class-based implementation."""
    
    def test_load_model(self):
        """Test model loading."""
        model = TinyTransformerClassBased.from_json(str(MODEL_PATH))
        assert model.weights.vocab_size == 10
        assert model.weights.embed_dim == 5
        assert model.weights.num_heads == 1
    
    def test_forward_pass(self):
        """Test forward pass returns LogitPrediction objects."""
        model = TinyTransformerClassBased.from_json(str(MODEL_PATH))
        predictions = model.forward(TEST_TOKENS, trace=False)
        
        assert len(predictions) == len(TEST_TOKENS)
        
        for i, pred in enumerate(predictions):
            assert pred.position == i
            assert len(pred.logits) == 10
            assert isinstance(pred.predicted_token_id, int)
            assert 0 <= pred.predicted_token_id < 10
            assert pred.predicted_token_name.startswith('token_')
    
    def test_get_attention_matrix(self):
        """Test attention matrix extraction."""
        model = TinyTransformerClassBased.from_json(str(MODEL_PATH))
        attn_matrix = model.get_attention_matrix(TEST_TOKENS)
        
        assert len(attn_matrix) == len(TEST_TOKENS)
        
        for row in attn_matrix:
            assert len(row) == len(TEST_TOKENS)
            # Attention weights should sum to ~1.0
            assert abs(sum(row) - 1.0) < 0.001
    
    def test_explain_prediction(self):
        """Test prediction explanation."""
        model = TinyTransformerClassBased.from_json(str(MODEL_PATH))
        explanation = model.explain_prediction(TEST_TOKENS, position=1)
        
        assert isinstance(explanation, str)
        assert 'PREDICTION EXPLANATION' in explanation
        assert 'Position 1' in explanation


class TestFunctionalInference:
    """Test functional implementation."""
    
    def test_load_weights(self):
        """Test weight loading."""
        weights = load_model_weights_functional(str(MODEL_PATH))
        
        assert len(weights) == 6
        embedding_weights, query_weights, key_weights, value_weights, output_weights, embed_dim = weights
        
        assert len(embedding_weights) == 10
        assert len(embedding_weights[0]) == 5
        assert embed_dim == 5
    
    def test_forward_pass(self):
        """Test functional forward pass."""
        weights = load_model_weights_functional(str(MODEL_PATH))
        predictions = tiny_transformer_functional(
            TEST_TOKENS,
            *weights,
            return_details=False
        )
        
        assert len(predictions) == len(TEST_TOKENS)
        for pred in predictions:
            assert 0 <= pred < 10
    
    def test_forward_with_details(self):
        """Test forward pass with detailed outputs."""
        weights = load_model_weights_functional(str(MODEL_PATH))
        result = tiny_transformer_functional(
            TEST_TOKENS,
            *weights,
            return_details=True
        )
        
        assert 'token_ids' in result
        assert 'embeddings' in result
        assert 'queries' in result
        assert 'keys' in result
        assert 'values' in result
        assert 'attention_weights' in result
        assert 'logits' in result
        assert 'predictions' in result
    
    def test_create_inference_function(self):
        """Test creating specialized inference function."""
        inference_fn = create_inference_function(str(MODEL_PATH))
        predictions = inference_fn(TEST_TOKENS)
        
        assert len(predictions) == len(TEST_TOKENS)
        for pred in predictions:
            assert 0 <= pred < 10


class TestConsistencyAcrossImplementations:
    """Test that all implementations produce identical results."""
    
    def test_all_implementations_match(self):
        """Verify all implementations produce same predictions."""
        # List-based
        model_list = TinyTransformerListBased.from_json(str(MODEL_PATH))
        logits_list = model_list.forward(TEST_TOKENS, trace=False)
        preds_list = [logit_vec.index(max(logit_vec)) for logit_vec in logits_list]
        
        # Dict-based
        model_dict = TinyTransformerDictBased.from_json(str(MODEL_PATH))
        result_dict = model_dict.forward(TEST_TOKENS, trace=False)
        preds_dict = result_dict['predictions']
        
        # Class-based
        model_class = TinyTransformerClassBased.from_json(str(MODEL_PATH))
        result_class = model_class.forward(TEST_TOKENS, trace=False)
        preds_class = [pred.predicted_token_id for pred in result_class]
        
        # Functional
        weights = load_model_weights_functional(str(MODEL_PATH))
        preds_func = tiny_transformer_functional(
            TEST_TOKENS,
            *weights,
            return_details=False
        )
        
        # All should match
        assert preds_list == preds_dict
        assert preds_dict == preds_class
        assert preds_class == preds_func
    
    def test_consistency_different_inputs(self):
        """Test consistency across different input sequences."""
        test_cases = [
            [0],
            [0, 1],
            [0, 1, 2],
            [5, 6, 7],
            [0, 1, 2, 3, 4],
        ]
        
        for tokens in test_cases:
            # Get predictions from each implementation
            model_list = TinyTransformerListBased.from_json(str(MODEL_PATH))
            logits_list = model_list.forward(tokens, trace=False)
            preds_list = [logit_vec.index(max(logit_vec)) for logit_vec in logits_list]
            
            model_dict = TinyTransformerDictBased.from_json(str(MODEL_PATH))
            result_dict = model_dict.forward(tokens, trace=False)
            preds_dict = result_dict['predictions']
            
            model_class = TinyTransformerClassBased.from_json(str(MODEL_PATH))
            result_class = model_class.forward(tokens, trace=False)
            preds_class = [pred.predicted_token_id for pred in result_class]
            
            weights = load_model_weights_functional(str(MODEL_PATH))
            preds_func = tiny_transformer_functional(
                tokens,
                *weights,
                return_details=False
            )
            
            # Verify all match
            assert preds_list == preds_dict == preds_class == preds_func, \
                f"Implementations differ for input {tokens}"


class TestEdgeCases:
    """Test edge cases and special inputs."""
    
    def test_single_token(self):
        """Test inference with single token."""
        model = TinyTransformerListBased.from_json(str(MODEL_PATH))
        logits = model.forward([5], trace=False)
        
        assert len(logits) == 1
        assert len(logits[0]) == 10
    
    def test_all_tokens(self):
        """Test inference with all vocabulary tokens."""
        model = TinyTransformerListBased.from_json(str(MODEL_PATH))
        all_tokens = list(range(10))
        logits = model.forward(all_tokens, trace=False)
        
        assert len(logits) == 10
    
    def test_repeated_tokens(self):
        """Test inference with repeated tokens."""
        model = TinyTransformerListBased.from_json(str(MODEL_PATH))
        logits = model.forward([0, 0, 0], trace=False)
        
        assert len(logits) == 3
        # Predictions may differ due to positional attention
