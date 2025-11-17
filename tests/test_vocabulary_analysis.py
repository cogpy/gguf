"""
Tests for vocabulary and layer activation analysis module.
"""

import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf_workbench.vocabulary_analysis import (
    VocabularyAnalyzer,
    load_conversations_from_json
)


class TestVocabularyAnalyzer:
    """Test VocabularyAnalyzer class."""
    
    def test_create_analyzer(self):
        """Test creating a vocabulary analyzer."""
        analyzer = VocabularyAnalyzer(
            total_vocab_size=50000,
            model_layers=12,
            embedding_dim=768,
            architecture="gpt2"
        )
        
        assert analyzer.total_vocab_size == 50000
        assert analyzer.model_layers == 12
        assert analyzer.embedding_dim == 768
        assert analyzer.architecture == "gpt2"
    
    def test_enumerate_vocabulary_basic(self):
        """Test basic vocabulary enumeration."""
        conversations = [
            {"query": "hello world", "response": "hi there"},
            {"query": "test query", "response": "test response"},
        ]
        
        analyzer = VocabularyAnalyzer()
        result = analyzer.enumerate_vocabulary(conversations)
        
        assert 'total_unique_words' in result
        assert 'total_word_instances' in result
        assert 'word_counts' in result
        assert result['total_unique_words'] > 0
        assert result['total_word_instances'] >= result['total_unique_words']
    
    def test_enumerate_vocabulary_with_model_metadata(self):
        """Test vocabulary enumeration with model metadata."""
        conversations = [
            {"query": "test", "response": "response", "model": "gpt-4"},
            {"query": "another", "response": "answer", "model": "gpt-3.5"},
        ]
        
        analyzer = VocabularyAnalyzer()
        result = analyzer.enumerate_vocabulary(conversations, include_metadata=True)
        
        assert 'models_found' in result
        assert len(result['models_found']) == 2
        assert 'gpt-4' in result['models_found']
        assert 'gpt-3.5' in result['models_found']
        
        assert 'per_model_vocabulary' in result
        assert 'gpt-4' in result['per_model_vocabulary']
        assert 'gpt-3.5' in result['per_model_vocabulary']
    
    def test_vocabulary_coverage_metrics(self):
        """Test vocabulary coverage calculation."""
        conversations = [
            {"query": "hello world", "response": "hi there"},
        ]
        
        analyzer = VocabularyAnalyzer(total_vocab_size=1000)
        result = analyzer.enumerate_vocabulary(conversations)
        
        assert 'vocabulary_coverage' in result
        coverage = result['vocabulary_coverage']
        
        assert 'expressed_vocab_size' in coverage
        assert 'total_vocab_size' in coverage
        assert 'coverage_ratio' in coverage
        assert 'coverage_percentage' in coverage
        assert 'unexpressed_vocab_size' in coverage
        
        assert coverage['total_vocab_size'] == 1000
        assert coverage['expressed_vocab_size'] <= 1000
        assert 0 <= coverage['coverage_ratio'] <= 1
        assert 0 <= coverage['coverage_percentage'] <= 100
    
    def test_tokenize_helper(self):
        """Test the tokenization helper method."""
        analyzer = VocabularyAnalyzer()
        
        text = "Hello, world! This is a test."
        tokens = analyzer._tokenize(text)
        
        assert isinstance(tokens, list)
        assert 'hello' in tokens
        assert 'world' in tokens
        assert 'test' in tokens
        # Punctuation should be excluded
        assert ',' not in tokens
        assert '!' not in tokens
    
    def test_analyze_layer_activations(self):
        """Test layer activation analysis."""
        conversations = [
            {"query": "What is AI?", "response": "AI is artificial intelligence."},
            {"query": "How does it work?", "response": "It uses algorithms and data."},
        ]
        
        analyzer = VocabularyAnalyzer(
            total_vocab_size=50000,
            model_layers=24,
            embedding_dim=768
        )
        
        vocab_stats = analyzer.enumerate_vocabulary(conversations)
        layer_insights = analyzer.analyze_layer_activations(vocab_stats, conversations)
        
        assert 'description' in layer_insights
        assert 'layer_activation_estimates' in layer_insights
        assert 'attention_head_insights' in layer_insights
        assert 'embedding_layer_analysis' in layer_insights
        assert 'output_layer_analysis' in layer_insights
        
        # Check embedding analysis
        embedding = layer_insights['embedding_layer_analysis']
        assert 'active_embeddings' in embedding
        assert 'interpretation' in embedding
        
        # Check layer estimates
        estimates = layer_insights['layer_activation_estimates']
        assert len(estimates) > 0
        assert all('layer_range' in e for e in estimates)
        assert all('layer_type' in e for e in estimates)
        assert all('activation_level' in e for e in estimates)
    
    def test_analyze_echo_reservoir_interaction(self):
        """Test echo reservoir interaction analysis."""
        conversations = [
            {"query": "test query", "response": "test response"},
            {"query": "another test", "response": "another answer"},
        ]
        
        analyzer = VocabularyAnalyzer()
        vocab_stats = analyzer.enumerate_vocabulary(conversations)
        layer_insights = analyzer.analyze_layer_activations(vocab_stats, conversations)
        
        echo_analysis = analyzer.analyze_echo_reservoir_interaction(
            vocab_stats,
            layer_insights,
            conversations
        )
        
        assert 'description' in echo_analysis
        assert 'reservoir_dynamics' in echo_analysis
        assert 'feedback_loops' in echo_analysis
        assert 'persona_variables' in echo_analysis
        assert 'temporal_patterns' in echo_analysis
        assert 'recommendations' in echo_analysis
        
        # Check feedback loops
        feedback = echo_analysis['feedback_loops']
        assert 'echo_ratio' in feedback
        assert 'feedback_strength' in feedback
        assert 0 <= feedback['echo_ratio'] <= 1
        
        # Check persona variables
        persona = echo_analysis['persona_variables']
        assert 'persona_dimensions_detected' in persona
        assert 'dominant_persona' in persona
        assert 'persona_diversity' in persona
    
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        conversations = [
            {"query": "What is machine learning?", "response": "Machine learning is a subset of AI."},
            {"query": "How does it learn?", "response": "It learns from data and patterns."},
        ]
        
        analyzer = VocabularyAnalyzer(
            total_vocab_size=50000,
            model_layers=12,
            embedding_dim=768,
            architecture="transformer"
        )
        
        report = analyzer.generate_comprehensive_report(conversations)
        
        assert 'vocabulary_analysis' in report
        assert 'layer_activation_analysis' in report
        assert 'echo_reservoir_analysis' in report
        assert 'model_configuration' in report
        assert 'summary' in report
        
        # Check model configuration
        config = report['model_configuration']
        assert config['total_vocab_size'] == 50000
        assert config['model_layers'] == 12
        assert config['embedding_dim'] == 768
        assert config['architecture'] == "transformer"
        
        # Check summary is a string
        assert isinstance(report['summary'], str)
        assert len(report['summary']) > 0
    
    def test_empty_conversations(self):
        """Test handling of empty conversation list."""
        analyzer = VocabularyAnalyzer()
        result = analyzer.enumerate_vocabulary([])
        
        assert result['total_unique_words'] == 0
        assert result['total_word_instances'] == 0
        assert result['vocabulary_diversity'] == 0
    
    def test_conversations_without_model_metadata(self):
        """Test handling conversations without model metadata."""
        conversations = [
            {"query": "test", "response": "response"},
        ]
        
        analyzer = VocabularyAnalyzer()
        result = analyzer.enumerate_vocabulary(conversations, include_metadata=True)
        
        # Should still work, just with empty model list
        assert 'models_found' in result
        assert len(result['models_found']) == 0


class TestLoadConversationsFromJson:
    """Test conversation loading functions."""
    
    def test_load_from_json_list(self, tmp_path):
        """Test loading from JSON array format."""
        conversations = [
            {"query": "test1", "response": "response1"},
            {"query": "test2", "response": "response2", "model": "gpt-4"},
        ]
        
        json_file = tmp_path / "conversations.json"
        with open(json_file, 'w') as f:
            json.dump(conversations, f)
        
        loaded = load_conversations_from_json(str(json_file))
        
        assert len(loaded) == 2
        assert loaded[0]['query'] == "test1"
        assert loaded[1]['model'] == "gpt-4"
    
    def test_load_from_jsonl(self, tmp_path):
        """Test loading from JSONL format."""
        jsonl_file = tmp_path / "conversations.jsonl"
        
        with open(jsonl_file, 'w') as f:
            f.write(json.dumps({"query": "q1", "response": "r1"}) + '\n')
            f.write(json.dumps({"query": "q2", "response": "r2", "model": "gpt-3.5"}) + '\n')
        
        loaded = load_conversations_from_json(str(jsonl_file))
        
        assert len(loaded) == 2
        assert loaded[0]['query'] == "q1"
        assert loaded[1]['model'] == "gpt-3.5"
    
    def test_load_from_jsonl_messages_format(self, tmp_path):
        """Test loading from JSONL with messages format."""
        jsonl_file = tmp_path / "conversations.jsonl"
        
        with open(jsonl_file, 'w') as f:
            conv = {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"}
                ],
                "model": "gpt-4"
            }
            f.write(json.dumps(conv) + '\n')
        
        loaded = load_conversations_from_json(str(jsonl_file))
        
        assert len(loaded) == 1
        assert loaded[0]['query'] == "Hello"
        assert loaded[0]['response'] == "Hi there"
        assert loaded[0]['model'] == "gpt-4"
    
    def test_load_from_json_mapping_format(self, tmp_path):
        """Test loading from JSON mapping format (ChatGPT export style)."""
        data = {
            "title": "Test Conversation",
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello"]},
                        "metadata": {"model_slug": "gpt-4"}
                    }
                },
                "msg2": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Hi there"]}
                    }
                }
            }
        }
        
        json_file = tmp_path / "conversation.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        loaded = load_conversations_from_json(str(json_file))
        
        assert len(loaded) == 2
        # Should extract conversations from mapping
        assert any(conv.get('query') == "Hello" or conv.get('response') == "Hello" 
                  for conv in loaded)
    
    def test_load_empty_file(self, tmp_path):
        """Test loading from empty JSONL file."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")
        
        loaded = load_conversations_from_json(str(jsonl_file))
        
        assert len(loaded) == 0
    
    def test_load_with_invalid_lines(self, tmp_path):
        """Test loading JSONL with some invalid lines."""
        jsonl_file = tmp_path / "mixed.jsonl"
        
        with open(jsonl_file, 'w') as f:
            f.write(json.dumps({"query": "valid", "response": "response"}) + '\n')
            f.write("invalid json line\n")
            f.write(json.dumps({"query": "also valid", "response": "another"}) + '\n')
        
        loaded = load_conversations_from_json(str(jsonl_file))
        
        # Should skip invalid lines but load valid ones
        assert len(loaded) == 2
        assert loaded[0]['query'] == "valid"
        assert loaded[1]['query'] == "also valid"


class TestIntegration:
    """Integration tests for vocabulary analysis."""
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow from loading to report generation."""
        # Create test data
        conversations = [
            {
                "query": "What is a neural network?",
                "response": "A neural network is a computational model inspired by biological neurons.",
                "model": "gpt-4"
            },
            {
                "query": "How does it learn?",
                "response": "It learns by adjusting weights through backpropagation and gradient descent.",
                "model": "gpt-4"
            },
        ]
        
        # Save to file
        json_file = tmp_path / "test_conversations.json"
        with open(json_file, 'w') as f:
            json.dump(conversations, f)
        
        # Load conversations
        loaded = load_conversations_from_json(str(json_file))
        assert len(loaded) == 2
        
        # Analyze
        analyzer = VocabularyAnalyzer(
            total_vocab_size=50000,
            model_layers=24,
            embedding_dim=768,
            architecture="transformer"
        )
        
        report = analyzer.generate_comprehensive_report(loaded)
        
        # Verify report structure
        assert 'vocabulary_analysis' in report
        assert 'layer_activation_analysis' in report
        assert 'echo_reservoir_analysis' in report
        
        # Save report
        report_file = tmp_path / "analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        assert report_file.exists()
        
        # Verify report can be reloaded
        with open(report_file, 'r') as f:
            reloaded = json.load(f)
        
        assert reloaded['model_configuration']['architecture'] == "transformer"
    
    def test_vocabulary_diversity_calculation(self):
        """Test vocabulary diversity with known inputs."""
        # All unique words
        conversations = [
            {"query": "one two three", "response": "four five six"}
        ]
        
        analyzer = VocabularyAnalyzer()
        result = analyzer.enumerate_vocabulary(conversations)
        
        # All 6 words are unique
        assert result['total_unique_words'] == 6
        assert result['total_word_instances'] == 6
        assert result['vocabulary_diversity'] == 1.0
        
        # Repeated words
        conversations = [
            {"query": "the the the", "response": "the the the"}
        ]
        
        result = analyzer.enumerate_vocabulary(conversations)
        
        # Only 1 unique word, 6 instances
        assert result['total_unique_words'] == 1
        assert result['total_word_instances'] == 6
        assert abs(result['vocabulary_diversity'] - (1/6)) < 0.001
    
    def test_persona_detection(self):
        """Test detection of different persona dimensions."""
        # Technical conversation
        technical_convs = [
            {"query": "Explain the algorithm", 
             "response": "The algorithm uses a function to implement the model architecture"}
        ]
        
        analyzer = VocabularyAnalyzer()
        vocab_stats = analyzer.enumerate_vocabulary(technical_convs)
        layer_insights = analyzer.analyze_layer_activations(vocab_stats, technical_convs)
        echo_analysis = analyzer.analyze_echo_reservoir_interaction(
            vocab_stats, layer_insights, technical_convs
        )
        
        persona = echo_analysis['persona_variables']
        assert persona['persona_dimensions_detected']['technical_language'] > 0
        
        # Emotional conversation
        emotional_convs = [
            {"query": "How do you feel about this?",
             "response": "I think it's great and I love how it works"}
        ]
        
        vocab_stats = analyzer.enumerate_vocabulary(emotional_convs)
        layer_insights = analyzer.analyze_layer_activations(vocab_stats, emotional_convs)
        echo_analysis = analyzer.analyze_echo_reservoir_interaction(
            vocab_stats, layer_insights, emotional_convs
        )
        
        persona = echo_analysis['persona_variables']
        assert persona['persona_dimensions_detected']['emotional_language'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
