"""
Tests for conversation data analysis module.
"""

import json
import pytest
from pathlib import Path
from gguf_workbench.analysis import (
    ConversationPair,
    ConversationDataset,
    ConversationAnalyzer,
    AnalysisResult,
)


class TestConversationPair:
    """Test ConversationPair class."""
    
    def test_create_pair(self):
        """Test creating a conversation pair."""
        pair = ConversationPair(
            query="What is AI?",
            response="AI is artificial intelligence.",
            metadata={'temperature': 0.7},
            position=0
        )
        
        assert pair.query == "What is AI?"
        assert pair.response == "AI is artificial intelligence."
        assert pair.metadata == {'temperature': 0.7}
        assert pair.position == 0
    
    def test_pair_to_dict(self):
        """Test converting pair to dictionary."""
        pair = ConversationPair(
            query="Hello",
            response="Hi there!",
            position=1
        )
        
        d = pair.to_dict()
        assert d['query'] == "Hello"
        assert d['response'] == "Hi there!"
        assert d['position'] == 1
        assert 'metadata' in d
    
    def test_pair_from_dict(self):
        """Test creating pair from dictionary."""
        data = {
            'query': 'Test query',
            'response': 'Test response',
            'metadata': {'key': 'value'},
            'position': 2
        }
        
        pair = ConversationPair.from_dict(data)
        assert pair.query == 'Test query'
        assert pair.response == 'Test response'
        assert pair.metadata == {'key': 'value'}
        assert pair.position == 2


class TestConversationDataset:
    """Test ConversationDataset class."""
    
    def test_create_dataset(self):
        """Test creating a conversation dataset."""
        pairs = [
            ConversationPair("Q1", "R1"),
            ConversationPair("Q2", "R2"),
        ]
        
        dataset = ConversationDataset(
            pairs=pairs,
            is_sequential=True,
            global_metadata={'model': 'test'}
        )
        
        assert len(dataset) == 2
        assert dataset.is_sequential is True
        assert dataset.global_metadata == {'model': 'test'}
    
    def test_dataset_indexing(self):
        """Test dataset indexing."""
        pairs = [
            ConversationPair("Q1", "R1"),
            ConversationPair("Q2", "R2"),
        ]
        
        dataset = ConversationDataset(pairs)
        assert dataset[0].query == "Q1"
        assert dataset[1].response == "R2"
    
    def test_dataset_json_roundtrip(self, tmp_path):
        """Test saving and loading dataset from JSON."""
        pairs = [
            ConversationPair("Q1", "R1", position=0),
            ConversationPair("Q2", "R2", position=1),
        ]
        
        dataset = ConversationDataset(
            pairs=pairs,
            is_sequential=True,
            global_metadata={'test': 'data'}
        )
        
        filepath = tmp_path / "test_dataset.json"
        dataset.to_json_file(str(filepath))
        
        loaded = ConversationDataset.from_json_file(str(filepath))
        assert len(loaded) == 2
        assert loaded.is_sequential is True
        assert loaded.global_metadata == {'test': 'data'}
        assert loaded[0].query == "Q1"


class TestConversationAnalyzer:
    """Test ConversationAnalyzer class."""
    
    @pytest.fixture
    def simple_dataset(self):
        """Create a simple test dataset."""
        pairs = [
            ConversationPair("What is AI?", "AI is artificial intelligence."),
            ConversationPair("Tell me more", "AI enables machines to learn."),
            ConversationPair("How does it work?", "Through algorithms and data."),
        ]
        return ConversationDataset(pairs, is_sequential=False)
    
    @pytest.fixture
    def sequential_dataset(self):
        """Create a sequential test dataset."""
        pairs = [
            ConversationPair("Hello", "Hi there!", position=0),
            ConversationPair("How are you?", "I'm doing well, thanks!", position=1),
            ConversationPair("What's your name?", "I'm an AI assistant.", position=2),
        ]
        return ConversationDataset(pairs, is_sequential=True)
    
    @pytest.fixture
    def metadata_dataset(self):
        """Create dataset with metadata."""
        pairs = [
            ConversationPair(
                "Tell me a joke",
                "Why did the chicken cross the road?",
                metadata={'temperature': 0.9},
                position=0
            ),
        ]
        return ConversationDataset(
            pairs,
            is_sequential=True,
            global_metadata={'model': 'test-model'}
        )
    
    def test_create_analyzer(self):
        """Test creating analyzer."""
        analyzer = ConversationAnalyzer(
            model_vocab_size=1000,
            embedding_dim=128
        )
        
        assert analyzer.model_vocab_size == 1000
        assert analyzer.embedding_dim == 128
    
    def test_analyze_simple_dataset(self, simple_dataset):
        """Test analyzing a simple dataset."""
        analyzer = ConversationAnalyzer()
        result = analyzer.analyze(simple_dataset)
        
        assert isinstance(result, AnalysisResult)
        assert result.basic_stats['total_pairs'] == 3
        assert result.basic_stats['is_sequential'] is False
        assert 'unique_tokens' in result.token_analysis
        assert result.token_analysis['unique_tokens'] > 0
    
    def test_analyze_sequential_dataset(self, sequential_dataset):
        """Test analyzing sequential dataset."""
        analyzer = ConversationAnalyzer()
        result = analyzer.analyze(sequential_dataset)
        
        assert result.basic_stats['is_sequential'] is True
        assert 'sequential_insights' in result.to_dict()
        assert result.sequential_insights  # Should not be empty
    
    def test_analyze_metadata_dataset(self, metadata_dataset):
        """Test analyzing dataset with metadata."""
        analyzer = ConversationAnalyzer()
        result = analyzer.analyze(metadata_dataset)
        
        assert result.basic_stats['has_metadata'] is True
        assert result.basic_stats['pairs_with_metadata'] >= 1
        assert result.metadata_insights  # Should not be empty
    
    def test_basic_statistics(self, simple_dataset):
        """Test basic statistics computation."""
        analyzer = ConversationAnalyzer()
        result = analyzer.analyze(simple_dataset)
        
        stats = result.basic_stats
        assert stats['total_pairs'] == 3
        assert 'avg_query_length' in stats
        assert 'avg_response_length' in stats
        assert 'total_tokens' in stats
        assert stats['avg_query_length'] > 0
    
    def test_token_analysis(self, simple_dataset):
        """Test token analysis."""
        analyzer = ConversationAnalyzer(model_vocab_size=1000)
        result = analyzer.analyze(simple_dataset)
        
        token_info = result.token_analysis
        assert 'unique_tokens' in token_info
        assert 'total_token_instances' in token_info
        assert 'vocab_coverage' in token_info
        assert 'token_diversity' in token_info
        assert token_info['vocab_coverage'] is not None
    
    def test_unordered_insights(self, simple_dataset):
        """Test unordered dataset insights."""
        analyzer = ConversationAnalyzer()
        result = analyzer.analyze(simple_dataset)
        
        insights = result.unordered_insights
        assert 'learnable_aspects' in insights
        assert 'limitations' in insights
        assert len(insights['learnable_aspects']) > 0
        assert len(insights['limitations']) > 0
        
        # Check structure
        aspect = insights['learnable_aspects'][0]
        assert 'aspect' in aspect
        assert 'description' in aspect
        assert 'confidence' in aspect
    
    def test_sequential_insights(self, sequential_dataset):
        """Test sequential dataset insights."""
        analyzer = ConversationAnalyzer()
        result = analyzer.analyze(sequential_dataset)
        
        insights = result.sequential_insights
        assert 'additional_learnable_aspects' in insights
        assert 'enhanced_capabilities' in insights
        assert len(insights['additional_learnable_aspects']) > 0
    
    def test_metadata_insights(self, metadata_dataset):
        """Test metadata insights."""
        analyzer = ConversationAnalyzer()
        result = analyzer.analyze(metadata_dataset)
        
        insights = result.metadata_insights
        assert 'metadata_types' in insights
        assert 'learnable_aspects' in insights
        assert 'temperature' in insights['metadata_types']
    
    def test_inference_limits(self, simple_dataset):
        """Test inference limits analysis."""
        analyzer = ConversationAnalyzer(
            model_vocab_size=1000,
            embedding_dim=128
        )
        result = analyzer.analyze(simple_dataset)
        
        limits = result.inference_limits
        assert 'theoretical_limits' in limits
        assert 'practical_constraints' in limits
        assert 'what_is_possible' in limits
        assert 'what_is_impossible' in limits
        assert len(limits['theoretical_limits']) > 0
        assert len(limits['what_is_possible']) > 0


class TestAnalysisResult:
    """Test AnalysisResult class."""
    
    @pytest.fixture
    def sample_result(self, tmp_path):
        """Create a sample analysis result."""
        pairs = [
            ConversationPair("Test query", "Test response"),
        ]
        dataset = ConversationDataset(pairs)
        analyzer = ConversationAnalyzer()
        return analyzer.analyze(dataset)
    
    def test_result_summary(self, sample_result):
        """Test generating summary."""
        summary = sample_result.summary()
        
        assert isinstance(summary, str)
        assert 'CONVERSATION DATA ANALYSIS SUMMARY' in summary
        assert 'DATASET STATISTICS' in summary
        assert 'WHAT CAN WE LEARN' in summary
        assert 'FUNDAMENTAL LIMITS' in summary
    
    def test_result_to_dict(self, sample_result):
        """Test converting result to dictionary."""
        d = sample_result.to_dict()
        
        assert 'basic_stats' in d
        assert 'token_analysis' in d
        assert 'unordered_insights' in d
        assert 'inference_limits' in d
    
    def test_result_to_json(self, sample_result, tmp_path):
        """Test saving result to JSON file."""
        filepath = tmp_path / "result.json"
        sample_result.to_json_file(str(filepath))
        
        assert filepath.exists()
        
        with open(filepath) as f:
            data = json.load(f)
        
        assert 'basic_stats' in data
        assert 'token_analysis' in data


class TestTokenCooccurrence:
    """Test token co-occurrence analysis."""
    
    def test_cooccurrence_computation(self):
        """Test computing token co-occurrence."""
        pairs = [
            ConversationPair("hello world", "hi there"),
            ConversationPair("hello again", "hi again"),
        ]
        dataset = ConversationDataset(pairs)
        analyzer = ConversationAnalyzer()
        
        result = analyzer.analyze(dataset)
        cooccurrence = result.unordered_insights.get('token_cooccurrence_sample', [])
        
        assert isinstance(cooccurrence, list)
        if cooccurrence:
            assert 'query_token' in cooccurrence[0]
            assert 'response_token' in cooccurrence[0]
            assert 'count' in cooccurrence[0]


class TestSequentialStatistics:
    """Test sequential statistics computation."""
    
    def test_sequential_stats(self):
        """Test computing sequential statistics."""
        pairs = [
            ConversationPair("What is AI?", "AI is intelligence.", position=0),
            ConversationPair("Tell me more about intelligence", "Intelligence involves learning.", position=1),
        ]
        dataset = ConversationDataset(pairs, is_sequential=True)
        analyzer = ConversationAnalyzer()
        
        result = analyzer.analyze(dataset)
        seq_stats = result.sequential_insights.get('sequential_stats', {})
        
        assert 'conversation_length' in seq_stats
        assert seq_stats['conversation_length'] == 2


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataset(self):
        """Test analyzing empty dataset."""
        dataset = ConversationDataset([])
        analyzer = ConversationAnalyzer()
        
        result = analyzer.analyze(dataset)
        assert result.basic_stats['total_pairs'] == 0
    
    def test_single_pair(self):
        """Test analyzing single conversation pair."""
        pairs = [ConversationPair("Q", "R")]
        dataset = ConversationDataset(pairs)
        analyzer = ConversationAnalyzer()
        
        result = analyzer.analyze(dataset)
        assert result.basic_stats['total_pairs'] == 1
    
    def test_empty_strings(self):
        """Test handling empty strings."""
        pairs = [ConversationPair("", "")]
        dataset = ConversationDataset(pairs)
        analyzer = ConversationAnalyzer()
        
        result = analyzer.analyze(dataset)
        assert result.basic_stats['total_pairs'] == 1
        assert result.token_analysis['unique_tokens'] == 0
    
    def test_no_model_info(self):
        """Test analyzer without model information."""
        analyzer = ConversationAnalyzer()
        pairs = [ConversationPair("Q", "R")]
        dataset = ConversationDataset(pairs)
        
        result = analyzer.analyze(dataset)
        # Should still work, just won't have vocab coverage
        assert result.token_analysis['vocab_coverage'] is None


class TestIntegration:
    """Integration tests for the analysis module."""
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow from dataset creation to analysis."""
        # Create dataset
        pairs = [
            ConversationPair(
                "What is Python?",
                "Python is a programming language.",
                metadata={'temperature': 0.7},
                position=0
            ),
            ConversationPair(
                "What is it used for?",
                "It's used for web development, data science, and more.",
                metadata={'temperature': 0.7},
                position=1
            ),
        ]
        
        dataset = ConversationDataset(
            pairs=pairs,
            is_sequential=True,
            global_metadata={'model': 'test'}
        )
        
        # Save dataset
        dataset_path = tmp_path / "dataset.json"
        dataset.to_json_file(str(dataset_path))
        
        # Load dataset
        loaded_dataset = ConversationDataset.from_json_file(str(dataset_path))
        
        # Analyze
        analyzer = ConversationAnalyzer(
            model_vocab_size=5000,
            embedding_dim=256
        )
        result = analyzer.analyze(loaded_dataset)
        
        # Verify results
        assert result.basic_stats['total_pairs'] == 2
        assert result.basic_stats['is_sequential'] is True
        assert result.sequential_insights is not None
        assert result.metadata_insights is not None
        
        # Save results
        result_path = tmp_path / "result.json"
        result.to_json_file(str(result_path))
        
        # Verify saved file
        assert result_path.exists()
        with open(result_path) as f:
            data = json.load(f)
            assert data['basic_stats']['total_pairs'] == 2
    
    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        # Create 100 pairs
        pairs = [
            ConversationPair(
                f"Query {i}",
                f"Response {i}",
                position=i
            )
            for i in range(100)
        ]
        
        dataset = ConversationDataset(pairs, is_sequential=True)
        analyzer = ConversationAnalyzer()
        
        result = analyzer.analyze(dataset)
        assert result.basic_stats['total_pairs'] == 100
        assert len(result.sequential_insights['sequential_stats']) > 0
