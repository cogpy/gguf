"""
Tests for Deep Tree Echo analysis tools.
"""

import pytest
import json
from pathlib import Path
import sys

# Add examples directory to path
examples_dir = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_dir))

from analyze_deep_tree_echo import CharacterModel, ConversationAnalyzer
from analyze_all_echo_data import DatasetAnalyzer, ComprehensiveEchoAnalyzer
from visualize_echo_insights import InsightsVisualizer


class TestCharacterModel:
    """Test CharacterModel class."""
    
    def test_create_character_model(self, tmp_path):
        """Test creating a character model."""
        model = CharacterModel("Test Character", str(tmp_path / "test.md"))
        assert model.name == "Test Character"
        assert len(model.persona_dimensions) == 0
        assert len(model.key_concepts) == 0
    
    def test_load_from_markdown(self, tmp_path):
        """Test loading character model from markdown."""
        # Create a test markdown file
        md_content = """
# Test Character

## Persona Dimensions

1. **Cognitive**: Analytical reasoning
2. **Adaptive**: Dynamic responses
3. **Recursive**: Multi-level processing

## Key Concepts

This character uses memory, feedback, and recursive processing.
"""
        md_file = tmp_path / "character.md"
        md_file.write_text(md_content)
        
        model = CharacterModel("Test", str(md_file))
        model.load_from_markdown()
        
        # Should extract persona dimensions
        assert len(model.persona_dimensions) == 3
        assert model.persona_dimensions[0] == ('Cognitive', 'Analytical reasoning')
        
        # Should extract headers as concepts
        assert 'Test Character' in model.key_concepts
        assert 'Persona Dimensions' in model.key_concepts
        
        # Should find architecture components
        arch_keywords = [comp['keyword'] for comp in model.architecture_components]
        assert 'memory' in arch_keywords
        assert 'feedback' in arch_keywords
        assert 'recursive' in arch_keywords


class TestConversationAnalyzer:
    """Test ConversationAnalyzer class."""
    
    def test_create_analyzer(self, tmp_path):
        """Test creating conversation analyzer."""
        conv_file = tmp_path / "conversation.jsonl"
        conv_file.write_text('{"title": "Test", "mapping": {}}')
        
        analyzer = ConversationAnalyzer(str(conv_file))
        assert analyzer.conversation_file == str(conv_file)
    
    def test_load_conversation(self, tmp_path):
        """Test loading conversation data."""
        # Create test conversation
        conv_data = {
            "title": "Test Conversation",
            "create_time": 1700000000,
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello"]},
                        "create_time": 1700000000
                    },
                    "parent": None,
                    "children": ["msg2"]
                },
                "msg2": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Hi there!"]},
                        "create_time": 1700000001
                    },
                    "parent": "msg1",
                    "children": []
                }
            }
        }
        
        conv_file = tmp_path / "conversation.jsonl"
        conv_file.write_text(json.dumps(conv_data))
        
        analyzer = ConversationAnalyzer(str(conv_file))
        analyzer.load_conversation()
        
        assert len(analyzer.messages) == 2
        assert analyzer.messages[0]['role'] == 'user'
        assert analyzer.messages[1]['role'] == 'assistant'
    
    def test_analyze_conversation_flow(self, tmp_path):
        """Test conversation flow analysis."""
        conv_data = {
            "title": "Flow Test",
            "create_time": 1700000000,
            "update_time": 1700001000,
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Question"]},
                    },
                    "parent": None,
                    "children": []
                },
                "msg2": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Answer"]},
                    },
                    "parent": None,
                    "children": []
                }
            }
        }
        
        conv_file = tmp_path / "conversation.jsonl"
        conv_file.write_text(json.dumps(conv_data))
        
        analyzer = ConversationAnalyzer(str(conv_file))
        analyzer.load_conversation()
        
        flow = analyzer.analyze_conversation_flow()
        
        assert flow['total_messages'] == 2
        assert 'user' in flow['role_distribution']
        assert 'assistant' in flow['role_distribution']
        assert flow['conversation_title'] == 'Flow Test'
    
    def test_analyze_echo_patterns(self, tmp_path):
        """Test Echo pattern analysis."""
        conv_data = {
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Using recursive adaptive processing"]},
                    },
                    "parent": None,
                    "children": []
                }
            }
        }
        
        conv_file = tmp_path / "conversation.jsonl"
        conv_file.write_text(json.dumps(conv_data))
        
        analyzer = ConversationAnalyzer(str(conv_file))
        analyzer.load_conversation()
        
        patterns = analyzer.analyze_echo_patterns()
        
        assert 'pattern_frequencies' in patterns
        assert patterns['pattern_frequencies']['recursive_language'] > 0
        assert patterns['pattern_frequencies']['adaptive_language'] > 0


class TestDatasetAnalyzer:
    """Test DatasetAnalyzer class."""
    
    def test_load_dataset(self, tmp_path):
        """Test loading JSONL dataset."""
        dataset_file = tmp_path / "dataset.jsonl"
        
        # Write test conversations
        convs = [
            {"messages": [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1 {{template}}"}
            ]},
            {"messages": [
                {"role": "user", "content": "Question 2"},
                {"role": "assistant", "content": "Answer 2"}
            ]}
        ]
        
        with open(dataset_file, 'w') as f:
            for conv in convs:
                f.write(json.dumps(conv) + '\n')
        
        analyzer = DatasetAnalyzer(str(dataset_file))
        analyzer.load_dataset()
        
        assert len(analyzer.conversations) == 2
    
    def test_analyze_training_patterns(self, tmp_path):
        """Test training pattern analysis."""
        dataset_file = tmp_path / "dataset.jsonl"
        
        convs = [
            {"messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response {{p_system.execute()}}"}
            ]}
        ]
        
        with open(dataset_file, 'w') as f:
            for conv in convs:
                f.write(json.dumps(conv) + '\n')
        
        analyzer = DatasetAnalyzer(str(dataset_file))
        analyzer.load_dataset()
        patterns = analyzer.analyze_training_patterns()
        
        assert patterns['total_conversations'] == 1
        assert patterns['user_message_count'] == 1
        assert patterns['assistant_message_count'] == 1
        assert 'template_usage' in patterns


class TestComprehensiveEchoAnalyzer:
    """Test ComprehensiveEchoAnalyzer class."""
    
    def test_create_analyzer(self, tmp_path):
        """Test creating comprehensive analyzer."""
        analyzer = ComprehensiveEchoAnalyzer(str(tmp_path))
        assert analyzer.base_dir == tmp_path
    
    def test_analyze_cognitive_architecture(self, tmp_path):
        """Test cognitive architecture analysis."""
        # Create test files
        conv_file = tmp_path / "deep_tree_echo_dan_conversation.jsonl"
        conv_data = {
            "mapping": {
                "msg1": {
                    "message": {
                        "content": {"parts": ["Using reservoir and echo state network"]},
                    }
                }
            }
        }
        conv_file.write_text(json.dumps(conv_data))
        
        analyzer = ComprehensiveEchoAnalyzer(str(tmp_path))
        analyzer.load_all_data()
        
        arch = analyzer.analyze_cognitive_architecture()
        
        assert 'reservoir' in arch
        assert 'echo_state_network' in arch
        assert arch['reservoir'] > 0
        assert arch['echo_state_network'] > 0


class TestInsightsVisualizer:
    """Test InsightsVisualizer class."""
    
    def test_load_analysis(self, tmp_path):
        """Test loading analysis data."""
        analysis_data = {
            "overview": {
                "data_sources": {
                    "conversation_loaded": True,
                    "training_data_loaded": True,
                    "character_models": ["Test Model"]
                }
            },
            "key_insights": ["Insight 1", "Insight 2"]
        }
        
        analysis_file = tmp_path / "analysis.json"
        analysis_file.write_text(json.dumps(analysis_data))
        
        visualizer = InsightsVisualizer(str(analysis_file))
        visualizer.load_analysis()
        
        assert visualizer.data is not None
        assert len(visualizer.data['key_insights']) == 2
    
    def test_generate_markdown_report(self, tmp_path):
        """Test markdown report generation."""
        analysis_data = {
            "overview": {
                "data_sources": {
                    "conversation_loaded": True,
                    "character_models": ["Test"]
                }
            },
            "key_insights": ["Test insight"],
            "cognitive_architecture": {
                "reservoir": 10,
                "memory_systems": 20
            }
        }
        
        analysis_file = tmp_path / "analysis.json"
        analysis_file.write_text(json.dumps(analysis_data))
        
        visualizer = InsightsVisualizer(str(analysis_file))
        visualizer.load_analysis()
        
        report = visualizer.generate_markdown_report()
        
        assert "# Deep Tree Echo Conversation Analysis Report" in report
        assert "Test insight" in report
        assert "Cognitive Architecture Components" in report


class TestIntegration:
    """Integration tests."""
    
    def test_full_analysis_workflow(self, tmp_path):
        """Test complete analysis workflow."""
        # Create minimal test data
        conv_data = {
            "title": "Integration Test",
            "create_time": 1700000000,
            "update_time": 1700001000,
            "mapping": {
                "msg1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Test question"]},
                    },
                    "parent": None,
                    "children": []
                },
                "msg2": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Test response using reservoir"]},
                    },
                    "parent": None,
                    "children": []
                }
            }
        }
        
        conv_file = tmp_path / "deep_tree_echo_dan_conversation.jsonl"
        conv_file.write_text(json.dumps(conv_data))
        
        # Create dataset
        dataset_file = tmp_path / "training_dataset_fixed.jsonl"
        dataset_file.write_text(json.dumps({
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"}
            ]
        }))
        
        # Run comprehensive analysis
        analyzer = ComprehensiveEchoAnalyzer(str(tmp_path))
        analyzer.load_all_data()
        
        report = analyzer.generate_comprehensive_report()
        
        assert 'overview' in report
        assert 'cognitive_architecture' in report
        assert 'key_insights' in report
        
        # Save and verify
        output_file = tmp_path / "test_report.json"
        analyzer.save_report(str(output_file))
        
        assert output_file.exists()
        
        # Visualize
        visualizer = InsightsVisualizer(str(output_file))
        visualizer.load_analysis()
        
        md_report = visualizer.generate_markdown_report()
        assert len(md_report) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
