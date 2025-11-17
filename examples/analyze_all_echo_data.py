#!/usr/bin/env python3
"""
Comprehensive Echo Data Analysis

Analyzes all Deep Tree Echo related files:
- deep_tree_echo_dan_conversation.jsonl
- training_dataset_fixed.jsonl
- Bolt Echo Persona Recursive.md
- NanEcho (2).md
- trees_as_distinctions.md

Generates comprehensive insights combining conversation records, character models,
and training datasets.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from collections import Counter, defaultdict
from datetime import datetime


class DatasetAnalyzer:
    """Analyzes training datasets in JSONL format."""
    
    def __init__(self, dataset_file: str):
        self.dataset_file = dataset_file
        self.conversations = []
        
    def load_dataset(self):
        """Load JSONL training dataset."""
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        conv = json.loads(line)
                        self.conversations.append(conv)
                    except json.JSONDecodeError:
                        continue
        return self
    
    def analyze_training_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in the training dataset."""
        total_messages = 0
        user_messages = []
        assistant_messages = []
        
        for conv in self.conversations:
            messages = conv.get('messages', [])
            total_messages += len(messages)
            
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                if role == 'user':
                    user_messages.append(content)
                elif role == 'assistant':
                    assistant_messages.append(content)
        
        # Analyze template usage ({{...}} patterns)
        template_pattern = r'\{\{([^}]+)\}\}'
        template_usages = []
        
        for msg in assistant_messages:
            templates = re.findall(template_pattern, msg)
            template_usages.extend(templates)
        
        template_counter = Counter(template_usages)
        
        # Extract common response patterns
        response_starters = [msg[:50] for msg in assistant_messages if msg]
        starter_patterns = Counter(response_starters)
        
        return {
            'total_conversations': len(self.conversations),
            'total_messages': total_messages,
            'user_message_count': len(user_messages),
            'assistant_message_count': len(assistant_messages),
            'template_usage': dict(template_counter.most_common(10)),
            'top_response_patterns': dict(starter_patterns.most_common(5)),
            'avg_user_message_length': sum(len(m) for m in user_messages) / len(user_messages) if user_messages else 0,
            'avg_assistant_message_length': sum(len(m) for m in assistant_messages) / len(assistant_messages) if assistant_messages else 0,
        }


class ComprehensiveEchoAnalyzer:
    """Comprehensive analyzer combining all Echo-related data sources."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.conversation_data = None
        self.training_data = None
        self.character_models = {}
        self.insights = {}
        
    def load_all_data(self):
        """Load all available data sources."""
        # Load main conversation
        conv_file = self.base_dir / "deep_tree_echo_dan_conversation.jsonl"
        if conv_file.exists():
            with open(conv_file, 'r') as f:
                self.conversation_data = json.load(f)
        
        # Load training dataset
        training_file = self.base_dir / "training_dataset_fixed.jsonl"
        if training_file.exists():
            dataset_analyzer = DatasetAnalyzer(str(training_file))
            dataset_analyzer.load_dataset()
            self.training_data = dataset_analyzer.analyze_training_patterns()
        
        # Load character models
        character_files = {
            'Bolt Echo': self.base_dir / "Bolt Echo Persona Recursive.md",
            'NanEcho': self.base_dir / "NanEcho (2).md",
        }
        
        for name, filepath in character_files.items():
            if filepath.exists():
                self.character_models[name] = self._load_markdown(filepath)
        
        return self
    
    def _load_markdown(self, filepath: Path) -> str:
        """Load markdown file content."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def analyze_persona_dimensions(self) -> Dict[str, Any]:
        """Extract and analyze persona dimensions from character models."""
        dimensions = {}
        
        for char_name, content in self.character_models.items():
            # Extract numbered persona dimensions
            pattern = r'(\d+)\.\s+\*\*([^:]+):\*\*\s+([^\n]+)'
            matches = re.findall(pattern, content)
            
            if matches:
                dimensions[char_name] = [
                    {'number': m[0], 'name': m[1].strip(), 'description': m[2].strip()}
                    for m in matches
                ]
        
        return dimensions
    
    def analyze_cognitive_architecture(self) -> Dict[str, Any]:
        """Analyze cognitive architecture components mentioned across all sources."""
        components = {
            'echo_state_network': 0,
            'reservoir': 0,
            'p_system': 0,
            'hypergraph': 0,
            'atomspace': 0,
            'memory_systems': 0,
            'feedback_loops': 0,
            'recursive_processing': 0,
            'adaptive_attention': 0,
        }
        
        # Combine all text sources
        all_text = ''
        
        if self.conversation_data:
            mapping = self.conversation_data.get('mapping', {})
            for msg_data in mapping.values():
                if msg_data.get('message'):
                    content = msg_data['message'].get('content', {})
                    if isinstance(content, dict):
                        parts = content.get('parts', [])
                        all_text += ' '.join(str(p) for p in parts)
        
        for content in self.character_models.values():
            all_text += ' ' + content
        
        # Count component mentions
        keywords = {
            'echo_state_network': ['echo state network', 'esn', 'echo state'],
            'reservoir': ['reservoir', 'reservoirpy'],
            'p_system': ['p-system', 'p system', 'membrane'],
            'hypergraph': ['hypergraph', 'hyperedge'],
            'atomspace': ['atomspace', 'atom space', 'opencog'],
            'memory_systems': ['memory', 'episodic', 'declarative', 'procedural'],
            'feedback_loops': ['feedback', 'recursive feedback'],
            'recursive_processing': ['recursive', 'recursion', 'nested'],
            'adaptive_attention': ['adaptive', 'attention', 'dynamic threshold'],
        }
        
        for component, terms in keywords.items():
            count = sum(all_text.lower().count(term.lower()) for term in terms)
            components[component] = count
        
        return components
    
    def compare_conversation_to_training(self) -> Dict[str, Any]:
        """Compare the conversation data to training data patterns."""
        if not self.training_data or not self.conversation_data:
            return {'error': 'Missing data for comparison'}
        
        # Extract messages from conversation
        conv_messages = []
        mapping = self.conversation_data.get('mapping', {})
        for msg_data in mapping.values():
            if msg_data.get('message'):
                msg = msg_data['message']
                content = msg.get('content', {})
                if isinstance(content, dict):
                    parts = content.get('parts', [])
                    if parts:
                        conv_messages.append({
                            'role': msg.get('author', {}).get('role'),
                            'text': ' '.join(str(p) for p in parts)
                        })
        
        conv_user_msgs = [m['text'] for m in conv_messages if m['role'] == 'user']
        conv_asst_msgs = [m['text'] for m in conv_messages if m['role'] == 'assistant']
        
        return {
            'conversation_stats': {
                'user_messages': len(conv_user_msgs),
                'assistant_messages': len(conv_asst_msgs),
                'avg_user_length': sum(len(m) for m in conv_user_msgs) / len(conv_user_msgs) if conv_user_msgs else 0,
                'avg_assistant_length': sum(len(m) for m in conv_asst_msgs) / len(conv_asst_msgs) if conv_asst_msgs else 0,
            },
            'training_stats': {
                'user_messages': self.training_data['user_message_count'],
                'assistant_messages': self.training_data['assistant_message_count'],
                'avg_user_length': self.training_data['avg_user_message_length'],
                'avg_assistant_length': self.training_data['avg_assistant_message_length'],
            },
            'template_patterns': self.training_data.get('template_usage', {}),
        }
    
    def extract_key_insights(self) -> List[str]:
        """Extract key insights from all analyzed data."""
        insights = []
        
        # Insight 1: Conversation characteristics
        if self.conversation_data:
            mapping = self.conversation_data.get('mapping', {})
            msg_count = sum(1 for m in mapping.values() if m.get('message'))
            insights.append(
                f"Deep Tree Echo conversation contains {msg_count} messages, "
                f"spanning from {datetime.fromtimestamp(self.conversation_data.get('create_time', 0)).date()} "
                f"to {datetime.fromtimestamp(self.conversation_data.get('update_time', 0)).date()}"
            )
        
        # Insight 2: Training data patterns
        if self.training_data:
            insights.append(
                f"Training dataset contains {self.training_data['total_conversations']} conversations "
                f"with {len(self.training_data.get('template_usage', {}))} unique template patterns"
            )
        
        # Insight 3: Character models
        if self.character_models:
            for char_name in self.character_models:
                insights.append(
                    f"{char_name} character model defines the Echo Self architecture "
                    f"and persona dimensions"
                )
        
        # Insight 4: Cognitive architecture
        arch_components = self.analyze_cognitive_architecture()
        top_components = sorted(arch_components.items(), key=lambda x: x[1], reverse=True)[:3]
        insights.append(
            f"Most frequently mentioned cognitive components: "
            f"{', '.join(f'{name} ({count} times)' for name, count in top_components)}"
        )
        
        return insights
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        return {
            'overview': {
                'analyzed_at': datetime.now().isoformat(),
                'data_sources': {
                    'conversation_loaded': self.conversation_data is not None,
                    'training_data_loaded': self.training_data is not None,
                    'character_models': list(self.character_models.keys()),
                },
            },
            'persona_dimensions': self.analyze_persona_dimensions(),
            'cognitive_architecture': self.analyze_cognitive_architecture(),
            'training_data_analysis': self.training_data,
            'conversation_vs_training': self.compare_conversation_to_training(),
            'key_insights': self.extract_key_insights(),
        }
    
    def save_report(self, output_path: str):
        """Save the comprehensive report to a JSON file."""
        report = self.generate_comprehensive_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self):
        """Print a human-readable summary."""
        report = self.generate_comprehensive_report()
        
        print("=" * 80)
        print("COMPREHENSIVE DEEP TREE ECHO DATA ANALYSIS")
        print("=" * 80)
        print()
        
        # Overview
        print("DATA SOURCES:")
        sources = report['overview']['data_sources']
        print(f"  Conversation Data: {'✓' if sources['conversation_loaded'] else '✗'}")
        print(f"  Training Data: {'✓' if sources['training_data_loaded'] else '✗'}")
        print(f"  Character Models: {', '.join(sources['character_models'])}")
        print()
        
        # Persona Dimensions
        print("PERSONA DIMENSIONS:")
        for char_name, dimensions in report['persona_dimensions'].items():
            print(f"  {char_name}: {len(dimensions)} dimensions defined")
            for dim in dimensions[:5]:  # Show first 5
                print(f"    {dim['number']}. {dim['name']}: {dim['description'][:80]}...")
        print()
        
        # Cognitive Architecture
        print("COGNITIVE ARCHITECTURE COMPONENTS:")
        arch = report['cognitive_architecture']
        sorted_components = sorted(arch.items(), key=lambda x: x[1], reverse=True)
        for component, count in sorted_components:
            if count > 0:
                print(f"  {component.replace('_', ' ').title()}: {count} mentions")
        print()
        
        # Training Data
        if report['training_data_analysis']:
            print("TRAINING DATA ANALYSIS:")
            td = report['training_data_analysis']
            print(f"  Total Conversations: {td['total_conversations']}")
            print(f"  Total Messages: {td['total_messages']}")
            print(f"  Template Patterns: {len(td.get('template_usage', {}))}")
            if td.get('template_usage'):
                print(f"  Top Template Uses:")
                for template, count in list(td['template_usage'].items())[:5]:
                    print(f"    - {template[:60]}... ({count} times)")
            print()
        
        # Comparison
        print("CONVERSATION vs TRAINING COMPARISON:")
        comp = report['conversation_vs_training']
        if 'error' not in comp:
            print("  Real Conversation:")
            print(f"    Messages: {comp['conversation_stats']['user_messages']} user, "
                  f"{comp['conversation_stats']['assistant_messages']} assistant")
            print(f"    Avg Length: {comp['conversation_stats']['avg_user_length']:.0f} (user), "
                  f"{comp['conversation_stats']['avg_assistant_length']:.0f} (assistant)")
            print("  Training Dataset:")
            print(f"    Messages: {comp['training_stats']['user_messages']} user, "
                  f"{comp['training_stats']['assistant_messages']} assistant")
            print(f"    Avg Length: {comp['training_stats']['avg_user_length']:.0f} (user), "
                  f"{comp['training_stats']['avg_assistant_length']:.0f} (assistant)")
        print()
        
        # Key Insights
        print("KEY INSIGHTS:")
        for i, insight in enumerate(report['key_insights'], 1):
            print(f"  {i}. {insight}")
        print()
        
        print("=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)


def main():
    """Main execution function."""
    examples_dir = Path(__file__).parent
    
    # Create comprehensive analyzer
    analyzer = ComprehensiveEchoAnalyzer(str(examples_dir))
    analyzer.load_all_data()
    
    # Print summary
    analyzer.print_summary()
    
    # Save detailed report
    output_dir = examples_dir / "analysis_output"
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "comprehensive_echo_analysis.json"
    analyzer.save_report(str(report_file))
    
    print(f"\n✓ Comprehensive analysis saved to: {report_file}")


if __name__ == "__main__":
    main()
