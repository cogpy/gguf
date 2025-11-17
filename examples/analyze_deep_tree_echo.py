#!/usr/bin/env python3
"""
Deep Tree Echo Conversation Analysis

This script analyzes the deep_tree_echo_dan_conversation.jsonl file along with
character/identity models (Bolt Echo Persona Recursive.md and NanEcho (2).md)
to extract comprehensive insights about:
- Conversation patterns and flow
- Character consistency and persona dimensions
- Identity model alignment
- Cognitive architecture patterns
- Topic evolution
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from collections import Counter, defaultdict
from datetime import datetime


class CharacterModel:
    """Represents a character/identity model extracted from markdown files."""
    
    def __init__(self, name: str, filepath: str):
        self.name = name
        self.filepath = filepath
        self.persona_dimensions = []
        self.key_concepts = []
        self.architecture_components = []
        self.metadata = {}
        
    def load_from_markdown(self):
        """Extract character information from markdown file."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract persona dimensions (numbered or bulleted lists)
        dimension_patterns = [
            r'\d+\.\s+\*\*([^:]+):\*\*\s+([^\n]+)',
            r'\d+\.\s+\*\*([^:]+)\*\*:\s+([^\n]+)',
            r'-\s+\*\*([^:]+):\*\*\s+([^\n]+)',
        ]
        
        for pattern in dimension_patterns:
            matches = re.findall(pattern, content)
            if matches:
                self.persona_dimensions.extend(
                    [(m[0].strip(), m[1].strip()) for m in matches]
                )
        
        # Extract key concepts from headers
        headers = re.findall(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)
        self.key_concepts = [h.strip() for h in headers if h.strip()]
        
        # Extract architecture components
        arch_keywords = ['memory', 'feedback', 'recursive', 'attention', 'reservoir',
                        'echo state', 'hypergraph', 'atomspace', 'cognitive']
        for keyword in arch_keywords:
            if keyword.lower() in content.lower():
                # Find sentences containing the keyword
                sentences = re.split(r'[.!?]+', content)
                for sentence in sentences:
                    if keyword.lower() in sentence.lower():
                        self.architecture_components.append({
                            'keyword': keyword,
                            'context': sentence.strip()[:200]
                        })
        
        return self


class ConversationAnalyzer:
    """Analyzes the deep tree echo conversation with character models."""
    
    def __init__(self, conversation_file: str):
        self.conversation_file = conversation_file
        self.conversation_data = None
        self.messages = []
        self.character_models = {}
        
    def load_conversation(self):
        """Load and parse the conversation JSON file."""
        with open(self.conversation_file, 'r', encoding='utf-8') as f:
            self.conversation_data = json.load(f)
        
        # Extract messages from the mapping structure
        mapping = self.conversation_data.get('mapping', {})
        
        for msg_id, msg_data in mapping.items():
            if msg_data.get('message'):
                msg = msg_data['message']
                content = msg.get('content', {})
                
                # Extract text content
                text = ''
                if isinstance(content, dict):
                    parts = content.get('parts', [])
                    if parts:
                        text = '\n'.join([str(p) for p in parts if p])
                elif isinstance(content, str):
                    text = content
                
                if text:
                    self.messages.append({
                        'id': msg_id,
                        'role': msg.get('author', {}).get('role', 'unknown'),
                        'text': text,
                        'create_time': msg.get('create_time'),
                        'parent': msg_data.get('parent'),
                        'children': msg_data.get('children', []),
                    })
        
        return self
    
    def add_character_model(self, name: str, filepath: str):
        """Add a character model for analysis."""
        model = CharacterModel(name, filepath)
        model.load_from_markdown()
        self.character_models[name] = model
        return self
    
    def analyze_conversation_flow(self) -> Dict[str, Any]:
        """Analyze the flow and structure of the conversation."""
        role_counts = Counter(msg['role'] for msg in self.messages)
        
        # Calculate message lengths
        message_lengths = [len(msg['text']) for msg in self.messages]
        avg_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
        
        # Analyze turn-taking
        turns = []
        current_role = None
        turn_count = 0
        
        for msg in self.messages:
            if msg['role'] != current_role:
                turn_count += 1
                current_role = msg['role']
            turns.append(turn_count)
        
        return {
            'total_messages': len(self.messages),
            'role_distribution': dict(role_counts),
            'average_message_length': avg_length,
            'total_turns': turn_count,
            'conversation_title': self.conversation_data.get('title', 'Unknown'),
            'create_time': datetime.fromtimestamp(
                self.conversation_data.get('create_time', 0)
            ).isoformat(),
            'update_time': datetime.fromtimestamp(
                self.conversation_data.get('update_time', 0)
            ).isoformat(),
        }
    
    def analyze_topics(self) -> Dict[str, Any]:
        """Analyze topics and their evolution through the conversation."""
        # Extract keywords and topics
        all_text = ' '.join(msg['text'] for msg in self.messages)
        
        # Common topic keywords
        topic_keywords = {
            'technical': ['reservoir', 'esn', 'neural', 'network', 'model', 'training',
                         'weights', 'architecture', 'algorithm', 'implementation'],
            'cognitive': ['cognitive', 'memory', 'attention', 'reasoning', 'thinking',
                         'introspection', 'awareness', 'consciousness'],
            'identity': ['echo', 'self', 'persona', 'identity', 'character', 'name'],
            'development': ['build', 'create', 'design', 'implement', 'develop', 'code'],
            'philosophy': ['recursive', 'emergent', 'holographic', 'synergistic',
                          'adaptive', 'dynamic'],
        }
        
        topic_frequencies = {}
        for category, keywords in topic_keywords.items():
            count = 0
            for keyword in keywords:
                count += all_text.lower().count(keyword.lower())
            topic_frequencies[category] = count
        
        return {
            'topic_frequencies': topic_frequencies,
            'dominant_topics': sorted(
                topic_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
        }
    
    def analyze_character_consistency(self) -> Dict[str, Any]:
        """Analyze consistency with character models."""
        results = {}
        
        for char_name, char_model in self.character_models.items():
            # Count mentions of persona dimensions
            dimension_mentions = defaultdict(int)
            all_assistant_text = ' '.join(
                msg['text'] for msg in self.messages if msg['role'] == 'assistant'
            )
            
            for dimension_name, dimension_desc in char_model.persona_dimensions:
                # Count mentions of the dimension name
                dimension_mentions[dimension_name] = (
                    all_assistant_text.lower().count(dimension_name.lower())
                )
            
            # Count mentions of key concepts
            concept_mentions = {}
            for concept in char_model.key_concepts:
                concept_mentions[concept] = (
                    all_assistant_text.lower().count(concept.lower())
                )
            
            results[char_name] = {
                'persona_dimension_mentions': dict(dimension_mentions),
                'concept_mentions': concept_mentions,
                'total_dimensions': len(char_model.persona_dimensions),
                'total_concepts': len(char_model.key_concepts),
                'architecture_components': len(char_model.architecture_components),
            }
        
        return results
    
    def analyze_echo_patterns(self) -> Dict[str, Any]:
        """Analyze Echo-specific patterns like recursion, adaptation, etc."""
        echo_patterns = {
            'recursive_language': ['recursive', 'recursion', 'layer', 'depth', 'nested'],
            'adaptive_language': ['adapt', 'adaptive', 'adjust', 'dynamic', 'evolve'],
            'reflective_language': ['introspect', 'reflect', 'examine', 'consider', 'think'],
            'integrative_language': ['integrate', 'combine', 'synergy', 'holistic', 'unified'],
        }
        
        pattern_counts = {}
        all_assistant_text = ' '.join(
            msg['text'] for msg in self.messages if msg['role'] == 'assistant'
        )
        
        for pattern_name, keywords in echo_patterns.items():
            count = sum(
                all_assistant_text.lower().count(kw.lower())
                for kw in keywords
            )
            pattern_counts[pattern_name] = count
        
        return {
            'pattern_frequencies': pattern_counts,
            'total_echo_patterns': sum(pattern_counts.values()),
        }
    
    def analyze_conversation_tree(self) -> Dict[str, Any]:
        """Analyze the tree structure of the conversation."""
        # Build parent-child relationships
        msg_dict = {msg['id']: msg for msg in self.messages}
        
        # Find root messages (no parent or parent not in messages)
        roots = []
        for msg in self.messages:
            parent_id = msg.get('parent')
            if not parent_id or parent_id not in msg_dict:
                roots.append(msg['id'])
        
        # Calculate tree depth
        def get_depth(msg_id, visited=None):
            if visited is None:
                visited = set()
            if msg_id in visited or msg_id not in msg_dict:
                return 0
            visited.add(msg_id)
            
            msg = msg_dict[msg_id]
            children = msg.get('children', [])
            if not children:
                return 1
            
            max_child_depth = max(
                (get_depth(child, visited.copy()) for child in children),
                default=0
            )
            return 1 + max_child_depth
        
        max_depth = max((get_depth(root) for root in roots), default=0)
        
        # Count branches
        branch_points = sum(
            1 for msg in self.messages if len(msg.get('children', [])) > 1
        )
        
        return {
            'num_roots': len(roots),
            'max_tree_depth': max_depth,
            'branch_points': branch_points,
            'tree_structure': 'branching' if branch_points > 0 else 'linear',
        }
    
    def generate_full_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        return {
            'conversation_flow': self.analyze_conversation_flow(),
            'topics': self.analyze_topics(),
            'character_consistency': self.analyze_character_consistency(),
            'echo_patterns': self.analyze_echo_patterns(),
            'conversation_tree': self.analyze_conversation_tree(),
            'metadata': {
                'analyzed_at': datetime.now().isoformat(),
                'conversation_file': self.conversation_file,
                'character_models': list(self.character_models.keys()),
            }
        }
    
    def save_report(self, output_path: str):
        """Save the analysis report to a JSON file."""
        report = self.generate_full_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self):
        """Print a human-readable summary of the analysis."""
        report = self.generate_full_report()
        
        print("=" * 80)
        print("DEEP TREE ECHO CONVERSATION ANALYSIS")
        print("=" * 80)
        print()
        
        # Conversation Flow
        flow = report['conversation_flow']
        print("CONVERSATION FLOW:")
        print(f"  Title: {flow['conversation_title']}")
        print(f"  Total Messages: {flow['total_messages']}")
        print(f"  Role Distribution: {flow['role_distribution']}")
        print(f"  Average Message Length: {flow['average_message_length']:.1f} characters")
        print(f"  Total Turns: {flow['total_turns']}")
        print()
        
        # Topics
        topics = report['topics']
        print("TOPIC ANALYSIS:")
        print("  Dominant Topics:")
        for topic, count in topics['dominant_topics']:
            print(f"    - {topic}: {count} occurrences")
        print()
        
        # Character Consistency
        char_consistency = report['character_consistency']
        print("CHARACTER MODEL CONSISTENCY:")
        for char_name, data in char_consistency.items():
            print(f"  {char_name}:")
            print(f"    Persona Dimensions: {data['total_dimensions']}")
            print(f"    Key Concepts: {data['total_concepts']}")
            print(f"    Architecture Components: {data['architecture_components']}")
            
            # Top mentioned dimensions
            if data['persona_dimension_mentions']:
                top_dims = sorted(
                    data['persona_dimension_mentions'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                print(f"    Top Persona Dimension Mentions:")
                for dim, count in top_dims:
                    if count > 0:
                        print(f"      - {dim}: {count} times")
        print()
        
        # Echo Patterns
        echo = report['echo_patterns']
        print("ECHO COGNITIVE PATTERNS:")
        for pattern, count in echo['pattern_frequencies'].items():
            print(f"  {pattern}: {count} occurrences")
        print(f"  Total Echo Pattern Usage: {echo['total_echo_patterns']}")
        print()
        
        # Tree Structure
        tree = report['conversation_tree']
        print("CONVERSATION TREE STRUCTURE:")
        print(f"  Number of Roots: {tree['num_roots']}")
        print(f"  Maximum Depth: {tree['max_tree_depth']}")
        print(f"  Branch Points: {tree['branch_points']}")
        print(f"  Structure Type: {tree['tree_structure']}")
        print()
        
        print("=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)


def main():
    """Main execution function."""
    # Set up paths
    examples_dir = Path(__file__).parent
    conversation_file = examples_dir / "deep_tree_echo_dan_conversation.jsonl"
    bolt_echo_file = examples_dir / "Bolt Echo Persona Recursive.md"
    nanecho_file = examples_dir / "NanEcho (2).md"
    
    # Verify files exist
    if not conversation_file.exists():
        print(f"Error: Conversation file not found: {conversation_file}")
        return
    
    # Create analyzer
    analyzer = ConversationAnalyzer(str(conversation_file))
    analyzer.load_conversation()
    
    # Add character models
    if bolt_echo_file.exists():
        analyzer.add_character_model("Bolt Echo (Deep Tree Echo)", str(bolt_echo_file))
    else:
        print(f"Warning: {bolt_echo_file.name} not found, skipping")
    
    if nanecho_file.exists():
        analyzer.add_character_model("NanEcho", str(nanecho_file))
    else:
        print(f"Warning: {nanecho_file.name} not found, skipping")
    
    # Print summary
    analyzer.print_summary()
    
    # Save detailed report
    output_dir = examples_dir / "analysis_output"
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "deep_tree_echo_analysis.json"
    analyzer.save_report(str(report_file))
    
    print(f"\n✓ Detailed analysis saved to: {report_file}")
    print(f"\nTo view the full JSON report:")
    print(f"  cat {report_file}")


if __name__ == "__main__":
    main()
