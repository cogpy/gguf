"""
Vocabulary and Layer Activation Analysis Module

This module provides enhanced vocabulary analysis for conversation datasets,
particularly for understanding:
1. Complete vocabulary enumeration with word counts
2. Relationship between expressed vocabulary and total available vocabulary
3. Which layers and activations likely reflect vocabulary interactions
4. Connection to deep tree echo state reservoir computing framework

The analysis helps answer:
- What vocabulary is actually used vs. available?
- Which model components (layers, attention heads) are likely activated?
- How does the echo state reservoir interact with transformer layers?
- What dynamic persona variables are reflected in vocabulary patterns?
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
import numpy as np


class VocabularyAnalyzer:
    """
    Analyzes vocabulary usage patterns in conversations with model metadata.
    
    This analyzer focuses on understanding:
    - Complete vocabulary enumeration from conversations
    - Coverage analysis (expressed vs. total available)
    - Layer-specific vocabulary patterns
    - Echo state reservoir interactions
    """
    
    def __init__(
        self,
        total_vocab_size: Optional[int] = None,
        model_layers: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        architecture: Optional[str] = None
    ):
        """
        Initialize the vocabulary analyzer.
        
        Args:
            total_vocab_size: Total vocabulary size of the model
            model_layers: Number of layers in the model
            embedding_dim: Embedding dimension
            architecture: Model architecture (e.g., 'llama', 'gpt2')
        """
        self.total_vocab_size = total_vocab_size
        self.model_layers = model_layers
        self.embedding_dim = embedding_dim
        self.architecture = architecture
    
    def enumerate_vocabulary(
        self,
        conversations: List[Dict[str, Any]],
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Enumerate complete vocabulary with word counts from conversations.
        
        Args:
            conversations: List of conversation dicts with 'query', 'response',
                         and optional 'model' metadata
            include_metadata: Whether to track per-model vocabulary
        
        Returns:
            Dictionary containing vocabulary statistics
        """
        # Overall vocabulary counts
        word_counts = Counter()
        query_word_counts = Counter()
        response_word_counts = Counter()
        
        # Per-model vocabulary if metadata available
        model_vocabularies = defaultdict(Counter)
        
        # Track models mentioned
        models_found = set()
        
        for conv in conversations:
            # Extract query and response text
            query = conv.get('query', '')
            response = conv.get('response', '')
            model_info = conv.get('model', conv.get('metadata', {}).get('model'))
            
            if model_info:
                models_found.add(str(model_info))
            
            # Tokenize (simple whitespace + punctuation split)
            query_words = self._tokenize(query)
            response_words = self._tokenize(response)
            
            # Update overall counts
            word_counts.update(query_words + response_words)
            query_word_counts.update(query_words)
            response_word_counts.update(response_words)
            
            # Update per-model counts if metadata present
            if include_metadata and model_info:
                model_vocabularies[str(model_info)].update(query_words + response_words)
        
        # Calculate statistics
        unique_words = len(word_counts)
        total_words = sum(word_counts.values())
        
        result = {
            'total_unique_words': unique_words,
            'total_word_instances': total_words,
            'word_counts': dict(word_counts.most_common(100)),  # Top 100
            'query_unique_words': len(query_word_counts),
            'response_unique_words': len(response_word_counts),
            'models_found': list(models_found),
            'vocabulary_diversity': unique_words / total_words if total_words > 0 else 0,
        }
        
        # Add per-model vocabulary if tracked
        if include_metadata and model_vocabularies:
            result['per_model_vocabulary'] = {
                model: {
                    'unique_words': len(vocab),
                    'total_instances': sum(vocab.values()),
                    'top_words': dict(vocab.most_common(20))
                }
                for model, vocab in model_vocabularies.items()
            }
        
        # Add coverage metrics if total vocab known
        if self.total_vocab_size:
            result['vocabulary_coverage'] = {
                'expressed_vocab_size': unique_words,
                'total_vocab_size': self.total_vocab_size,
                'coverage_ratio': unique_words / self.total_vocab_size,
                'coverage_percentage': (unique_words / self.total_vocab_size) * 100,
                'unexpressed_vocab_size': self.total_vocab_size - unique_words,
            }
        
        return result
    
    def analyze_layer_activations(
        self,
        vocabulary_stats: Dict[str, Any],
        conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Infer which layers and activations are likely involved based on vocabulary patterns.
        
        This analysis uses vocabulary complexity and patterns to estimate which
        transformer layers are most active.
        
        Args:
            vocabulary_stats: Output from enumerate_vocabulary
            conversations: Original conversation data
        
        Returns:
            Dictionary containing layer activation insights
        """
        insights = {
            'description': 'Inferred layer activation patterns based on vocabulary usage',
            'layer_activation_estimates': [],
            'attention_head_insights': [],
            'embedding_layer_analysis': {},
            'output_layer_analysis': {},
        }
        
        # Analyze embedding layer involvement
        unique_words = vocabulary_stats['total_unique_words']
        coverage = vocabulary_stats.get('vocabulary_coverage', {})
        
        insights['embedding_layer_analysis'] = {
            'active_embeddings': unique_words,
            'total_embeddings': self.total_vocab_size if self.total_vocab_size else 'unknown',
            'activation_ratio': coverage.get('coverage_ratio', 'unknown'),
            'interpretation': self._interpret_embedding_activation(coverage.get('coverage_ratio'))
        }
        
        # Analyze vocabulary complexity to infer deeper layer activation
        word_counts = Counter(vocabulary_stats['word_counts'])
        
        # Calculate vocabulary complexity metrics
        avg_word_freq = np.mean(list(word_counts.values())) if word_counts else 0
        std_word_freq = np.std(list(word_counts.values())) if word_counts else 0
        
        # Estimate layer involvement based on complexity
        if self.model_layers:
            layer_estimates = self._estimate_layer_involvement(
                unique_words,
                avg_word_freq,
                std_word_freq,
                conversations
            )
            insights['layer_activation_estimates'] = layer_estimates
        
        # Analyze attention patterns from vocabulary
        attention_insights = self._infer_attention_patterns(
            vocabulary_stats,
            conversations
        )
        insights['attention_head_insights'] = attention_insights
        
        # Analyze output layer
        insights['output_layer_analysis'] = {
            'output_vocabulary_size': vocabulary_stats['response_unique_words'],
            'query_to_response_vocab_ratio': (
                vocabulary_stats['response_unique_words'] /
                vocabulary_stats['query_unique_words']
                if vocabulary_stats['query_unique_words'] > 0 else 0
            ),
            'interpretation': 'Output layer actively generates diverse responses'
                if vocabulary_stats['response_unique_words'] > vocabulary_stats['query_unique_words']
                else 'Output layer mirrors input vocabulary'
        }
        
        return insights
    
    def analyze_echo_reservoir_interaction(
        self,
        vocabulary_stats: Dict[str, Any],
        layer_insights: Dict[str, Any],
        conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze how the echo state reservoir computing framework interacts
        with transformer layers based on vocabulary patterns.
        
        Args:
            vocabulary_stats: Vocabulary enumeration results
            layer_insights: Layer activation insights
            conversations: Original conversation data
        
        Returns:
            Dictionary containing echo reservoir interaction analysis
        """
        analysis = {
            'description': (
                'Analysis of deep tree echo state reservoir computing '
                'interaction with transformer architecture'
            ),
            'reservoir_dynamics': {},
            'feedback_loops': {},
            'persona_variables': {},
            'temporal_patterns': {},
            'recommendations': []
        }
        
        # Analyze reservoir state dynamics
        analysis['reservoir_dynamics'] = self._analyze_reservoir_dynamics(
            vocabulary_stats,
            conversations
        )
        
        # Analyze feedback loops (echo patterns)
        analysis['feedback_loops'] = self._analyze_feedback_patterns(
            conversations
        )
        
        # Analyze dynamic persona variables reflected in vocabulary
        analysis['persona_variables'] = self._analyze_persona_variables(
            vocabulary_stats,
            conversations
        )
        
        # Analyze temporal patterns
        analysis['temporal_patterns'] = self._analyze_temporal_vocabulary_patterns(
            conversations
        )
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(
            vocabulary_stats,
            layer_insights,
            analysis
        )
        
        return analysis
    
    def generate_comprehensive_report(
        self,
        conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive vocabulary and layer activation analysis report.
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Complete analysis report
        """
        # Enumerate vocabulary
        vocab_stats = self.enumerate_vocabulary(conversations)
        
        # Analyze layer activations
        layer_insights = self.analyze_layer_activations(vocab_stats, conversations)
        
        # Analyze echo reservoir interactions
        echo_analysis = self.analyze_echo_reservoir_interaction(
            vocab_stats,
            layer_insights,
            conversations
        )
        
        return {
            'vocabulary_analysis': vocab_stats,
            'layer_activation_analysis': layer_insights,
            'echo_reservoir_analysis': echo_analysis,
            'model_configuration': {
                'total_vocab_size': self.total_vocab_size,
                'model_layers': self.model_layers,
                'embedding_dim': self.embedding_dim,
                'architecture': self.architecture,
            },
            'summary': self._generate_summary(
                vocab_stats,
                layer_insights,
                echo_analysis
            )
        }
    
    # Helper methods
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization of text."""
        import re
        # Split on whitespace and punctuation, keep words and numbers
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _interpret_embedding_activation(self, coverage_ratio: Any) -> str:
        """Interpret embedding layer activation level."""
        if coverage_ratio == 'unknown' or coverage_ratio is None:
            return 'Unknown - total vocabulary size not provided'
        
        ratio = float(coverage_ratio)
        if ratio < 0.01:
            return 'Very sparse activation - only a tiny fraction of embeddings used'
        elif ratio < 0.05:
            return 'Sparse activation - small subset of vocabulary active'
        elif ratio < 0.20:
            return 'Moderate activation - significant vocabulary subset in use'
        elif ratio < 0.50:
            return 'High activation - large portion of vocabulary engaged'
        else:
            return 'Very high activation - majority of vocabulary in active use'
    
    def _estimate_layer_involvement(
        self,
        unique_words: int,
        avg_freq: float,
        std_freq: float,
        conversations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Estimate which layers are most involved based on vocabulary patterns."""
        layer_estimates = []
        
        # Early layers (1-25% depth): Primarily embedding and basic pattern recognition
        early_ratio = min(self.model_layers, max(1, self.model_layers // 4))
        layer_estimates.append({
            'layer_range': f'1-{early_ratio}',
            'layer_type': 'Early Layers',
            'primary_function': 'Embedding lookup and basic pattern recognition',
            'activation_level': 'high',
            'evidence': f'All {unique_words} unique words require embedding lookups',
            'contribution': 'Converting tokens to dense vector representations'
        })
        
        # Middle layers (25-75% depth): Contextual understanding and composition
        mid_start = early_ratio + 1
        mid_end = max(mid_start, self.model_layers * 3 // 4)
        layer_estimates.append({
            'layer_range': f'{mid_start}-{mid_end}',
            'layer_type': 'Middle Layers',
            'primary_function': 'Contextual composition and semantic processing',
            'activation_level': self._estimate_middle_layer_activation(avg_freq, std_freq),
            'evidence': f'Vocabulary diversity of {unique_words} words suggests complex semantic processing',
            'contribution': 'Building compositional representations and context integration'
        })
        
        # Late layers (75-100% depth): Task-specific processing and output preparation
        late_start = mid_end + 1
        layer_estimates.append({
            'layer_range': f'{late_start}-{self.model_layers}',
            'layer_type': 'Late Layers',
            'primary_function': 'Task-specific processing and output generation',
            'activation_level': 'high',
            'evidence': 'Response generation requires full model depth',
            'contribution': 'Preparing final outputs and selecting next tokens'
        })
        
        return layer_estimates
    
    def _estimate_middle_layer_activation(self, avg_freq: float, std_freq: float) -> str:
        """Estimate middle layer activation based on vocabulary statistics."""
        # Higher diversity (lower avg frequency, higher std) suggests more middle layer activation
        if avg_freq > 10:
            return 'moderate'  # Repetitive vocabulary needs less processing
        elif std_freq > avg_freq:
            return 'very high'  # High variance suggests complex processing
        else:
            return 'high'
    
    def _infer_attention_patterns(
        self,
        vocabulary_stats: Dict[str, Any],
        conversations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Infer attention head activation patterns from vocabulary."""
        patterns = []
        
        # Analyze query-response vocabulary overlap
        query_words = set(vocabulary_stats.get('word_counts', {}).keys())
        
        patterns.append({
            'pattern_type': 'Cross-Attention',
            'description': 'Attention from response to query tokens',
            'likelihood': 'very high',
            'reasoning': (
                'Response generation requires attending to query context. '
                'All models use cross-attention between query and response.'
            )
        })
        
        patterns.append({
            'pattern_type': 'Self-Attention (Query)',
            'description': 'Within-query token relationships',
            'likelihood': 'high',
            'reasoning': 'Query encoding requires self-attention to build contextual representations'
        })
        
        patterns.append({
            'pattern_type': 'Self-Attention (Response)',
            'description': 'Within-response token dependencies',
            'likelihood': 'very high',
            'reasoning': 'Autoregressive generation uses causal self-attention for coherent outputs'
        })
        
        # Estimate number of active attention heads based on vocabulary diversity
        vocab_diversity = vocabulary_stats.get('vocabulary_diversity', 0)
        if vocab_diversity > 0.1:
            patterns.append({
                'pattern_type': 'Multi-Head Diversity',
                'description': 'Multiple attention heads tracking different aspects',
                'likelihood': 'high',
                'reasoning': (
                    f'Vocabulary diversity of {vocab_diversity:.3f} suggests '
                    'different heads attend to syntax, semantics, and context'
                )
            })
        
        return patterns
    
    def _analyze_reservoir_dynamics(
        self,
        vocabulary_stats: Dict[str, Any],
        conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze echo state reservoir dynamics."""
        return {
            'state_space_dimensionality': (
                f'Estimated at {vocabulary_stats["total_unique_words"]} based on vocabulary'
            ),
            'recurrent_connections': (
                'Echo state network maintains recurrent connections that '
                'act as a reservoir of temporal dynamics'
            ),
            'readout_layer': (
                'Transformer layers act as trainable readout from reservoir states'
            ),
            'temporal_memory': (
                'Conversation history encoded in reservoir state, '
                'providing context for transformer processing'
            ),
            'state_update_mechanism': (
                'Each new token updates reservoir state via recurrent dynamics, '
                'which then feeds into transformer attention mechanisms'
            )
        }
    
    def _analyze_feedback_patterns(
        self,
        conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze feedback loop patterns in conversations."""
        # Look for word repetition patterns (echo effects)
        all_queries = [conv.get('query', '') for conv in conversations]
        all_responses = [conv.get('response', '') for conv in conversations]
        
        # Count query words that appear in responses
        echo_count = 0
        total_query_words = 0
        
        for query, response in zip(all_queries, all_responses):
            query_words = set(self._tokenize(query))
            response_words = set(self._tokenize(response))
            total_query_words += len(query_words)
            echo_count += len(query_words & response_words)
        
        echo_ratio = echo_count / total_query_words if total_query_words > 0 else 0
        
        return {
            'echo_ratio': echo_ratio,
            'interpretation': (
                f'{echo_ratio:.1%} of query words appear in responses, '
                'suggesting feedback from input to output'
            ),
            'feedback_strength': (
                'strong' if echo_ratio > 0.3 else
                'moderate' if echo_ratio > 0.1 else
                'weak'
            ),
            'mechanism': (
                'Reservoir state captures query information and feeds it back '
                'through transformer layers during response generation'
            )
        }
    
    def _analyze_persona_variables(
        self,
        vocabulary_stats: Dict[str, Any],
        conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze dynamic persona variables reflected in vocabulary."""
        # Define persona-related vocabulary categories
        persona_indicators = {
            'formal_language': ['please', 'kindly', 'respectfully', 'formally', 'sir', 'madam'],
            'casual_language': ['hey', 'yeah', 'cool', 'awesome', 'lol', 'btw'],
            'technical_language': ['algorithm', 'function', 'implementation', 'architecture', 'model'],
            'emotional_language': ['feel', 'believe', 'think', 'love', 'hate', 'enjoy'],
            'cognitive_language': ['understand', 'analyze', 'consider', 'examine', 'evaluate'],
        }
        
        word_counts = vocabulary_stats.get('word_counts', {})
        
        persona_scores = {}
        for category, indicators in persona_indicators.items():
            score = sum(word_counts.get(word, 0) for word in indicators)
            persona_scores[category] = score
        
        # Find dominant persona dimensions
        total_persona_words = sum(persona_scores.values())
        
        return {
            'persona_dimensions_detected': persona_scores,
            'dominant_persona': max(persona_scores.items(), key=lambda x: x[1])[0] if persona_scores else 'unknown',
            'persona_diversity': len([s for s in persona_scores.values() if s > 0]),
            'interpretation': (
                'Multiple persona dimensions active in responses, '
                'suggesting dynamic persona adjustment by reservoir-transformer system'
                if len([s for s in persona_scores.values() if s > 0]) > 2
                else 'Limited persona variation detected'
            ),
            'total_persona_markers': total_persona_words
        }
    
    def _analyze_temporal_vocabulary_patterns(
        self,
        conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze how vocabulary patterns change over time/sequence."""
        if not conversations:
            return {'pattern': 'no data'}
        
        # Analyze vocabulary evolution if conversations have sequence info
        vocab_over_time = []
        
        for i, conv in enumerate(conversations):
            words = self._tokenize(conv.get('query', '') + ' ' + conv.get('response', ''))
            vocab_over_time.append({
                'position': i,
                'unique_words': len(set(words)),
                'total_words': len(words)
            })
        
        # Calculate vocabulary growth
        if len(vocab_over_time) > 1:
            early_vocab = np.mean([v['unique_words'] for v in vocab_over_time[:len(vocab_over_time)//2]])
            late_vocab = np.mean([v['unique_words'] for v in vocab_over_time[len(vocab_over_time)//2:]])
            
            return {
                'vocabulary_growth': late_vocab - early_vocab,
                'early_avg_unique_words': early_vocab,
                'late_avg_unique_words': late_vocab,
                'pattern': (
                    'expanding' if late_vocab > early_vocab * 1.1 else
                    'contracting' if late_vocab < early_vocab * 0.9 else
                    'stable'
                ),
                'interpretation': (
                    'Vocabulary expands over conversation, suggesting increasing context complexity'
                    if late_vocab > early_vocab * 1.1 else
                    'Vocabulary contracts over conversation, suggesting focus/convergence'
                    if late_vocab < early_vocab * 0.9 else
                    'Stable vocabulary usage throughout conversation'
                )
            }
        
        return {'pattern': 'insufficient data for temporal analysis'}
    
    def _generate_recommendations(
        self,
        vocabulary_stats: Dict[str, Any],
        layer_insights: Dict[str, Any],
        echo_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Vocabulary coverage recommendations
        coverage = vocabulary_stats.get('vocabulary_coverage', {})
        if coverage and coverage.get('coverage_ratio', 0) < 0.05:
            recommendations.append(
                'Low vocabulary coverage detected. Consider expanding conversation '
                'dataset to engage more of the model\'s vocabulary capacity.'
            )
        
        # Layer activation recommendations
        recommendations.append(
            'Focus interpretability efforts on middle layers where semantic '
            'composition occurs, as evidenced by vocabulary complexity patterns.'
        )
        
        # Echo reservoir recommendations
        feedback = echo_analysis.get('feedback_loops', {})
        if feedback.get('feedback_strength') == 'weak':
            recommendations.append(
                'Weak feedback loops detected. Consider adjusting reservoir parameters '
                'to enhance echo state dynamics and memory retention.'
            )
        
        # Persona recommendations
        persona = echo_analysis.get('persona_variables', {})
        if persona.get('persona_diversity', 0) < 2:
            recommendations.append(
                'Limited persona diversity. Consider incorporating varied conversation '
                'styles to engage different persona dimensions in the model.'
            )
        
        recommendations.append(
            'Monitor temporal vocabulary patterns to understand how reservoir state '
            'evolution affects transformer layer processing over conversation turns.'
        )
        
        return recommendations
    
    def _generate_summary(
        self,
        vocab_stats: Dict[str, Any],
        layer_insights: Dict[str, Any],
        echo_analysis: Dict[str, Any]
    ) -> str:
        """Generate a human-readable summary."""
        lines = []
        
        lines.append("VOCABULARY AND LAYER ACTIVATION ANALYSIS SUMMARY")
        lines.append("=" * 70)
        lines.append("")
        
        lines.append("VOCABULARY COVERAGE:")
        lines.append(f"  Unique words expressed: {vocab_stats['total_unique_words']}")
        lines.append(f"  Total word instances: {vocab_stats['total_word_instances']}")
        
        coverage = vocab_stats.get('vocabulary_coverage', {})
        if coverage:
            lines.append(f"  Coverage: {coverage['coverage_percentage']:.2f}% "
                        f"({coverage['expressed_vocab_size']} / {coverage['total_vocab_size']})")
        
        lines.append("")
        lines.append("LAYER ACTIVATION INSIGHTS:")
        
        embedding = layer_insights.get('embedding_layer_analysis', {})
        if embedding:
            lines.append(f"  Embedding layer: {embedding.get('interpretation', 'N/A')}")
        
        lines.append("")
        lines.append("ECHO RESERVOIR DYNAMICS:")
        
        feedback = echo_analysis.get('feedback_loops', {})
        if feedback:
            lines.append(f"  Feedback strength: {feedback.get('feedback_strength', 'unknown')}")
            lines.append(f"  Echo ratio: {feedback.get('echo_ratio', 0):.1%}")
        
        persona = echo_analysis.get('persona_variables', {})
        if persona:
            lines.append(f"  Dominant persona: {persona.get('dominant_persona', 'unknown')}")
            lines.append(f"  Persona diversity: {persona.get('persona_diversity', 0)} dimensions")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


def load_conversations_from_json(filepath: str) -> List[Dict[str, Any]]:
    """
    Load conversations from a JSON file.
    
    Expected format:
    [
        {
            "query": "...",
            "response": "...",
            "model": "gpt-4",  # optional
            "metadata": {...}   # optional
        },
        ...
    ]
    
    Or for JSONL files with 'messages' format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    Args:
        filepath: Path to JSON or JSONL file
    
    Returns:
        List of conversation dictionaries
    """
    conversations = []
    path = Path(filepath)
    
    if path.suffix == '.jsonl':
        # JSONL format
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Convert messages format to query/response format
                    if 'messages' in data:
                        messages = data['messages']
                        query = ''
                        response = ''
                        model = data.get('model')
                        
                        for msg in messages:
                            role = msg.get('role', '')
                            content = msg.get('content', '')
                            
                            if role == 'user':
                                query = content
                            elif role == 'assistant':
                                response = content
                                # Check if model is in assistant metadata
                                if not model and 'model' in msg:
                                    model = msg['model']
                        
                        if query or response:
                            conv = {'query': query, 'response': response}
                            if model:
                                conv['model'] = model
                            conversations.append(conv)
                    else:
                        conversations.append(data)
                        
                except json.JSONDecodeError:
                    continue
    else:
        # Regular JSON format
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                conversations = data
            elif isinstance(data, dict):
                # Handle mapping format (like ChatGPT exports)
                mapping = data.get('mapping', {})
                
                # Extract conversations from mapping
                for msg_id, msg_data in mapping.items():
                    if msg_data.get('message'):
                        msg = msg_data['message']
                        role = msg.get('author', {}).get('role')
                        content = msg.get('content', {})
                        
                        if isinstance(content, dict):
                            text = '\n'.join(str(p) for p in content.get('parts', []) if p)
                        else:
                            text = str(content)
                        
                        # For now, treat each message as a potential conversation
                        # This is simplified; real implementation might pair user/assistant
                        if text and role:
                            conversations.append({
                                'query': text if role == 'user' else '',
                                'response': text if role == 'assistant' else '',
                                'model': msg.get('metadata', {}).get('model_slug')
                            })
    
    return conversations
