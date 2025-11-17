"""
Conversation data analysis module for understanding what can be learned about
model weights and activations from query-response pairs.

This module helps answer questions like:
- What can we learn about model internals from conversation datasets?
- How does sequential ordering affect our analysis capabilities?
- What additional insights come from character prompts and parameters?
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import numpy as np


class ConversationPair:
    """Represents a single query-response pair."""
    
    def __init__(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        position: Optional[int] = None
    ):
        """
        Initialize a conversation pair.
        
        Args:
            query: User query/prompt
            response: Assistant response
            metadata: Optional metadata (character prompts, parameters, etc.)
            position: Position in conversation sequence (if applicable)
        """
        self.query = query
        self.response = response
        self.metadata = metadata or {}
        self.position = position
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'query': self.query,
            'response': self.response,
            'metadata': self.metadata,
            'position': self.position,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationPair':
        """Create from dictionary representation."""
        return cls(
            query=data.get('query', ''),
            response=data.get('response', ''),
            metadata=data.get('metadata', {}),
            position=data.get('position')
        )


class ConversationDataset:
    """Container for a dataset of conversation pairs."""
    
    def __init__(
        self,
        pairs: List[ConversationPair],
        is_sequential: bool = False,
        global_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a conversation dataset.
        
        Args:
            pairs: List of conversation pairs
            is_sequential: Whether pairs maintain conversation order
            global_metadata: Global metadata (model parameters, etc.)
        """
        self.pairs = pairs
        self.is_sequential = is_sequential
        self.global_metadata = global_metadata or {}
    
    def __len__(self) -> int:
        """Return number of conversation pairs."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> ConversationPair:
        """Get conversation pair by index."""
        return self.pairs[idx]
    
    @classmethod
    def from_json_file(cls, filepath: str) -> 'ConversationDataset':
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        pairs = [ConversationPair.from_dict(p) for p in data.get('pairs', [])]
        return cls(
            pairs=pairs,
            is_sequential=data.get('is_sequential', False),
            global_metadata=data.get('metadata', {})
        )
    
    def to_json_file(self, filepath: str):
        """Save dataset to JSON file."""
        data = {
            'pairs': [p.to_dict() for p in self.pairs],
            'is_sequential': self.is_sequential,
            'metadata': self.global_metadata,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class ConversationAnalyzer:
    """
    Analyzer for understanding what can be learned about model internals
    from conversation datasets.
    
    This analyzer helps answer fundamental questions:
    1. With known topology and vocabulary, what can we learn from unordered pairs?
    2. What additional insights come from sequential conversation ordering?
    3. How do character prompts and parameters enhance understanding?
    
    Example:
        >>> analyzer = ConversationAnalyzer(model_vocab_size=1000)
        >>> dataset = ConversationDataset.from_json_file('conversations.json')
        >>> analysis = analyzer.analyze(dataset)
        >>> print(analysis.summary())
    """
    
    def __init__(
        self,
        model_vocab_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        model_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the conversation analyzer.
        
        Args:
            model_vocab_size: Size of model vocabulary (if known)
            embedding_dim: Dimension of embeddings (if known)
            model_metadata: Additional model metadata
        """
        self.model_vocab_size = model_vocab_size
        self.embedding_dim = embedding_dim
        self.model_metadata = model_metadata or {}
    
    def analyze(self, dataset: ConversationDataset) -> 'AnalysisResult':
        """
        Perform comprehensive analysis of conversation dataset.
        
        Args:
            dataset: Conversation dataset to analyze
        
        Returns:
            AnalysisResult containing insights and statistics
        """
        result = AnalysisResult(dataset, self)
        
        # Basic statistics
        result.basic_stats = self._compute_basic_statistics(dataset)
        
        # Token-level analysis (vocabulary coverage, frequency)
        result.token_analysis = self._analyze_tokens(dataset)
        
        # What we can learn from unordered pairs
        result.unordered_insights = self._analyze_unordered_capabilities(dataset)
        
        # Additional insights from sequential ordering (if applicable)
        if dataset.is_sequential:
            result.sequential_insights = self._analyze_sequential_capabilities(dataset)
        
        # Insights from metadata (character prompts, parameters)
        if dataset.global_metadata or any(p.metadata for p in dataset.pairs):
            result.metadata_insights = self._analyze_metadata_capabilities(dataset)
        
        # Weight and activation inference limits
        result.inference_limits = self._analyze_inference_limits(dataset)
        
        return result
    
    def _compute_basic_statistics(self, dataset: ConversationDataset) -> Dict[str, Any]:
        """Compute basic dataset statistics."""
        query_lengths = [len(p.query.split()) for p in dataset.pairs]
        response_lengths = [len(p.response.split()) for p in dataset.pairs]
        
        return {
            'total_pairs': len(dataset),
            'is_sequential': dataset.is_sequential,
            'avg_query_length': np.mean(query_lengths) if query_lengths else 0,
            'avg_response_length': np.mean(response_lengths) if response_lengths else 0,
            'total_tokens': sum(query_lengths) + sum(response_lengths),
            'has_metadata': bool(dataset.global_metadata),
            'pairs_with_metadata': sum(1 for p in dataset.pairs if p.metadata),
        }
    
    def _analyze_tokens(self, dataset: ConversationDataset) -> Dict[str, Any]:
        """Analyze token-level patterns in the dataset."""
        all_tokens = []
        query_tokens = []
        response_tokens = []
        
        for pair in dataset.pairs:
            q_tokens = pair.query.split()
            r_tokens = pair.response.split()
            
            all_tokens.extend(q_tokens + r_tokens)
            query_tokens.extend(q_tokens)
            response_tokens.extend(r_tokens)
        
        token_counts = Counter(all_tokens)
        unique_tokens = len(token_counts)
        
        # Calculate vocabulary coverage if model vocab is known
        vocab_coverage = None
        if self.model_vocab_size:
            vocab_coverage = min(1.0, unique_tokens / self.model_vocab_size)
        
        return {
            'unique_tokens': unique_tokens,
            'total_token_instances': len(all_tokens),
            'vocab_coverage': vocab_coverage,
            'top_tokens': token_counts.most_common(20),
            'query_unique_tokens': len(set(query_tokens)),
            'response_unique_tokens': len(set(response_tokens)),
            'token_diversity': unique_tokens / len(all_tokens) if all_tokens else 0,
        }
    
    def _analyze_unordered_capabilities(
        self,
        dataset: ConversationDataset
    ) -> Dict[str, Any]:
        """
        Analyze what can be learned from unordered query-response pairs.
        
        With known model topology and vocabulary, unordered pairs can reveal:
        - Token co-occurrence patterns
        - Input-output relationships
        - Statistical correlations
        - But NOT causal dependencies or temporal dynamics
        """
        insights = {
            'description': 'Analysis of unordered conversation pairs',
            'learnable_aspects': [],
            'limitations': [],
        }
        
        # What we CAN learn
        insights['learnable_aspects'] = [
            {
                'aspect': 'Token Co-occurrence',
                'description': 'Which tokens appear together in queries and responses',
                'confidence': 'high',
                'reasoning': 'Direct observation from data'
            },
            {
                'aspect': 'Input-Output Mapping',
                'description': 'Statistical correlation between query and response patterns',
                'confidence': 'medium',
                'reasoning': 'Can observe correlations but not causation'
            },
            {
                'aspect': 'Vocabulary Usage',
                'description': 'Which parts of vocabulary are actively used',
                'confidence': 'high',
                'reasoning': 'Direct counting of token occurrences'
            },
            {
                'aspect': 'Response Style',
                'description': 'General characteristics of response generation',
                'confidence': 'medium',
                'reasoning': 'Statistical patterns in response structure'
            },
        ]
        
        # What we CANNOT learn without ordering
        insights['limitations'] = [
            {
                'limitation': 'Temporal Dependencies',
                'description': 'Cannot learn how context builds across conversation',
                'impact': 'high',
                'reasoning': 'No sequential information preserved'
            },
            {
                'limitation': 'Causal Relationships',
                'description': 'Cannot determine causation, only correlation',
                'impact': 'high',
                'reasoning': 'Need controlled experiments or sequential data'
            },
            {
                'limitation': 'Activation Patterns',
                'description': 'Cannot directly observe internal activations',
                'impact': 'critical',
                'reasoning': 'Only have inputs/outputs, not intermediate states'
            },
            {
                'limitation': 'Weight Values',
                'description': 'Cannot directly recover weight matrices',
                'impact': 'critical',
                'reasoning': 'Under-determined system (infinite solutions)'
            },
        ]
        
        # Compute actual co-occurrence patterns
        cooccurrence = self._compute_token_cooccurrence(dataset)
        insights['token_cooccurrence_sample'] = cooccurrence[:10]
        
        return insights
    
    def _analyze_sequential_capabilities(
        self,
        dataset: ConversationDataset
    ) -> Dict[str, Any]:
        """
        Analyze additional insights from sequential conversation ordering.
        
        Sequential ordering adds:
        - Context flow analysis
        - Topic evolution
        - Reference resolution patterns
        - Temporal dependencies
        """
        insights = {
            'description': 'Additional insights from sequential ordering',
            'additional_learnable_aspects': [],
            'enhanced_capabilities': [],
        }
        
        # Additional aspects we can learn with ordering
        insights['additional_learnable_aspects'] = [
            {
                'aspect': 'Context Accumulation',
                'description': 'How context builds across conversation turns',
                'confidence': 'high',
                'reasoning': 'Can track information flow across turns'
            },
            {
                'aspect': 'Topic Evolution',
                'description': 'How topics shift and develop in conversation',
                'confidence': 'high',
                'reasoning': 'Sequential analysis of topic changes'
            },
            {
                'aspect': 'Reference Patterns',
                'description': 'How pronouns and references relate to prior context',
                'confidence': 'medium',
                'reasoning': 'Can analyze anaphora resolution patterns'
            },
            {
                'aspect': 'Attention Patterns (Approximate)',
                'description': 'Approximate attention to previous turns',
                'confidence': 'low',
                'reasoning': 'Can infer which prior tokens are referenced'
            },
        ]
        
        # How this enhances our understanding
        insights['enhanced_capabilities'] = [
            'Better estimation of context window usage',
            'Understanding of multi-turn reasoning',
            'Identification of memory/forgetting patterns',
            'Analysis of consistency across turns',
        ]
        
        # Compute sequential statistics
        insights['sequential_stats'] = self._compute_sequential_statistics(dataset)
        
        return insights
    
    def _analyze_metadata_capabilities(
        self,
        dataset: ConversationDataset
    ) -> Dict[str, Any]:
        """
        Analyze insights from character prompts and parameters.
        
        Metadata (character prompts, temperature, etc.) provides:
        - Conditioning information
        - Generation parameters
        - Persona/style indicators
        """
        insights = {
            'description': 'Insights from metadata (prompts, parameters)',
            'metadata_types': [],
            'learnable_aspects': [],
        }
        
        # Identify available metadata
        metadata_keys = set()
        if dataset.global_metadata:
            metadata_keys.update(dataset.global_metadata.keys())
        for pair in dataset.pairs:
            if pair.metadata:
                metadata_keys.update(pair.metadata.keys())
        
        insights['metadata_types'] = list(metadata_keys)
        
        # What we can learn with metadata
        insights['learnable_aspects'] = [
            {
                'aspect': 'Conditioning Effects',
                'description': 'How character prompts influence response style',
                'confidence': 'medium',
                'reasoning': 'Can correlate metadata with response characteristics'
            },
            {
                'aspect': 'Parameter Sensitivity',
                'description': 'How generation parameters (temp, top_p) affect outputs',
                'confidence': 'medium',
                'reasoning': 'Can analyze output variation under different parameters'
            },
            {
                'aspect': 'Persona Consistency',
                'description': 'How well responses maintain character/style',
                'confidence': 'high',
                'reasoning': 'Direct comparison with character prompt'
            },
        ]
        
        # Enhanced understanding
        insights['enhanced_understanding'] = [
            'Separation of content from style',
            'Understanding of conditioning mechanisms',
            'Identification of prompt-sensitive regions',
            'Better approximation of embedding biases',
        ]
        
        return insights
    
    def _analyze_inference_limits(
        self,
        dataset: ConversationDataset
    ) -> Dict[str, Any]:
        """
        Analyze fundamental limits of what can be inferred about weights and activations.
        
        This addresses the core question: Given topology, vocabulary, and conversations,
        what are the theoretical and practical limits of inferring internal states?
        """
        limits = {
            'description': 'Theoretical and practical limits of inference',
            'theoretical_limits': [],
            'practical_constraints': [],
            'what_is_possible': [],
            'what_is_impossible': [],
        }
        
        # Theoretical limits
        limits['theoretical_limits'] = [
            {
                'limit': 'Under-determined System',
                'description': 'Many weight configurations can produce same outputs',
                'explanation': (
                    'With N parameters and M < N observations, the system is '
                    'under-determined. Infinite weight configurations exist that '
                    'match the observed input-output pairs.'
                ),
                'mathematical_basis': 'Linear algebra: rank(A) < num_parameters'
            },
            {
                'limit': 'Hidden Activations',
                'description': 'Internal activations are not directly observable',
                'explanation': (
                    'We only observe inputs and outputs. Intermediate layer '
                    'activations, attention patterns, and hidden states remain '
                    'hidden without access to model internals.'
                ),
                'workaround': 'Approximate through probing or gradient-based methods'
            },
            {
                'limit': 'Non-uniqueness of Representations',
                'description': 'Multiple internal representations can be equivalent',
                'explanation': (
                    'Transformations like rotation, scaling, or permutation of '
                    'hidden dimensions can produce identical outputs while having '
                    'different internal representations.'
                ),
                'implication': 'Cannot recover "the" weights, only "compatible" weights'
            },
        ]
        
        # Practical constraints
        data_size = len(dataset)
        limits['practical_constraints'] = [
            {
                'constraint': 'Limited Data',
                'description': f'Only {data_size} conversation pairs available',
                'impact': (
                    f'With typical LLM having millions of parameters, {data_size} '
                    'samples provide minimal constraints on weight space.'
                ),
            },
            {
                'constraint': 'No Gradient Information',
                'description': 'Cannot backpropagate or compute gradients',
                'impact': (
                    'Cannot use gradient-based probing or optimization to explore '
                    'weight space. Limited to statistical analysis.'
                ),
            },
            {
                'constraint': 'Discrete Tokens',
                'description': 'Working with discrete tokens, not continuous values',
                'impact': (
                    'Lose fine-grained information about probability distributions. '
                    'Can only see argmax outputs, not full logit distributions.'
                ),
            },
        ]
        
        # What IS possible
        limits['what_is_possible'] = [
            {
                'capability': 'Statistical Pattern Recognition',
                'description': 'Identify correlations and patterns in data',
                'examples': [
                    'Token co-occurrence frequencies',
                    'Query-response pair similarities',
                    'Topic distributions',
                    'Style characteristics',
                ],
            },
            {
                'capability': 'Behavioral Characterization',
                'description': 'Understand model behavior without knowing weights',
                'examples': [
                    'Response tendencies for query types',
                    'Vocabulary usage patterns',
                    'Context sensitivity (if sequential)',
                    'Consistency across similar inputs',
                ],
            },
            {
                'capability': 'Approximate Probing',
                'description': 'Train probe models to approximate internal states',
                'examples': [
                    'Linear probes for semantic properties',
                    'Clustering of embedding spaces',
                    'Topic models for attention patterns',
                ],
                'caveat': 'Requires additional model training on top of dataset'
            },
        ]
        
        # What is NOT possible
        limits['what_is_impossible'] = [
            {
                'impossibility': 'Exact Weight Recovery',
                'description': 'Cannot determine exact weight values',
                'reason': 'System is vastly under-determined',
            },
            {
                'impossibility': 'Activation Reconstruction',
                'description': 'Cannot reconstruct exact activation patterns',
                'reason': 'No access to intermediate computations',
            },
            {
                'impossibility': 'Causal Mechanism Identification',
                'description': 'Cannot identify exact causal mechanisms',
                'reason': 'Correlation does not imply causation',
            },
            {
                'impossibility': 'Complete Knowledge Extraction',
                'description': 'Cannot extract all knowledge encoded in model',
                'reason': 'Knowledge is distributed across parameters non-linearly',
            },
        ]
        
        # Summary with dataset context
        limits['summary'] = self._generate_inference_summary(dataset)
        
        return limits
    
    def _compute_token_cooccurrence(
        self,
        dataset: ConversationDataset
    ) -> List[Dict[str, Any]]:
        """Compute token co-occurrence patterns."""
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for pair in dataset.pairs:
            query_tokens = pair.query.split()
            response_tokens = pair.response.split()
            
            # Query-response co-occurrence
            for q_token in query_tokens:
                for r_token in response_tokens:
                    cooccurrence[q_token][r_token] += 1
        
        # Convert to sorted list
        results = []
        for q_token, r_tokens in cooccurrence.items():
            for r_token, count in r_tokens.items():
                results.append({
                    'query_token': q_token,
                    'response_token': r_token,
                    'count': count,
                })
        
        results.sort(key=lambda x: x['count'], reverse=True)
        return results
    
    def _compute_sequential_statistics(
        self,
        dataset: ConversationDataset
    ) -> Dict[str, Any]:
        """Compute statistics specific to sequential conversations."""
        if not dataset.is_sequential:
            return {}
        
        # Analyze context carry-over
        context_overlap = []
        for i in range(1, len(dataset.pairs)):
            prev_tokens = set(dataset.pairs[i-1].response.split())
            curr_tokens = set(dataset.pairs[i].query.split())
            
            if prev_tokens and curr_tokens:
                overlap = len(prev_tokens & curr_tokens) / len(prev_tokens | curr_tokens)
                context_overlap.append(overlap)
        
        return {
            'avg_context_overlap': np.mean(context_overlap) if context_overlap else 0,
            'conversation_length': len(dataset.pairs),
            'potential_context_window': len(dataset.pairs),
        }
    
    def _generate_inference_summary(
        self,
        dataset: ConversationDataset
    ) -> Dict[str, Any]:
        """Generate a summary of what can be inferred from this specific dataset."""
        data_size = len(dataset)
        
        # Estimate parameters (if model info available)
        estimated_params = None
        if self.model_vocab_size and self.embedding_dim:
            # Very rough estimate for a transformer
            estimated_params = (
                self.model_vocab_size * self.embedding_dim +  # Embeddings
                self.embedding_dim ** 2 * 12  # Rough transformer layer estimate
            )
        
        summary = {
            'dataset_size': data_size,
            'is_sequential': dataset.is_sequential,
            'has_metadata': bool(dataset.global_metadata),
        }
        
        if estimated_params:
            summary['parameter_to_data_ratio'] = estimated_params / data_size
            summary['interpretation'] = (
                f"With approximately {estimated_params:,} parameters and "
                f"{data_size} samples, the system is severely under-determined. "
                f"Each sample constrains ~{estimated_params / data_size:.0f} parameters "
                "on average, making exact weight recovery impossible."
            )
        else:
            summary['interpretation'] = (
                f"With {data_size} conversation pairs and no model architecture, "
                "can only perform statistical analysis of input-output patterns."
            )
        
        return summary


class AnalysisResult:
    """Container for analysis results."""
    
    def __init__(self, dataset: ConversationDataset, analyzer: ConversationAnalyzer):
        """Initialize analysis result."""
        self.dataset = dataset
        self.analyzer = analyzer
        
        self.basic_stats: Dict[str, Any] = {}
        self.token_analysis: Dict[str, Any] = {}
        self.unordered_insights: Dict[str, Any] = {}
        self.sequential_insights: Dict[str, Any] = {}
        self.metadata_insights: Dict[str, Any] = {}
        self.inference_limits: Dict[str, Any] = {}
    
    def summary(self) -> str:
        """Generate a human-readable summary of the analysis."""
        lines = [
            "=" * 80,
            "CONVERSATION DATA ANALYSIS SUMMARY",
            "=" * 80,
            "",
            "DATASET STATISTICS:",
            f"  Total conversation pairs: {self.basic_stats.get('total_pairs', 0)}",
            f"  Sequential ordering: {self.basic_stats.get('is_sequential', False)}",
            f"  Total tokens: {self.basic_stats.get('total_tokens', 0)}",
            f"  Average query length: {self.basic_stats.get('avg_query_length', 0):.1f} tokens",
            f"  Average response length: {self.basic_stats.get('avg_response_length', 0):.1f} tokens",
            "",
            "VOCABULARY ANALYSIS:",
            f"  Unique tokens observed: {self.token_analysis.get('unique_tokens', 0)}",
            f"  Token diversity: {self.token_analysis.get('token_diversity', 0):.3f}",
        ]
        
        if self.token_analysis.get('vocab_coverage') is not None:
            lines.append(
                f"  Model vocabulary coverage: {self.token_analysis['vocab_coverage']:.1%}"
            )
        
        lines.extend([
            "",
            "=" * 80,
            "WHAT CAN WE LEARN FROM THIS DATA?",
            "=" * 80,
        ])
        
        # Unordered insights
        if self.unordered_insights:
            lines.append("\nFrom UNORDERED query-response pairs:")
            for aspect in self.unordered_insights.get('learnable_aspects', []):
                lines.append(f"  ✓ {aspect['aspect']}: {aspect['description']}")
                lines.append(f"    Confidence: {aspect['confidence']}")
        
        # Sequential insights
        if self.sequential_insights:
            lines.append("\nADDITIONAL insights from SEQUENTIAL ordering:")
            for aspect in self.sequential_insights.get('additional_learnable_aspects', []):
                lines.append(f"  ✓ {aspect['aspect']}: {aspect['description']}")
                lines.append(f"    Confidence: {aspect['confidence']}")
        
        # Metadata insights
        if self.metadata_insights:
            lines.append("\nADDITIONAL insights from METADATA (prompts, parameters):")
            for aspect in self.metadata_insights.get('learnable_aspects', []):
                lines.append(f"  ✓ {aspect['aspect']}: {aspect['description']}")
        
        lines.extend([
            "",
            "=" * 80,
            "FUNDAMENTAL LIMITS",
            "=" * 80,
        ])
        
        # Theoretical limits
        if self.inference_limits:
            lines.append("\nTheoretical limits:")
            for limit in self.inference_limits.get('theoretical_limits', [])[:3]:
                lines.append(f"  ✗ {limit['limit']}")
                lines.append(f"    {limit['explanation']}")
                lines.append("")
            
            # Summary
            summary = self.inference_limits.get('summary', {})
            if 'interpretation' in summary:
                lines.append("CONCLUSION:")
                lines.append(f"  {summary['interpretation']}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            'basic_stats': self.basic_stats,
            'token_analysis': self.token_analysis,
            'unordered_insights': self.unordered_insights,
            'sequential_insights': self.sequential_insights,
            'metadata_insights': self.metadata_insights,
            'inference_limits': self.inference_limits,
        }
    
    def to_json_file(self, filepath: str):
        """Save analysis to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
