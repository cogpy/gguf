#!/usr/bin/env python3
"""
Example: Vocabulary and Layer Activation Analysis

Demonstrates the enhanced vocabulary analysis features that help understand:
1. Complete vocabulary enumeration with word counts
2. Relationship between expressed and total available vocabulary
3. Which layers and activations are likely involved
4. Deep tree echo state reservoir computing framework interactions

This addresses the question: "if we analyze the deep tree echo conversation record 
to enumerate the whole vocabulary with word counts where the models are indicated 
in the json assistant record for each response, what further insights can we gain 
about how the expressed vocabulary relates to the total available vocabulary and 
which layers and activations likely reflect the interaction with other dynamic 
persona variables related to the deep tree echo state reservoir computing framework 
interacting with the transformer etc"
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import gguf_workbench
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf_workbench.vocabulary_analysis import (
    VocabularyAnalyzer,
    load_conversations_from_json
)


def create_sample_conversations():
    """Create sample conversations with model metadata for demonstration."""
    return [
        {
            "query": "What is a transformer model?",
            "response": "A transformer is a neural network architecture that uses self-attention mechanisms to process sequential data. It was introduced in the 'Attention is All You Need' paper.",
            "model": "gpt-4"
        },
        {
            "query": "How does self-attention work?",
            "response": "Self-attention allows each token in a sequence to attend to all other tokens. It computes attention weights that determine how much focus to place on different parts of the input when processing each token.",
            "model": "gpt-4"
        },
        {
            "query": "Explain the echo state network",
            "response": "An echo state network is a type of recurrent neural network with a sparsely connected hidden layer (the reservoir). The reservoir has fixed random weights, and only the output layer is trained. This creates a rich temporal representation.",
            "model": "gpt-4"
        },
        {
            "query": "How do transformers and reservoir computing relate?",
            "response": "Transformers and reservoir computing both process temporal sequences but differently. Transformers use learned attention weights across the sequence, while reservoir computing uses fixed recurrent dynamics. Combining them creates a hybrid where the reservoir provides rich temporal features and the transformer layers perform learned readout and composition.",
            "model": "gpt-4"
        },
        {
            "query": "What are dynamic persona variables?",
            "response": "Dynamic persona variables represent adaptable characteristics in conversation models. They might include formality level, technical depth, emotional tone, or cognitive style. These variables can shift based on context and user interaction, creating more natural and adaptive responses.",
            "model": "gpt-4"
        },
    ]


def main():
    """Run the vocabulary and layer activation analysis demo."""
    
    print("=" * 80)
    print("VOCABULARY AND LAYER ACTIVATION ANALYSIS DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create or load sample conversations
    conversations = create_sample_conversations()
    
    print(f"Loaded {len(conversations)} conversations with model metadata")
    print()
    
    # Initialize analyzer with model configuration
    # These would typically come from the GGUF file metadata
    analyzer = VocabularyAnalyzer(
        total_vocab_size=50257,  # Example: GPT-2/GPT-4 vocabulary size
        model_layers=32,          # Example: Number of transformer layers
        embedding_dim=768,        # Example: Embedding dimension
        architecture="transformer"
    )
    
    print("-" * 80)
    print("STEP 1: Enumerate Complete Vocabulary with Word Counts")
    print("-" * 80)
    print()
    
    vocab_stats = analyzer.enumerate_vocabulary(conversations, include_metadata=True)
    
    print(f"Total unique words found: {vocab_stats['total_unique_words']}")
    print(f"Total word instances: {vocab_stats['total_word_instances']}")
    print(f"Vocabulary diversity: {vocab_stats['vocabulary_diversity']:.4f}")
    print()
    
    print("Vocabulary Coverage:")
    coverage = vocab_stats.get('vocabulary_coverage', {})
    if coverage:
        print(f"  Expressed vocabulary: {coverage['expressed_vocab_size']} words")
        print(f"  Total available vocabulary: {coverage['total_vocab_size']} words")
        print(f"  Coverage ratio: {coverage['coverage_ratio']:.4f}")
        print(f"  Coverage percentage: {coverage['coverage_percentage']:.2f}%")
        print(f"  Unexpressed vocabulary: {coverage['unexpressed_vocab_size']} words")
    print()
    
    print("Top 20 most frequent words:")
    for i, (word, count) in enumerate(list(vocab_stats['word_counts'].items())[:20], 1):
        print(f"  {i:2d}. '{word}': {count} occurrences")
    print()
    
    # Show per-model vocabulary if available
    if 'per_model_vocabulary' in vocab_stats:
        print("Per-Model Vocabulary Analysis:")
        for model, model_stats in vocab_stats['per_model_vocabulary'].items():
            print(f"  Model: {model}")
            print(f"    Unique words: {model_stats['unique_words']}")
            print(f"    Total instances: {model_stats['total_instances']}")
            print(f"    Top 5 words: {list(model_stats['top_words'].items())[:5]}")
        print()
    
    print("-" * 80)
    print("STEP 2: Analyze Layer Activations Based on Vocabulary Patterns")
    print("-" * 80)
    print()
    
    layer_insights = analyzer.analyze_layer_activations(vocab_stats, conversations)
    
    print("Embedding Layer Analysis:")
    embedding = layer_insights['embedding_layer_analysis']
    print(f"  Active embeddings: {embedding['active_embeddings']}")
    print(f"  Total embeddings: {embedding['total_embeddings']}")
    print(f"  Activation ratio: {embedding['activation_ratio']}")
    print(f"  Interpretation: {embedding['interpretation']}")
    print()
    
    print("Layer Activation Estimates:")
    for estimate in layer_insights.get('layer_activation_estimates', []):
        print(f"  Layers {estimate['layer_range']}: {estimate['layer_type']}")
        print(f"    Function: {estimate['primary_function']}")
        print(f"    Activation level: {estimate['activation_level']}")
        print(f"    Evidence: {estimate['evidence']}")
        print()
    
    print("Attention Head Insights:")
    for pattern in layer_insights.get('attention_head_insights', []):
        print(f"  {pattern['pattern_type']}")
        print(f"    Description: {pattern['description']}")
        print(f"    Likelihood: {pattern['likelihood']}")
        print(f"    Reasoning: {pattern['reasoning']}")
        print()
    
    print("Output Layer Analysis:")
    output = layer_insights['output_layer_analysis']
    print(f"  Output vocabulary size: {output['output_vocabulary_size']}")
    print(f"  Query-to-response ratio: {output['query_to_response_vocab_ratio']:.2f}")
    print(f"  Interpretation: {output['interpretation']}")
    print()
    
    print("-" * 80)
    print("STEP 3: Analyze Echo State Reservoir Computing Interactions")
    print("-" * 80)
    print()
    
    echo_analysis = analyzer.analyze_echo_reservoir_interaction(
        vocab_stats,
        layer_insights,
        conversations
    )
    
    print("Reservoir Dynamics:")
    dynamics = echo_analysis['reservoir_dynamics']
    for key, value in dynamics.items():
        print(f"  {key.replace('_', ' ').title()}:")
        print(f"    {value}")
    print()
    
    print("Feedback Loop Analysis:")
    feedback = echo_analysis['feedback_loops']
    print(f"  Echo ratio: {feedback['echo_ratio']:.2%}")
    print(f"  Feedback strength: {feedback['feedback_strength']}")
    print(f"  Interpretation: {feedback['interpretation']}")
    print(f"  Mechanism: {feedback['mechanism']}")
    print()
    
    print("Dynamic Persona Variables:")
    persona = echo_analysis['persona_variables']
    print(f"  Dominant persona: {persona['dominant_persona']}")
    print(f"  Persona diversity: {persona['persona_diversity']} dimensions detected")
    print(f"  Total persona markers: {persona['total_persona_markers']}")
    print(f"  Interpretation: {persona['interpretation']}")
    print()
    print("  Persona dimension scores:")
    for dimension, score in sorted(persona['persona_dimensions_detected'].items(), 
                                  key=lambda x: x[1], reverse=True):
        if score > 0:
            print(f"    {dimension.replace('_', ' ').title()}: {score}")
    print()
    
    print("Temporal Vocabulary Patterns:")
    temporal = echo_analysis['temporal_patterns']
    if 'pattern' in temporal:
        print(f"  Pattern: {temporal['pattern']}")
        if 'interpretation' in temporal:
            print(f"  Interpretation: {temporal['interpretation']}")
        if 'vocabulary_growth' in temporal:
            print(f"  Vocabulary growth: {temporal['vocabulary_growth']:.1f} words")
    print()
    
    print("Recommendations:")
    for i, recommendation in enumerate(echo_analysis['recommendations'], 1):
        print(f"  {i}. {recommendation}")
    print()
    
    print("-" * 80)
    print("STEP 4: Generate Comprehensive Report")
    print("-" * 80)
    print()
    
    full_report = analyzer.generate_comprehensive_report(conversations)
    
    print(full_report['summary'])
    print()
    
    # Save the full report
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "vocabulary_layer_analysis.json"
    with open(report_file, 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print(f"✓ Full analysis saved to: {report_file}")
    print()
    
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("1. VOCABULARY COVERAGE:")
    print("   The analysis shows what fraction of the model's total vocabulary")
    print("   is actually expressed in the conversations. Low coverage suggests")
    print("   the model has much more expressive capacity than demonstrated.")
    print()
    print("2. LAYER ACTIVATION INFERENCE:")
    print("   Based on vocabulary complexity and patterns, we can estimate which")
    print("   transformer layers are most active. All layers participate, but")
    print("   middle layers handle the most complex semantic processing.")
    print()
    print("3. ECHO STATE RESERVOIR DYNAMICS:")
    print("   The reservoir maintains temporal context through recurrent dynamics.")
    print("   Transformer layers then perform learned readout from these rich")
    print("   temporal states, combining the benefits of both approaches.")
    print()
    print("4. DYNAMIC PERSONA VARIABLES:")
    print("   Vocabulary analysis reveals which persona dimensions are active")
    print("   (formal vs. casual, technical vs. general, etc.). These variables")
    print("   are reflected in word choice and likely modulate both reservoir")
    print("   state and transformer attention patterns.")
    print()
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("To analyze your own conversation data:")
    print("  1. Prepare conversations in JSON format with 'query', 'response',")
    print("     and optional 'model' metadata fields")
    print("  2. Initialize VocabularyAnalyzer with your model configuration")
    print("  3. Call generate_comprehensive_report(conversations)")
    print("  4. Review the detailed analysis")


if __name__ == "__main__":
    main()
