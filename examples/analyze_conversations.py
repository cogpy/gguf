#!/usr/bin/env python3
"""
Example: Analyzing conversation datasets to understand what can be learned
about model weights and activations.

This example demonstrates:
1. Creating a conversation dataset
2. Analyzing unordered query-response pairs
3. Analyzing sequential conversations
4. Understanding the limits of inference from conversation data
"""

import json
from pathlib import Path
from gguf_workbench import (
    ConversationPair,
    ConversationDataset,
    ConversationAnalyzer,
)


def create_sample_dataset_unordered():
    """Create a sample unordered conversation dataset."""
    pairs = [
        ConversationPair(
            query="What is the capital of France?",
            response="The capital of France is Paris."
        ),
        ConversationPair(
            query="How do I make a cake?",
            response="To make a cake, you need flour, eggs, sugar, and butter. Mix them together and bake."
        ),
        ConversationPair(
            query="What is machine learning?",
            response="Machine learning is a subset of AI that enables systems to learn from data."
        ),
        ConversationPair(
            query="Tell me about Python",
            response="Python is a high-level programming language known for its simplicity and versatility."
        ),
        ConversationPair(
            query="What's the weather like?",
            response="I don't have access to real-time weather data, but I can help you find weather services."
        ),
    ]
    
    return ConversationDataset(
        pairs=pairs,
        is_sequential=False,
        global_metadata={
            'source': 'example',
            'created': '2024-01-01',
        }
    )


def create_sample_dataset_sequential():
    """Create a sample sequential conversation dataset."""
    pairs = [
        ConversationPair(
            query="Hi! Can you help me learn about transformers?",
            response="Of course! I'd be happy to help you learn about transformers. They're a type of neural network architecture.",
            position=0
        ),
        ConversationPair(
            query="What makes them different from other neural networks?",
            response="Transformers use self-attention mechanisms instead of recurrence. This allows them to process all positions in parallel.",
            position=1
        ),
        ConversationPair(
            query="Can you explain self-attention?",
            response="Self-attention lets each position attend to all positions in the input. It computes attention weights to determine which parts to focus on.",
            position=2
        ),
        ConversationPair(
            query="That's interesting! What are they used for?",
            response="Transformers are used for many NLP tasks like translation, text generation, and question answering. Models like GPT and BERT are transformers.",
            position=3
        ),
        ConversationPair(
            query="Thanks! This is really helpful.",
            response="You're welcome! Feel free to ask if you have more questions about transformers or machine learning!",
            position=4
        ),
    ]
    
    return ConversationDataset(
        pairs=pairs,
        is_sequential=True,
        global_metadata={
            'source': 'example',
            'created': '2024-01-01',
            'conversation_id': 'conv_001',
        }
    )


def create_sample_dataset_with_metadata():
    """Create a dataset with character prompts and parameters."""
    pairs = [
        ConversationPair(
            query="Tell me a joke",
            response="Why did the programmer quit? Because they didn't get arrays! 😄",
            metadata={
                'character_prompt': 'You are a friendly, humorous assistant',
                'temperature': 0.9,
                'top_p': 0.95,
            },
            position=0
        ),
        ConversationPair(
            query="Another one please!",
            response="What's a programmer's favorite hangout spot? The Foo Bar! 🍻",
            metadata={
                'character_prompt': 'You are a friendly, humorous assistant',
                'temperature': 0.9,
                'top_p': 0.95,
            },
            position=1
        ),
        ConversationPair(
            query="Tell me about recursion",
            response="To understand recursion, you must first understand recursion. 😉 But seriously, it's when a function calls itself!",
            metadata={
                'character_prompt': 'You are a friendly, humorous assistant',
                'temperature': 0.9,
                'top_p': 0.95,
            },
            position=2
        ),
    ]
    
    return ConversationDataset(
        pairs=pairs,
        is_sequential=True,
        global_metadata={
            'model': 'gpt-4',
            'character_prompt': 'You are a friendly, humorous assistant who loves programming jokes',
            'default_temperature': 0.9,
        }
    )


def main():
    """Run the conversation analysis examples."""
    print("\n" + "=" * 80)
    print("CONVERSATION DATA ANALYSIS DEMO")
    print("=" * 80)
    
    # Example 1: Unordered dataset analysis
    print("\n" + "-" * 80)
    print("EXAMPLE 1: Analyzing Unordered Query-Response Pairs")
    print("-" * 80)
    
    dataset_unordered = create_sample_dataset_unordered()
    analyzer = ConversationAnalyzer(
        model_vocab_size=50000,  # Example: GPT-2 vocab size
        embedding_dim=768,
    )
    
    result_unordered = analyzer.analyze(dataset_unordered)
    print(result_unordered.summary())
    
    # Save to file
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    result_unordered.to_json_file("analysis_output/unordered_analysis.json")
    print(f"\n✓ Saved detailed analysis to analysis_output/unordered_analysis.json")
    
    # Example 2: Sequential dataset analysis
    print("\n" + "-" * 80)
    print("EXAMPLE 2: Analyzing Sequential Conversation")
    print("-" * 80)
    
    dataset_sequential = create_sample_dataset_sequential()
    result_sequential = analyzer.analyze(dataset_sequential)
    print(result_sequential.summary())
    
    result_sequential.to_json_file("analysis_output/sequential_analysis.json")
    print(f"\n✓ Saved detailed analysis to analysis_output/sequential_analysis.json")
    
    # Example 3: Dataset with metadata
    print("\n" + "-" * 80)
    print("EXAMPLE 3: Analyzing Conversations with Character Prompts")
    print("-" * 80)
    
    dataset_metadata = create_sample_dataset_with_metadata()
    result_metadata = analyzer.analyze(dataset_metadata)
    print(result_metadata.summary())
    
    result_metadata.to_json_file("analysis_output/metadata_analysis.json")
    print(f"\n✓ Saved detailed analysis to analysis_output/metadata_analysis.json")
    
    # Example 4: Compare what we can learn
    print("\n" + "=" * 80)
    print("KEY INSIGHTS: Comparing Different Analysis Scenarios")
    print("=" * 80)
    
    print("\n1. UNORDERED PAIRS:")
    print("   - Can learn: Token co-occurrence, vocabulary usage")
    print("   - Cannot learn: Context flow, temporal dependencies")
    print(f"   - Data size: {len(dataset_unordered)} pairs")
    
    print("\n2. SEQUENTIAL ORDERING:")
    print("   - Additional capability: Context accumulation, topic evolution")
    print("   - Can approximate: Attention to previous turns")
    print(f"   - Data size: {len(dataset_sequential)} pairs")
    
    print("\n3. WITH METADATA (CHARACTER PROMPTS):")
    print("   - Additional capability: Conditioning effects, persona consistency")
    print("   - Can separate: Content from style")
    print(f"   - Data size: {len(dataset_metadata)} pairs")
    
    print("\n4. FUNDAMENTAL LIMITS:")
    print("   - CANNOT recover exact weights (under-determined system)")
    print("   - CANNOT observe internal activations (hidden states)")
    print("   - CAN characterize behavior and statistical patterns")
    print("   - CAN approximate semantic properties through probing")
    
    # Example 5: Save sample datasets
    print("\n" + "-" * 80)
    print("Saving sample datasets...")
    print("-" * 80)
    
    dataset_unordered.to_json_file("analysis_output/sample_unordered.json")
    dataset_sequential.to_json_file("analysis_output/sample_sequential.json")
    dataset_metadata.to_json_file("analysis_output/sample_with_metadata.json")
    
    print("✓ Saved sample_unordered.json")
    print("✓ Saved sample_sequential.json")
    print("✓ Saved sample_with_metadata.json")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nAll analysis results saved to analysis_output/ directory")
    print("You can inspect the JSON files for detailed analysis.")
    print("\nTo analyze your own dataset:")
    print("  1. Create a ConversationDataset from your data")
    print("  2. Initialize a ConversationAnalyzer with model info")
    print("  3. Call analyzer.analyze(dataset)")
    print("  4. Review the AnalysisResult")


if __name__ == "__main__":
    main()
