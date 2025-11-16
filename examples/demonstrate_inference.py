#!/usr/bin/env python
"""
Comprehensive demonstration of TinyTransformer inference implementations.

This script shows all different implementation approaches and compares them:
1. List-based (simple arrays)
2. Dict-based (structured data)
3. Class-based (OOP with dataclasses)
4. Functional (pure functions)

Run this to see how the tiny transformer processes input step-by-step.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf_workbench.inference import (
    TinyTransformerListBased,
    TinyTransformerDictBased,
    TinyTransformerClassBased,
    tiny_transformer_functional,
    InferenceTracer,
)
from gguf_workbench.inference.functional import (
    load_model_weights_functional,
    trace_inference_functional,
    create_inference_function,
)


def demo_list_based():
    """Demonstrate list-based implementation."""
    print("=" * 80)
    print("1. LIST-BASED IMPLEMENTATION")
    print("=" * 80)
    print("\nThis is the simplest implementation using only Python lists.")
    print("It shows explicit indexing and loops.\n")
    
    model_path = Path(__file__).parent.parent / "tinytf" / "tiny_model_gguf.json"
    model = TinyTransformerListBased.from_json(str(model_path))
    
    # Simple inference
    tokens = [0, 1, 2]
    print(f"Input tokens: {tokens}")
    print(f"Token names: {[f'token_{t}' for t in tokens]}")
    print()
    
    # Forward pass with trace
    logits = model.forward(tokens, trace=True)
    
    print("\n" + "=" * 80)
    print("Inference complete!")
    print()


def demo_dict_based():
    """Demonstrate dict-based implementation."""
    print("\n\n")
    print("=" * 80)
    print("2. DICT-BASED IMPLEMENTATION")
    print("=" * 80)
    print("\nThis implementation uses dictionaries with named keys")
    print("for better organization and readability.\n")
    
    model_path = Path(__file__).parent.parent / "tinytf" / "tiny_model_gguf.json"
    model = TinyTransformerDictBased.from_json(str(model_path))
    
    tokens = [0, 1, 2]
    print(f"Input tokens: {tokens}")
    print()
    
    # Get full result dictionary
    result = model.forward(tokens, trace=True)
    
    # Show the structured output
    print("\n" + "-" * 80)
    print("Result structure:")
    print(f"  Keys: {list(result.keys())}")
    print(f"  Predictions: {result['predictions']}")
    print()
    
    # Generate detailed report
    print("\n" + "-" * 80)
    print("Detailed computation trace:")
    print(model.inspect_computation(tokens))
    
    print("\n" + "=" * 80)
    print("Inference complete!")
    print()


def demo_class_based():
    """Demonstrate class-based implementation."""
    print("\n\n")
    print("=" * 80)
    print("3. CLASS-BASED IMPLEMENTATION")
    print("=" * 80)
    print("\nThis implementation uses dataclasses for type safety")
    print("and clean object-oriented design.\n")
    
    model_path = Path(__file__).parent.parent / "tinytf" / "tiny_model_gguf.json"
    model = TinyTransformerClassBased.from_json(str(model_path))
    
    tokens = [0, 1, 2]
    print(f"Input tokens: {tokens}")
    print()
    
    # Forward pass
    predictions = model.forward(tokens, trace=True)
    
    # Show prediction objects
    print("\n" + "-" * 80)
    print("Prediction objects:")
    for pred in predictions:
        print(f"  Position {pred.position}:")
        print(f"    Predicted: {pred.predicted_token_name}")
        print(f"    Confidence: {pred.logits[pred.predicted_token_id]:.4f}")
    print()
    
    # Get attention matrix
    print("\n" + "-" * 80)
    print("Attention matrix:")
    attn_matrix = model.get_attention_matrix(tokens)
    for i, row in enumerate(attn_matrix):
        print(f"  Position {i}: {[f'{w:.4f}' for w in row]}")
    print()
    
    # Explain a specific prediction
    print("\n" + "-" * 80)
    explanation = model.explain_prediction(tokens, position=1)
    print(explanation)
    
    print("\n" + "=" * 80)
    print("Inference complete!")
    print()


def demo_functional():
    """Demonstrate functional implementation."""
    print("\n\n")
    print("=" * 80)
    print("4. FUNCTIONAL IMPLEMENTATION")
    print("=" * 80)
    print("\nThis implementation uses pure functions with no side effects.")
    print("All functions are composable and testable in isolation.\n")
    
    model_path = Path(__file__).parent.parent / "tinytf" / "tiny_model_gguf.json"
    
    # Load weights
    (
        embedding_weights,
        query_weights,
        key_weights,
        value_weights,
        output_weights,
        embed_dim,
    ) = load_model_weights_functional(str(model_path))
    
    tokens = [0, 1, 2]
    print(f"Input tokens: {tokens}")
    print()
    
    # Simple inference
    predictions = tiny_transformer_functional(
        tokens,
        embedding_weights,
        query_weights,
        key_weights,
        value_weights,
        output_weights,
        embed_dim,
        return_details=False,
    )
    
    print(f"Predictions: {predictions}")
    print(f"Predicted tokens: {[f'token_{p}' for p in predictions]}")
    print()
    
    # Detailed trace
    print("\n" + "-" * 80)
    trace = trace_inference_functional(tokens, str(model_path))
    print(trace)
    
    # Create a specialized inference function
    print("\n" + "-" * 80)
    print("Creating specialized inference function...")
    inference_fn = create_inference_function(str(model_path))
    
    # Use it
    new_tokens = [5, 6, 7]
    new_predictions = inference_fn(new_tokens)
    print(f"New input: {new_tokens}")
    print(f"Predictions: {[f'token_{p}' for p in new_predictions]}")
    
    print("\n" + "=" * 80)
    print("Inference complete!")
    print()


def demo_comparison():
    """Compare all implementations."""
    print("\n\n")
    print("=" * 80)
    print("5. IMPLEMENTATION COMPARISON")
    print("=" * 80)
    print()
    
    model_path = Path(__file__).parent.parent / "tinytf" / "tiny_model_gguf.json"
    tokens = [0, 1, 2]
    
    # List-based
    model_list = TinyTransformerListBased.from_json(str(model_path))
    logits_list = model_list.forward(tokens, trace=False)
    preds_list = [logit_vec.index(max(logit_vec)) for logit_vec in logits_list]
    
    # Dict-based
    model_dict = TinyTransformerDictBased.from_json(str(model_path))
    result_dict = model_dict.forward(tokens, trace=False)
    preds_dict = result_dict['predictions']
    
    # Class-based
    model_class = TinyTransformerClassBased.from_json(str(model_path))
    result_class = model_class.forward(tokens, trace=False)
    preds_class = [pred.predicted_token_id for pred in result_class]
    
    # Functional
    (
        embedding_weights,
        query_weights,
        key_weights,
        value_weights,
        output_weights,
        embed_dim,
    ) = load_model_weights_functional(str(model_path))
    
    preds_func = tiny_transformer_functional(
        tokens,
        embedding_weights,
        query_weights,
        key_weights,
        value_weights,
        output_weights,
        embed_dim,
        return_details=False,
    )
    
    # Compare
    print(f"Input: {tokens}\n")
    print("Predictions from each implementation:")
    print(f"  List-based:       {preds_list}")
    print(f"  Dict-based:       {preds_dict}")
    print(f"  Class-based:      {preds_class}")
    print(f"  Functional:       {preds_func}")
    print()
    
    # Verify they're all the same
    all_same = (
        preds_list == preds_dict == preds_class == preds_func
    )
    
    if all_same:
        print("✓ All implementations produce identical results!")
    else:
        print("✗ Warning: Implementations produce different results!")
    
    print()
    print("=" * 80)
    print()


def demo_analysis():
    """Demonstrate analysis capabilities."""
    print("\n\n")
    print("=" * 80)
    print("6. INFERENCE ANALYSIS")
    print("=" * 80)
    print("\nShowing how to analyze and understand inference in detail.\n")
    
    model_path = Path(__file__).parent.parent / "tinytf" / "tiny_model_gguf.json"
    model = TinyTransformerClassBased.from_json(str(model_path))
    
    # Example: Understanding attention patterns
    tokens = [0, 1, 2, 3, 4]
    print(f"Input sequence: {tokens}")
    print(f"Token names: {[f'token_{t}' for t in tokens]}")
    print()
    
    # Get attention matrix
    attn_matrix = model.get_attention_matrix(tokens)
    
    print("Attention Pattern Visualization:")
    print("(Each row shows how much that position attends to others)")
    print()
    
    # Create simple ASCII visualization
    print("      ", end="")
    for j in range(len(tokens)):
        print(f"  Pos{j}", end="")
    print()
    
    for i, row in enumerate(attn_matrix):
        print(f"Pos{i} ", end="")
        for weight in row:
            # Use bar chart characters
            if weight > 0.3:
                char = "█"
            elif weight > 0.15:
                char = "▓"
            elif weight > 0.05:
                char = "░"
            else:
                char = "·"
            print(f" {char}{weight:.2f}", end="")
        print()
    
    print()
    print("Legend: █ Strong (>0.3)  ▓ Medium (>0.15)  ░ Weak (>0.05)  · Very weak")
    print()
    
    print("=" * 80)
    print()


def main():
    """Run all demonstrations."""
    print("\n" * 2)
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  TINY TRANSFORMER INFERENCE IMPLEMENTATIONS DEMONSTRATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Showing multiple approaches to implement transformer inference".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    print("This demonstration shows 4 different ways to implement inference:")
    print("  1. List-based    - Simple, explicit, using only Python lists")
    print("  2. Dict-based    - Organized with named dictionaries")
    print("  3. Class-based   - Object-oriented with type hints")
    print("  4. Functional    - Pure functions with no side effects")
    print()
    
    # Run demos
    try:
        demo_list_based()
        
        demo_dict_based()
        
        demo_class_based()
        
        demo_functional()
        
        demo_comparison()
        
        demo_analysis()
        
    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Final summary
    print("\n" * 2)
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("All four implementations demonstrate the same transformer inference")
    print("but with different programming paradigms:")
    print()
    print("✓ List-based:    Best for learning, shows explicit operations")
    print("✓ Dict-based:    Best for inspection, named components")
    print("✓ Class-based:   Best for production, type-safe and maintainable")
    print("✓ Functional:    Best for testing, pure and composable")
    print()
    print("Choose the approach that best fits your needs!")
    print()
    print("=" * 80)
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
