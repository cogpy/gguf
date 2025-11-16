#!/usr/bin/env python
"""
Generate and compare different representations of the TinyTransformer model.

This script demonstrates various representation formats and their trade-offs.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf_workbench.representations import (  # noqa: E402
    HypergraphRepresentation,
    GraphRepresentation,
    SymbolicRepresentation,
    RepresentationComparator,
)


def main():
    """Generate all representations and comparison."""
    print("=" * 80)
    print("TinyTransformer Model Representations")
    print("=" * 80)
    print()

    output_dir = Path(__file__).parent.parent / "tinytf" / "representations"
    output_dir.mkdir(exist_ok=True)

    # 1. Generate Hypergraph Representation
    print("1. Generating Hypergraph Representation...")
    hypergraph = HypergraphRepresentation.from_tiny_transformer()

    # Save hypergraph outputs
    hg_json_path = output_dir / "tiny_transformer_hypergraph.json"
    hypergraph.to_json(hg_json_path)
    print(f"   ✓ Saved to {hg_json_path}")

    hg_dot_path = output_dir / "tiny_transformer_hypergraph.dot"
    hypergraph.export_graphviz(hg_dot_path)
    print(f"   ✓ Saved Graphviz DOT to {hg_dot_path}")

    # Print hypergraph statistics
    stats = hypergraph.get_statistics()
    print(f"   ✓ Vertices: {stats['vertex_count']}")
    print(f"   ✓ Hyperedges: {stats['hyperedge_count']}")
    print(f"   ✓ Avg hyperedge size: {stats['average_hyperedge_size']:.2f}")
    print()

    # 2. Generate DAG Representation
    print("2. Generating DAG (Directed Graph) Representation...")
    graph = GraphRepresentation.from_tiny_transformer()

    dag_json_path = output_dir / "tiny_transformer_dag.json"
    graph.to_json(dag_json_path)
    print(f"   ✓ Saved to {dag_json_path}")

    dag_dot_path = output_dir / "tiny_transformer_dag.dot"
    graph.export_graphviz(dag_dot_path)
    print(f"   ✓ Saved Graphviz DOT to {dag_dot_path}")

    stats = graph.get_statistics()
    print(f"   ✓ Nodes: {stats['node_count']}")
    print(f"   ✓ Edges: {stats['edge_count']}")
    print(f"   ✓ Avg in-degree: {stats['average_in_degree']:.2f}")
    print()

    # 3. Generate Symbolic Representation
    print("3. Generating Symbolic/Algebraic Representation...")
    symbolic = SymbolicRepresentation.from_tiny_transformer()

    sym_json_path = output_dir / "tiny_transformer_symbolic.json"
    symbolic.to_json(sym_json_path)
    print(f"   ✓ Saved to {sym_json_path}")

    sym_md_path = output_dir / "tiny_transformer_symbolic.md"
    symbolic.export_markdown(sym_md_path)
    print(f"   ✓ Saved Markdown to {sym_md_path}")

    sym_latex_path = output_dir / "tiny_transformer_symbolic.tex"
    symbolic.to_latex(sym_latex_path)
    print(f"   ✓ Saved LaTeX to {sym_latex_path}")

    stats = symbolic.get_statistics()
    print(f"   ✓ Parameters: {stats['parameter_count']}")
    print(f"   ✓ Expressions: {stats['expression_count']}")
    print(f"   ✓ Total parameters: {stats['total_parameters']}")
    print()

    # 4. Generate Comparison
    print("4. Generating Representation Comparison...")
    comparator = RepresentationComparator.create_default_comparison()

    comp_json_path = output_dir / "representation_comparison.json"
    comparator.to_json(comp_json_path)
    print(f"   ✓ Saved to {comp_json_path}")

    comp_md_path = output_dir / "representation_comparison.md"
    comparator.to_markdown(comp_md_path)
    print(f"   ✓ Saved Markdown to {comp_md_path}")

    comparison = comparator.compare_all()
    print(f"   ✓ Best overall: {comparison['summary']['best_overall']}")
    print(f"   ✓ Best transparency: {comparison['summary']['best_transparency']}")
    print(f"   ✓ Best efficiency: {comparison['summary']['best_efficiency']}")
    print()

    # 5. Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("All representations have been generated in:")
    print(f"  {output_dir}")
    print()
    print("Files created:")
    print("  - Hypergraph: JSON, DOT")
    print("  - DAG: JSON, DOT")
    print("  - Symbolic: JSON, Markdown, LaTeX")
    print("  - Comparison: JSON, Markdown")
    print()
    print("To visualize DOT files:")
    print(f"  dot -Tpng {hg_dot_path} -o hypergraph.png")
    print(f"  dot -Tpng {dag_dot_path} -o dag.png")
    print()
    print("To compile LaTeX:")
    print(f"  pdflatex {sym_latex_path}")
    print()

    # Display a snippet of hypergraph
    print("=" * 80)
    print("Hypergraph Sample - Attention Operation")
    print("=" * 80)
    print()

    # Find and display attention-related hyperedge
    for edge_id, edge in hypergraph.hyperedges.items():
        if "attention_scores" in edge_id:
            print(f"Hyperedge: {edge_id}")
            print(f"  Operation: {edge.operation}")
            print(f"  Sources: {', '.join(edge.sources)}")
            print(f"  Targets: {', '.join(edge.targets)}")
            print(f"  Description: {edge.properties.get('description', 'N/A')}")
            print()

            # Show connected vertices
            print("Connected Vertices:")
            for v_id in edge.sources + edge.targets:
                v = hypergraph.vertices[v_id]
                print(f"  - {v_id} ({v.type}): shape={v.shape}")
            print()
            break

    print("=" * 80)
    print("✓ Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
