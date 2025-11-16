#!/usr/bin/env python3
"""
Example demonstrating the generalized GGUF converter.

This script shows how to convert any GGUF file to various representation formats.
"""

from pathlib import Path
from gguf_workbench import GGUFConverter


def main():
    # Example: Convert the tiny transformer model
    gguf_file = "tinytf/tiny_model.gguf"
    output_dir = "/tmp/gguf_representations"
    
    print(f"Converting {gguf_file}...")
    print()
    
    # Create converter
    converter = GGUFConverter(gguf_file)
    
    # Display model info
    print("Model Information:")
    print(f"  Architecture: {converter.architecture}")
    for key, value in converter.model_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Convert to different formats (structure only, no weights)
    print("Converting to representation formats...")
    results = converter.export_all(
        output_dir=output_dir,
        include_weights=False
    )
    
    print()
    print("Generated files:")
    for fmt, path in results.items():
        if isinstance(path, dict):
            for subfmt, subpath in path.items():
                file_size = Path(subpath).stat().st_size
                print(f"  {fmt} ({subfmt}): {subpath} ({file_size:,} bytes)")
        else:
            file_size = Path(path).stat().st_size
            print(f"  {fmt}: {path} ({file_size:,} bytes)")
    
    print()
    print("Conversion complete!")
    print()
    
    # Demonstrate individual conversions
    print("Individual format examples:")
    print()
    
    # Hypergraph
    print("1. Hypergraph representation:")
    hg = converter.to_hypergraph(include_weights=False)
    stats = hg.get_statistics()
    print(f"   Vertices: {stats['vertex_count']}")
    print(f"   Hyperedges: {stats['hyperedge_count']}")
    print(f"   Operations: {stats['operation_types']}")
    print()
    
    # DAG
    print("2. DAG representation:")
    dag = converter.to_dag()
    print(f"   Nodes: {len(dag.nodes)}")
    print(f"   Edges: {len(dag.edges)}")
    print()
    
    # Symbolic
    print("3. Symbolic representation:")
    symbolic = converter.to_symbolic()
    print(f"   Parameters: {len(symbolic.parameters)}")
    print(f"   Expressions: {len(symbolic.expressions)}")
    print()
    
    # AIML
    print("4. AIML (chatbot) representation:")
    aiml = converter.to_aiml()
    print(f"   Categories: {len(aiml.categories)}")
    print()
    
    # OpenCog
    print("5. OpenCog AtomSpace representation:")
    atomspace = converter.to_atomspace()
    json_data = atomspace.to_json()
    print(f"   Atoms: {json_data['atom_count']}")
    print(f"   KB Rules: {json_data['knowledge_base_rules']}")
    print()
    
    # TOML Hypergraph
    print("6. TOML Hypergraph representation:")
    toml_hg = converter.to_toml_hypergraph(include_weights=False)
    toml_stats = toml_hg.to_json()['statistics']
    print(f"   Vertices: {toml_stats['vertex_count']}")
    print(f"   Hyperedges: {toml_stats['hyperedge_count']}")
    print()


if __name__ == "__main__":
    main()
