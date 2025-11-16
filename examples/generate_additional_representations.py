#!/usr/bin/env python
"""
Generate additional representation formats for the TinyTransformer model.

This script generates:
- AIML (Pandorabot) representation
- OpenCog AtomSpace representation (URE, PLN, ECAN, MOSES)
- TOML Hypergraph representation with explicit tuples
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf_workbench.representations import (  # noqa: E402
    AIMLRepresentation,
    OpenCogAtomSpaceRepresentation,
    TOMLHypergraphRepresentation,
)


def main():
    """Generate new representation formats."""
    print("=" * 80)
    print("TinyTransformer Additional Representations")
    print("=" * 80)
    print()

    output_dir = Path(__file__).parent.parent / "tinytf" / "representations"
    output_dir.mkdir(exist_ok=True)

    # 1. Generate AIML Representation
    print("1. Generating AIML (Pandorabot) Representation...")
    aiml = AIMLRepresentation.from_tiny_transformer()

    # Save AIML outputs
    aiml_xml_path = output_dir / "tiny_transformer.aiml"
    aiml.save_xml(aiml_xml_path)
    print(f"   ✓ Saved AIML XML to {aiml_xml_path}")

    aiml_json_path = output_dir / "tiny_transformer_aiml.json"
    aiml.save_json(aiml_json_path)
    print(f"   ✓ Saved AIML JSON to {aiml_json_path}")

    print(f"   ✓ Categories: {len(aiml.categories)}")
    print(f"   ✓ Topic: {aiml.topic}")
    print()

    # 2. Generate OpenCog AtomSpace Representation
    print("2. Generating OpenCog AtomSpace Representation...")
    atomspace = OpenCogAtomSpaceRepresentation.from_tiny_transformer()

    # Save OpenCog outputs
    scheme_path = output_dir / "tiny_transformer_atomspace.scm"
    atomspace.save_scheme(scheme_path)
    print(f"   ✓ Saved Scheme to {scheme_path}")

    opencog_json_path = output_dir / "tiny_transformer_opencog.json"
    atomspace.save_json(opencog_json_path)
    print(f"   ✓ Saved JSON to {opencog_json_path}")

    print(f"   ✓ Atoms: {len(atomspace.atoms)}")
    print(f"   ✓ KB Rules: {len(atomspace.knowledge_base)}")
    print(f"   ✓ Compatible with: URE, PLN, ECAN, MOSES")
    print()

    # 3. Generate TOML Hypergraph Representation
    print("3. Generating TOML Hypergraph Representation...")
    toml_hg = TOMLHypergraphRepresentation.from_tiny_transformer(include_weights=True)

    # Save TOML outputs
    toml_path = output_dir / "tiny_transformer_hypergraph.toml"
    toml_hg.save_toml(toml_path)
    print(f"   ✓ Saved TOML to {toml_path}")

    toml_json_path = output_dir / "tiny_transformer_toml_hypergraph.json"
    toml_hg.save_json(toml_json_path)
    print(f"   ✓ Saved JSON to {toml_json_path}")

    stats = toml_hg.to_json()["statistics"]
    print(f"   ✓ Vertices: {stats['vertex_count']}")
    print(f"   ✓ Hyperedges: {stats['hyperedge_count']}")
    print(f"   ✓ Parameters: {stats['parameter_count']}")
    print()

    # Summary
    print("=" * 80)
    print("Generation Complete!")
    print("=" * 80)
    print()
    print("Generated files:")
    print(f"  - {aiml_xml_path.name} (AIML for Pandorabots)")
    print(f"  - {scheme_path.name} (OpenCog Scheme)")
    print(f"  - {toml_path.name} (TOML Hypergraph)")
    print()
    print("All representations are consistent with the TinyTransformer model.")


if __name__ == "__main__":
    main()
