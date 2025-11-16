#!/usr/bin/env python3
"""
Complete demonstration of the GGUF Workbench functionality.

This script demonstrates all key features of the GGUF Workbench
including reading, modifying, and writing GGUF files.
"""

import tempfile
import struct
from pathlib import Path

from gguf_workbench import GGUFReader, GGUFWriter


def create_demo_gguf(filepath):
    """Create a minimal GGUF file for demonstration."""
    with open(filepath, "wb") as f:
        # Magic
        f.write(struct.pack("<I", 0x46554747))
        # Version
        f.write(struct.pack("<I", 3))
        # Tensor count
        f.write(struct.pack("<Q", 0))
        # KV count
        f.write(struct.pack("<Q", 5))

        # Key 1: general.name
        key = "general.name"
        f.write(struct.pack("<Q", len(key)))
        f.write(key.encode("utf-8"))
        f.write(struct.pack("<I", 8))  # STRING type
        value = "Demo LLaMA Model"
        f.write(struct.pack("<Q", len(value)))
        f.write(value.encode("utf-8"))

        # Key 2: general.architecture
        key = "general.architecture"
        f.write(struct.pack("<Q", len(key)))
        f.write(key.encode("utf-8"))
        f.write(struct.pack("<I", 8))  # STRING type
        value = "llama"
        f.write(struct.pack("<Q", len(value)))
        f.write(value.encode("utf-8"))

        # Key 3: general.parameter_count
        key = "general.parameter_count"
        f.write(struct.pack("<Q", len(key)))
        f.write(key.encode("utf-8"))
        f.write(struct.pack("<I", 10))  # UINT64 type
        f.write(struct.pack("<Q", 7000000000))

        # Key 4: llama.context_length
        key = "llama.context_length"
        f.write(struct.pack("<Q", len(key)))
        f.write(key.encode("utf-8"))
        f.write(struct.pack("<I", 4))  # UINT32 type
        f.write(struct.pack("<I", 4096))

        # Key 5: llama.rope.freq_base
        key = "llama.rope.freq_base"
        f.write(struct.pack("<Q", len(key)))
        f.write(key.encode("utf-8"))
        f.write(struct.pack("<I", 6))  # FLOAT32 type
        f.write(struct.pack("<f", 10000.0))


def main():
    """Run the complete demonstration."""
    print("=" * 70)
    print("GGUF WORKBENCH DEMONSTRATION")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "demo.gguf"
        modified_file = Path(tmpdir) / "demo_modified.gguf"

        # Create test file
        print("\n1. Creating test GGUF file...")
        create_demo_gguf(test_file)
        print(f"   ✓ Created: {test_file}")

        # Read and inspect
        print("\n2. Reading and inspecting GGUF file...")
        with GGUFReader(test_file) as reader:
            metadata = reader.get_metadata()
            print(f"   ✓ GGUF Version: {metadata.version}")
            print(f"   ✓ Metadata entries: {len(metadata.keys())}")
            print(f"   ✓ Model name: {metadata.get('general.name')}")
            print(f"   ✓ Architecture: {metadata.get('general.architecture')}")
            print(f"   ✓ Parameters: {metadata.get('general.parameter_count'):,}")

        # Modify metadata
        print("\n3. Modifying metadata...")
        with GGUFReader(test_file) as reader:
            metadata = reader.get_metadata()

        metadata.set("general.name", "Custom Demo Model")
        metadata.set("general.version", "1.0")
        metadata.set("general.description", "Modified by GGUF Workbench")
        print("   ✓ Set general.name = 'Custom Demo Model'")
        print("   ✓ Set general.version = '1.0'")
        print("   ✓ Set general.description = 'Modified by GGUF Workbench'")

        # Write modified file
        print("\n4. Writing modified GGUF file...")
        with GGUFWriter(modified_file, metadata) as writer:
            writer.write()
        print(f"   ✓ Wrote: {modified_file}")

        # Verify changes
        print("\n5. Verifying changes...")
        with GGUFReader(modified_file) as reader:
            new_metadata = reader.get_metadata()
            print(f"   ✓ New name: {new_metadata.get('general.name')}")
            print(f"   ✓ New version: {new_metadata.get('general.version')}")
            print(f"   ✓ New description: {new_metadata.get('general.description')}")

        # Delete a key
        print("\n6. Deleting a metadata key...")
        metadata.delete("llama.rope.freq_base")
        print("   ✓ Deleted 'llama.rope.freq_base'")

        final_file = Path(tmpdir) / "demo_final.gguf"
        with GGUFWriter(final_file, metadata) as writer:
            writer.write()

        with GGUFReader(final_file) as reader:
            final_metadata = reader.get_metadata()
            print(f"   ✓ Final metadata entries: {len(final_metadata.keys())}")

        # Export to dict
        print("\n7. Exporting to dictionary...")
        metadata_dict = final_metadata.to_dict()
        print(f"   ✓ Exported {len(metadata_dict['metadata'])} metadata entries")

        print("\n" + "=" * 70)
        print("ALL OPERATIONS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 70)
        print("\nThe GGUF Workbench is fully functional and ready to use!")
        print("\nKey capabilities demonstrated:")
        print("  • Reading GGUF files")
        print("  • Inspecting metadata")
        print("  • Modifying metadata values")
        print("  • Adding new metadata entries")
        print("  • Deleting metadata keys")
        print("  • Writing modified GGUF files")
        print("  • Exporting metadata to dictionaries")


if __name__ == "__main__":
    main()
