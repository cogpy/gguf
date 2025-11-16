"""
Example: Export GGUF metadata to JSON.

This script demonstrates how to export all metadata from a GGUF file
to a JSON file for analysis or documentation.
"""

import json
from gguf_workbench import GGUFReader


def main():
    input_file = "model.gguf"
    output_file = "metadata.json"
    
    print(f"Exporting metadata from: {input_file}\n")
    
    try:
        # Read the GGUF file
        with GGUFReader(input_file) as reader:
            metadata = reader.get_metadata()
            
            # Convert to dictionary
            metadata_dict = metadata.to_dict()
            
            # Write to JSON file
            with open(output_file, "w") as f:
                json.dump(metadata_dict, f, indent=2)
            
            print(f"✓ Metadata exported to {output_file}")
            print(f"\nExported {len(metadata.keys())} metadata keys")
            print(f"Tensor count: {metadata.tensor_count}")
            
            # Show a preview
            print("\nPreview of exported data:")
            print(json.dumps(metadata_dict, indent=2)[:500] + "...")
            
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        print("Please provide a valid GGUF file path.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
