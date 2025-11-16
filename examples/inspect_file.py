"""
Example: Inspect a GGUF file and print metadata.

This script demonstrates how to use the GGUF Workbench to read
and inspect a GGUF file.
"""

from gguf_workbench import GGUFReader


def main():
    # Path to your GGUF file
    gguf_file = "model.gguf"
    
    print(f"Inspecting GGUF file: {gguf_file}\n")
    
    try:
        # Open and read the GGUF file
        with GGUFReader(gguf_file) as reader:
            # Get metadata
            metadata = reader.get_metadata()
            
            # Print summary
            print("=" * 60)
            print("GGUF FILE SUMMARY")
            print("=" * 60)
            reader.inspect()
            
            print("\n" + "=" * 60)
            print("SPECIFIC METADATA EXAMPLES")
            print("=" * 60)
            
            # Access specific metadata fields
            model_name = metadata.get("general.name", "Unknown")
            print(f"\nModel Name: {model_name}")
            
            architecture = metadata.get("general.architecture", "Unknown")
            print(f"Architecture: {architecture}")
            
            # List all keys
            print(f"\nTotal metadata keys: {len(metadata.keys())}")
            
    except FileNotFoundError:
        print(f"Error: File '{gguf_file}' not found.")
        print("Please provide a valid GGUF file path.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
