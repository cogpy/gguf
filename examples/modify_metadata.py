"""
Example: Modify GGUF metadata.

This script demonstrates how to modify metadata in a GGUF file.
"""

from gguf_workbench import GGUFReader, GGUFWriter


def main():
    input_file = "model.gguf"
    output_file = "model_modified.gguf"
    
    print(f"Modifying GGUF file: {input_file}\n")
    
    try:
        # Read the existing GGUF file
        print("Reading original file...")
        with GGUFReader(input_file) as reader:
            metadata = reader.get_metadata()
            
            # Show original values
            print(f"Original name: {metadata.get('general.name', 'N/A')}")
            
        # Modify metadata
        print("\nModifying metadata...")
        metadata.set("general.name", "My Custom Model")
        metadata.set("general.description", "Modified with GGUF Workbench")
        metadata.set("general.custom_field", "This is a custom value")
        
        # Show new values
        print(f"New name: {metadata.get('general.name')}")
        print(f"Description: {metadata.get('general.description')}")
        
        # Write to new file
        print(f"\nWriting to {output_file}...")
        with GGUFWriter(output_file, metadata) as writer:
            writer.write()
        
        print("✓ File modified successfully!")
        
        # Verify the changes
        print("\nVerifying changes...")
        with GGUFReader(output_file) as reader:
            new_metadata = reader.get_metadata()
            print(f"Verified name: {new_metadata.get('general.name')}")
            print(f"Verified description: {new_metadata.get('general.description')}")
            
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        print("Please provide a valid GGUF file path.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
