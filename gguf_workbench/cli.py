"""
Command-line interface for GGUF Workbench.
"""

import argparse
import json
import sys

from . import (
    GGUFReader,
    GGUFWriter,
    GGUFConverter,
    ConversationDataset,
    ConversationAnalyzer,
    __version__,
)


def cmd_inspect(args) -> int:
    """Inspect a GGUF file and display its metadata."""
    try:
        with GGUFReader(args.file) as reader:
            reader.inspect()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_get(args) -> int:
    """Get a specific metadata value from a GGUF file."""
    try:
        with GGUFReader(args.file) as reader:
            metadata = reader.get_metadata()
            value = metadata.get(args.key)
            if value is None:
                print(f"Key '{args.key}' not found", file=sys.stderr)
                return 1

            if args.json:
                print(json.dumps(value, indent=2))
            else:
                print(value)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_set(args) -> int:
    """Set a metadata value in a GGUF file."""
    try:
        # Read existing file
        with GGUFReader(args.file) as reader:
            metadata = reader.get_metadata()

        # Parse the value
        value = args.value
        if args.type == "int":
            value = int(value)
        elif args.type == "float":
            value = float(value)
        elif args.type == "bool":
            value = value.lower() in ("true", "1", "yes")
        elif args.type == "json":
            value = json.loads(value)
        # Otherwise keep as string

        # Set the value
        metadata.set(args.key, value)

        # Determine output file
        output_file = args.output or args.file

        # Write the modified file
        with GGUFWriter(output_file, metadata) as writer:
            writer.write()

        print(f"Updated '{args.key}' in {output_file}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_delete(args) -> int:
    """Delete a metadata key from a GGUF file."""
    try:
        # Read existing file
        with GGUFReader(args.file) as reader:
            metadata = reader.get_metadata()

        # Delete the key
        if not metadata.delete(args.key):
            print(f"Key '{args.key}' not found", file=sys.stderr)
            return 1

        # Determine output file
        output_file = args.output or args.file

        # Write the modified file
        with GGUFWriter(output_file, metadata) as writer:
            writer.write()

        print(f"Deleted '{args.key}' from {output_file}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_export(args) -> int:
    """Export metadata to JSON."""
    try:
        with GGUFReader(args.file) as reader:
            metadata = reader.get_metadata()
            metadata_dict = metadata.to_dict()

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(metadata_dict, f, indent=2)
                print(f"Exported metadata to {args.output}")
            else:
                print(json.dumps(metadata_dict, indent=2))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list(args) -> int:
    """List all metadata keys."""
    try:
        with GGUFReader(args.file) as reader:
            metadata = reader.get_metadata()
            keys = sorted(metadata.keys())

            if args.verbose:
                print(f"Total keys: {len(keys)}\n")

            for key in keys:
                if args.verbose:
                    value = metadata.get(key)
                    value_str = str(value)
                    if len(value_str) > 60:
                        value_str = value_str[:57] + "..."
                    print(f"{key}: {value_str}")
                else:
                    print(key)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_convert(args) -> int:
    """Convert GGUF file to various representation formats."""
    try:
        converter = GGUFConverter(args.file)
        
        # Determine which formats to export
        formats = None
        if args.format:
            formats = [args.format]
        
        # Export all or specified formats
        results = converter.export_all(
            output_dir=args.output,
            include_weights=args.weights,
            formats=formats
        )
        
        print(f"Converted {args.file} to {args.output}/")
        print(f"\nGenerated formats:")
        for fmt, path in results.items():
            if isinstance(path, dict):
                for subfmt, subpath in path.items():
                    print(f"  {fmt} ({subfmt}): {subpath}")
            else:
                print(f"  {fmt}: {path}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_analyze_conversations(args) -> int:
    """Analyze conversation datasets to understand what can be learned about model internals."""
    try:
        # Load conversation dataset
        dataset = ConversationDataset.from_json_file(args.file)
        
        # Create analyzer with optional model info
        analyzer = ConversationAnalyzer(
            model_vocab_size=args.vocab_size,
            embedding_dim=args.embedding_dim,
        )
        
        # Perform analysis
        print("Analyzing conversation dataset...")
        result = analyzer.analyze(dataset)
        
        # Display summary
        if not args.quiet:
            print(result.summary())
        
        # Save detailed results if output specified
        if args.output:
            result.to_json_file(args.output)
            print(f"\n✓ Saved detailed analysis to {args.output}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="GGUF Workbench - A tool for inspecting and modifying GGUF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a GGUF file")
    inspect_parser.add_argument("file", help="Path to the GGUF file")
    inspect_parser.set_defaults(func=cmd_inspect)

    # Get command
    get_parser = subparsers.add_parser("get", help="Get a metadata value")
    get_parser.add_argument("file", help="Path to the GGUF file")
    get_parser.add_argument("key", help="Metadata key to retrieve")
    get_parser.add_argument("--json", action="store_true", help="Output as JSON")
    get_parser.set_defaults(func=cmd_get)

    # Set command
    set_parser = subparsers.add_parser("set", help="Set a metadata value")
    set_parser.add_argument("file", help="Path to the GGUF file")
    set_parser.add_argument("key", help="Metadata key to set")
    set_parser.add_argument("value", help="Value to set")
    set_parser.add_argument(
        "--type",
        choices=["string", "int", "float", "bool", "json"],
        default="string",
        help="Value type (default: string)",
    )
    set_parser.add_argument("-o", "--output", help="Output file (default: modify in-place)")
    set_parser.set_defaults(func=cmd_set)

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a metadata key")
    delete_parser.add_argument("file", help="Path to the GGUF file")
    delete_parser.add_argument("key", help="Metadata key to delete")
    delete_parser.add_argument("-o", "--output", help="Output file (default: modify in-place)")
    delete_parser.set_defaults(func=cmd_delete)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export metadata to JSON")
    export_parser.add_argument("file", help="Path to the GGUF file")
    export_parser.add_argument("-o", "--output", help="Output JSON file (default: stdout)")
    export_parser.set_defaults(func=cmd_export)

    # List command
    list_parser = subparsers.add_parser("list", help="List all metadata keys")
    list_parser.add_argument("file", help="Path to the GGUF file")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show key values")
    list_parser.set_defaults(func=cmd_list)
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert GGUF to various representations")
    convert_parser.add_argument("file", help="Path to the GGUF file")
    convert_parser.add_argument("output", help="Output directory for generated files")
    convert_parser.add_argument(
        "-f", "--format",
        choices=["hypergraph", "dag", "symbolic", "aiml", "atomspace", "toml"],
        help="Specific format to generate (default: all formats)"
    )
    convert_parser.add_argument(
        "-w", "--weights",
        action="store_true",
        help="Include weight values (creates large files for big models)"
    )
    convert_parser.set_defaults(func=cmd_convert)
    
    # Analyze conversations command
    analyze_parser = subparsers.add_parser(
        "analyze-conversations",
        help="Analyze conversation datasets to understand learning limits"
    )
    analyze_parser.add_argument("file", help="Path to conversation dataset JSON file")
    analyze_parser.add_argument(
        "-o", "--output",
        help="Output JSON file for detailed analysis (default: display summary only)"
    )
    analyze_parser.add_argument(
        "--vocab-size",
        type=int,
        help="Model vocabulary size (if known)"
    )
    analyze_parser.add_argument(
        "--embedding-dim",
        type=int,
        help="Model embedding dimension (if known)"
    )
    analyze_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress summary output (only save to file)"
    )
    analyze_parser.set_defaults(func=cmd_analyze_conversations)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
