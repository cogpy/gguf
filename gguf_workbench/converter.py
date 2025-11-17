"""
Generalized converter framework for GGUF files to various representations.

This module provides a unified interface for converting any GGUF file to and from
different representation formats. It automatically extracts model architecture
information and builds appropriate representations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from .reader import GGUFReader
from .representations import (
    HypergraphRepresentation,
    GraphRepresentation,
    SymbolicRepresentation,
    AIMLRepresentation,
    OpenCogAtomSpaceRepresentation,
    TOMLHypergraphRepresentation,
)


class GGUFConverter:
    """
    Generalized converter for GGUF files to various representations.
    
    This converter works with ANY GGUF file by:
    1. Extracting architecture metadata
    2. Parsing tensor information
    3. Building appropriate graph/symbolic representations
    4. Optionally including weights or using external references
    
    Example:
        converter = GGUFConverter("model.gguf")
        
        # Convert to different formats
        hypergraph = converter.to_hypergraph(include_weights=False)
        dag = converter.to_dag()
        symbolic = converter.to_symbolic()
        atomspace = converter.to_atomspace()
        
        # Save all representations
        converter.export_all("output_dir/")
    """
    
    def __init__(self, gguf_path: str):
        """
        Initialize converter with a GGUF file.
        
        Args:
            gguf_path: Path to GGUF file
        """
        self.gguf_path = Path(gguf_path)
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[Dict[str, Any]] = []
        self.architecture: str = "unknown"
        self.model_info: Dict[str, Any] = {}
        
        # Load and parse GGUF file
        self._load_gguf()
    
    def _load_gguf(self):
        """Load GGUF file and extract metadata and tensor information."""
        with GGUFReader(str(self.gguf_path)) as reader:
            metadata_obj = reader.get_metadata()
            self.metadata = metadata_obj.to_dict()
            
            # Extract model architecture information
            self.architecture = self.metadata.get(
                "general.architecture",
                self.metadata.get("Model_Architecture", "unknown")
            )
            
            # Extract key model parameters
            self._extract_model_info()
            
            # Get tensor information
            self.tensors = self._extract_tensor_info(reader)
    
    def _extract_model_info(self):
        """Extract key model information from metadata."""
        # Common keys across different architectures
        info = {
            "architecture": self.architecture,
            "name": self.metadata.get("general.name", "Unknown"),
        }
        
        # Try to get architecture-specific parameters
        arch = self.architecture
        
        # Common parameter patterns
        param_patterns = {
            "vocab_size": [
                f"{arch}.vocab_size",
                "tokenizer.ggml.vocab_size",
                "general.vocab_size"
            ],
            "embedding_dim": [
                f"{arch}.embedding_length",
                f"{arch}.embed_dim",
                "general.embedding_length"
            ],
            "context_length": [
                f"{arch}.context_length",
                "general.context_length"
            ],
            "num_layers": [
                f"{arch}.block_count",
                f"{arch}.layer_count",
                "general.block_count"
            ],
            "num_heads": [
                f"{arch}.attention.head_count",
                "general.attention_head_count"
            ],
            "ffn_dim": [
                f"{arch}.feed_forward_length",
                f"{arch}.ffn_dim"
            ],
        }
        
        # Try each pattern to find values
        for param_name, keys in param_patterns.items():
            for key in keys:
                if key in self.metadata:
                    info[param_name] = self.metadata[key]
                    break
        
        self.model_info = info
    
    def _extract_tensor_info(self, reader: GGUFReader) -> List[Dict[str, Any]]:
        """
        Extract tensor information from GGUF file.
        
        Args:
            reader: GGUFReader instance
            
        Returns:
            List of tensor information dictionaries
        """
        tensors = []
        
        # Access internal tensor information
        # Note: This is implementation-specific to GGUFReader
        # In a real implementation, we'd need to add methods to GGUFReader
        # to expose tensor metadata
        
        # For now, return empty list - will be populated when we enhance GGUFReader
        return tensors
    
    def to_hypergraph(
        self,
        include_weights: bool = False,
        weight_reference_path: Optional[str] = None
    ) -> HypergraphRepresentation:
        """
        Convert GGUF model to hypergraph representation.
        
        Args:
            include_weights: Whether to include actual weight values
            weight_reference_path: Path to external weight file if not including inline
            
        Returns:
            HypergraphRepresentation instance
        """
        hg = HypergraphRepresentation()
        
        # Set metadata
        hg.metadata = self.model_info.copy()
        
        # Build hypergraph structure based on architecture
        self._build_hypergraph_structure(hg, include_weights, weight_reference_path)
        
        return hg
    
    def _build_hypergraph_structure(
        self,
        hg: HypergraphRepresentation,
        include_weights: bool,
        weight_reference_path: Optional[str]
    ):
        """
        Build hypergraph structure from GGUF metadata.
        
        This method constructs a hypergraph representation by:
        1. Creating vertices for input, parameters, and intermediate tensors
        2. Creating hyperedges for operations (embedding, attention, FFN, etc.)
        3. Connecting them based on the architecture
        """
        from .representations.hypergraph import Vertex, Hyperedge
        
        arch = self.architecture
        info = self.model_info
        
        # Get dimensions
        vocab_size = info.get("vocab_size", 10)
        embed_dim = info.get("embedding_dim", 512)
        context_len = info.get("context_length", 512)
        num_layers = info.get("num_layers", 12)
        num_heads = info.get("num_heads", 8)
        ffn_dim = info.get("ffn_dim", embed_dim * 4)
        
        # Input tokens
        hg.add_vertex(Vertex(
            id="input_tokens",
            type="tensor",
            shape=(None, context_len),
            dtype="int64",
            properties={"description": "Input token IDs"}
        ))
        
        # Token embedding
        hg.add_vertex(Vertex(
            id="embedding_weights",
            type="parameter",
            shape=(vocab_size, embed_dim),
            dtype="float32",
            properties={
                "description": "Token embedding matrix",
                "parameter_count": vocab_size * embed_dim,
                "weight_reference": f"{weight_reference_path}#embedding" if weight_reference_path else None
            }
        ))
        
        hg.add_vertex(Vertex(
            id="embeddings",
            type="tensor",
            shape=(None, context_len, embed_dim),
            dtype="float32",
            properties={"description": "Token embeddings"}
        ))
        
        hg.add_hyperedge(Hyperedge(
            id="embed_tokens",
            sources=["input_tokens", "embedding_weights"],
            targets=["embeddings"],
            operation="embedding_lookup",
            properties={"description": "Convert token IDs to embeddings"}
        ))
        
        # Current tensor (flows through layers)
        current_tensor = "embeddings"
        
        # Transformer layers
        for layer_idx in range(num_layers):
            layer_prefix = f"layer_{layer_idx}"
            
            # Layer input
            layer_input = current_tensor
            
            # Self-attention block
            attn_output = self._add_attention_block(
                hg, layer_prefix, layer_input, embed_dim, num_heads,
                include_weights, weight_reference_path
            )
            
            # Add & Norm 1
            add_norm_1 = f"{layer_prefix}_add_norm_1"
            self._add_add_norm(
                hg, f"{layer_prefix}_an1", layer_input, attn_output, add_norm_1,
                embed_dim, include_weights, weight_reference_path
            )
            
            # Feed-forward block
            ffn_output = self._add_ffn_block(
                hg, layer_prefix, add_norm_1, embed_dim, ffn_dim,
                include_weights, weight_reference_path
            )
            
            # Add & Norm 2
            add_norm_2 = f"{layer_prefix}_add_norm_2"
            self._add_add_norm(
                hg, f"{layer_prefix}_an2", add_norm_1, ffn_output, add_norm_2,
                embed_dim, include_weights, weight_reference_path
            )
            
            current_tensor = add_norm_2
        
        # Final layer norm
        hg.add_vertex(Vertex(
            id="final_norm_output",
            type="tensor",
            shape=(None, context_len, embed_dim),
            dtype="float32"
        ))
        
        self._add_layer_norm(
            hg, "final", current_tensor, "final_norm_output",
            embed_dim, include_weights, weight_reference_path
        )
        
        # Output projection
        hg.add_vertex(Vertex(
            id="lm_head_weights",
            type="parameter",
            shape=(embed_dim, vocab_size),
            dtype="float32",
            properties={
                "description": "Language model head weights",
                "parameter_count": embed_dim * vocab_size,
                "weight_reference": f"{weight_reference_path}#lm_head" if weight_reference_path else None
            }
        ))
        
        hg.add_vertex(Vertex(
            id="logits",
            type="tensor",
            shape=(None, context_len, vocab_size),
            dtype="float32",
            properties={"description": "Output logits"}
        ))
        
        hg.add_hyperedge(Hyperedge(
            id="compute_logits",
            sources=["final_norm_output", "lm_head_weights"],
            targets=["logits"],
            operation="linear",
            properties={"description": "Compute output logits"}
        ))
    
    def _add_attention_block(
        self, hg, prefix, input_tensor, dim, num_heads,
        include_weights, weight_ref_path
    ):
        """Add self-attention block to hypergraph."""
        from .representations.hypergraph import Vertex, Hyperedge
        
        head_dim = dim // num_heads
        
        # Q, K, V projections
        for proj in ["query", "key", "value"]:
            hg.add_vertex(Vertex(
                id=f"{prefix}_{proj}_weights",
                type="parameter",
                shape=(dim, dim),
                dtype="float32",
                properties={
                    "description": f"Attention {proj} projection",
                    "weight_reference": f"{weight_ref_path}#{prefix}_{proj}" if weight_ref_path else None
                }
            ))
            
            hg.add_vertex(Vertex(
                id=f"{prefix}_{proj}",
                type="tensor",
                shape=(None, None, num_heads, head_dim),
                dtype="float32"
            ))
            
            hg.add_hyperedge(Hyperedge(
                id=f"{prefix}_project_{proj}",
                sources=[input_tensor, f"{prefix}_{proj}_weights"],
                targets=[f"{prefix}_{proj}"],
                operation="linear",
                properties={"projection_type": proj}
            ))
        
        # Attention scores
        hg.add_vertex(Vertex(
            id=f"{prefix}_attn_scores",
            type="tensor",
            shape=(None, num_heads, None, None),
            dtype="float32"
        ))
        
        hg.add_hyperedge(Hyperedge(
            id=f"{prefix}_compute_scores",
            sources=[f"{prefix}_query", f"{prefix}_key"],
            targets=[f"{prefix}_attn_scores"],
            operation="scaled_dot_product",
            properties={"scale_factor": 1.0 / (head_dim ** 0.5)}
        ))
        
        # Attention weights (softmax)
        hg.add_vertex(Vertex(
            id=f"{prefix}_attn_weights",
            type="tensor",
            shape=(None, num_heads, None, None),
            dtype="float32"
        ))
        
        hg.add_hyperedge(Hyperedge(
            id=f"{prefix}_softmax",
            sources=[f"{prefix}_attn_scores"],
            targets=[f"{prefix}_attn_weights"],
            operation="softmax",
            properties={"dim": -1}
        ))
        
        # Attention output
        hg.add_vertex(Vertex(
            id=f"{prefix}_attn_output_pre",
            type="tensor",
            shape=(None, None, num_heads, head_dim),
            dtype="float32"
        ))
        
        hg.add_hyperedge(Hyperedge(
            id=f"{prefix}_apply_attention",
            sources=[f"{prefix}_attn_weights", f"{prefix}_value"],
            targets=[f"{prefix}_attn_output_pre"],
            operation="attention_combine"
        ))
        
        # Output projection
        hg.add_vertex(Vertex(
            id=f"{prefix}_out_proj_weights",
            type="parameter",
            shape=(dim, dim),
            dtype="float32",
            properties={
                "weight_reference": f"{weight_ref_path}#{prefix}_out" if weight_ref_path else None
            }
        ))
        
        output_id = f"{prefix}_attn_output"
        hg.add_vertex(Vertex(
            id=output_id,
            type="tensor",
            shape=(None, None, dim),
            dtype="float32"
        ))
        
        hg.add_hyperedge(Hyperedge(
            id=f"{prefix}_out_projection",
            sources=[f"{prefix}_attn_output_pre", f"{prefix}_out_proj_weights"],
            targets=[output_id],
            operation="linear"
        ))
        
        return output_id
    
    def _add_ffn_block(self, hg, prefix, input_tensor, dim, ffn_dim, include_weights, weight_ref_path):
        """Add feed-forward block to hypergraph."""
        from .representations.hypergraph import Vertex, Hyperedge
        
        # First linear layer
        hg.add_vertex(Vertex(
            id=f"{prefix}_ffn_w1",
            type="parameter",
            shape=(dim, ffn_dim),
            dtype="float32",
            properties={
                "weight_reference": f"{weight_ref_path}#{prefix}_ffn_w1" if weight_ref_path else None
            }
        ))
        
        hg.add_vertex(Vertex(
            id=f"{prefix}_ffn_hidden",
            type="tensor",
            shape=(None, None, ffn_dim),
            dtype="float32"
        ))
        
        hg.add_hyperedge(Hyperedge(
            id=f"{prefix}_ffn_up",
            sources=[input_tensor, f"{prefix}_ffn_w1"],
            targets=[f"{prefix}_ffn_hidden"],
            operation="linear"
        ))
        
        # Activation (GELU/ReLU)
        hg.add_vertex(Vertex(
            id=f"{prefix}_ffn_activated",
            type="tensor",
            shape=(None, None, ffn_dim),
            dtype="float32"
        ))
        
        hg.add_hyperedge(Hyperedge(
            id=f"{prefix}_ffn_activation",
            sources=[f"{prefix}_ffn_hidden"],
            targets=[f"{prefix}_ffn_activated"],
            operation="gelu"
        ))
        
        # Second linear layer
        hg.add_vertex(Vertex(
            id=f"{prefix}_ffn_w2",
            type="parameter",
            shape=(ffn_dim, dim),
            dtype="float32",
            properties={
                "weight_reference": f"{weight_ref_path}#{prefix}_ffn_w2" if weight_ref_path else None
            }
        ))
        
        output_id = f"{prefix}_ffn_output"
        hg.add_vertex(Vertex(
            id=output_id,
            type="tensor",
            shape=(None, None, dim),
            dtype="float32"
        ))
        
        hg.add_hyperedge(Hyperedge(
            id=f"{prefix}_ffn_down",
            sources=[f"{prefix}_ffn_activated", f"{prefix}_ffn_w2"],
            targets=[output_id],
            operation="linear"
        ))
        
        return output_id
    
    def _add_layer_norm(self, hg, prefix, input_tensor, output_tensor, dim, include_weights, weight_ref_path):
        """Add layer normalization to hypergraph."""
        from .representations.hypergraph import Vertex, Hyperedge
        
        hg.add_vertex(Vertex(
            id=f"{prefix}_ln_weight",
            type="parameter",
            shape=(dim,),
            dtype="float32",
            properties={
                "weight_reference": f"{weight_ref_path}#{prefix}_ln_w" if weight_ref_path else None
            }
        ))
        
        hg.add_vertex(Vertex(
            id=f"{prefix}_ln_bias",
            type="parameter",
            shape=(dim,),
            dtype="float32",
            properties={
                "weight_reference": f"{weight_ref_path}#{prefix}_ln_b" if weight_ref_path else None
            }
        ))
        
        hg.add_hyperedge(Hyperedge(
            id=f"{prefix}_layer_norm",
            sources=[input_tensor, f"{prefix}_ln_weight", f"{prefix}_ln_bias"],
            targets=[output_tensor],
            operation="layer_norm"
        ))
    
    def _add_add_norm(self, hg, prefix, input1, input2, output, dim, include_weights, weight_ref_path):
        """Add residual connection + layer norm to hypergraph."""
        from .representations.hypergraph import Vertex, Hyperedge
        
        # Residual add
        residual_id = f"{prefix}_residual"
        hg.add_vertex(Vertex(
            id=residual_id,
            type="tensor",
            shape=(None, None, dim),
            dtype="float32"
        ))
        
        hg.add_hyperedge(Hyperedge(
            id=f"{prefix}_add",
            sources=[input1, input2],
            targets=[residual_id],
            operation="add"
        ))
        
        # Layer norm
        self._add_layer_norm(hg, prefix, residual_id, output, dim, include_weights, weight_ref_path)
    
    def to_dag(self) -> GraphRepresentation:
        """
        Convert GGUF model to DAG representation.
        
        Returns:
            GraphRepresentation instance
        """
        # First create hypergraph, then convert to DAG
        hg = self.to_hypergraph(include_weights=False)
        return hg.to_dag()
    
    def to_symbolic(self) -> SymbolicRepresentation:
        """
        Convert GGUF model to symbolic/mathematical representation.
        
        Returns:
            SymbolicRepresentation instance
        """
        from .representations.symbolic import SymbolicRepresentation
        
        sr = SymbolicRepresentation()
        
        # Set metadata
        sr.metadata = self.model_info.copy()
        
        # Build symbolic representation
        self._build_symbolic_representation(sr)
        
        return sr
    
    def _build_symbolic_representation(self, sr: SymbolicRepresentation):
        """Build symbolic representation from model info."""
        info = self.model_info
        
        # Define parameters
        sr.add_parameter("E", "Token embedding matrix", (info.get("vocab_size", "V"), info.get("embedding_dim", "d")))
        sr.add_parameter("W^Q", "Query projection", (info.get("embedding_dim", "d"), info.get("embedding_dim", "d")))
        sr.add_parameter("W^K", "Key projection", (info.get("embedding_dim", "d"), info.get("embedding_dim", "d")))
        sr.add_parameter("W^V", "Value projection", (info.get("embedding_dim", "d"), info.get("embedding_dim", "d")))
        sr.add_parameter("W^O", "Output projection", (info.get("embedding_dim", "d"), info.get("embedding_dim", "d")))
        sr.add_parameter("W_1", "FFN first layer", (info.get("embedding_dim", "d"), info.get("ffn_dim", "4d")))
        sr.add_parameter("W_2", "FFN second layer", (info.get("ffn_dim", "4d"), info.get("embedding_dim", "d")))
        
        # Define expressions
        sr.add_expression("h_0 = E[x]", "Embedding lookup")
        sr.add_expression("Q = h * W^Q", "Query projection")
        sr.add_expression("K = h * W^K", "Key projection")
        sr.add_expression("V = h * W^V", "Value projection")
        sr.add_expression("A = softmax(QK^T / sqrt(d_k))", "Attention weights")
        sr.add_expression("h_attn = (A * V) * W^O", "Attention output")
        sr.add_expression("h_ffn = GELU(h * W_1) * W_2", "Feed-forward")
        sr.add_expression("y = LayerNorm(h_ffn)", "Final output")
    
    def to_atomspace(self) -> OpenCogAtomSpaceRepresentation:
        """
        Convert GGUF model to OpenCog AtomSpace representation.
        
        Returns:
            OpenCogAtomSpaceRepresentation instance
        """
        # Create from hypergraph
        hg = self.to_hypergraph(include_weights=False)
        return OpenCogAtomSpaceRepresentation.from_hypergraph(hg, self.model_info)
    
    def to_toml_hypergraph(self, include_weights: bool = False) -> TOMLHypergraphRepresentation:
        """
        Convert GGUF model to TOML hypergraph representation.
        
        Args:
            include_weights: Whether to include weight values
            
        Returns:
            TOMLHypergraphRepresentation instance
        """
        hg = self.to_hypergraph(include_weights=include_weights)
        return TOMLHypergraphRepresentation.from_hypergraph(hg, include_weights=include_weights)
    
    def to_aiml(self) -> AIMLRepresentation:
        """
        Convert GGUF model to AIML representation.
        
        Returns:
            AIMLRepresentation instance
        """
        aiml = AIMLRepresentation()
        
        info = self.model_info
        
        # Add categories about the model
        aiml.add_category("WHAT IS THE MODEL NAME", f"The model name is {info.get('name', 'Unknown')}.")
        aiml.add_category("WHAT IS THE ARCHITECTURE", f"This is a {info.get('architecture', 'transformer')} architecture model.")
        
        if "vocab_size" in info:
            aiml.add_category("WHAT IS THE VOCABULARY SIZE", f"The vocabulary size is {info['vocab_size']}.")
        
        if "embedding_dim" in info:
            aiml.add_category("WHAT IS THE EMBEDDING DIMENSION", f"The embedding dimension is {info['embedding_dim']}.")
        
        if "num_layers" in info:
            aiml.add_category("HOW MANY LAYERS", f"The model has {info['num_layers']} transformer layers.")
        
        if "num_heads" in info:
            aiml.add_category("HOW MANY ATTENTION HEADS", f"Each layer has {info['num_heads']} attention heads.")
        
        return aiml
    
    def export_all(
        self,
        output_dir: str,
        include_weights: bool = False,
        formats: Optional[List[str]] = None
    ):
        """
        Export model to all supported representation formats.
        
        Args:
            output_dir: Directory to save output files
            include_weights: Whether to include weight values
            formats: List of formats to export (default: all)
                     Options: 'hypergraph', 'dag', 'symbolic', 'aiml', 'atomspace', 'toml'
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if formats is None:
            formats = ['hypergraph', 'dag', 'symbolic', 'aiml', 'atomspace', 'toml']
        
        base_name = self.gguf_path.stem
        
        results = {}
        
        if 'hypergraph' in formats:
            hg = self.to_hypergraph(include_weights=include_weights)
            path = output_path / f"{base_name}_hypergraph.json"
            hg.to_json(path)
            results['hypergraph'] = str(path)
        
        if 'dag' in formats:
            dag = self.to_dag()
            path = output_path / f"{base_name}_dag.json"
            dag.to_json(path)
            results['dag'] = str(path)
        
        if 'symbolic' in formats:
            symbolic = self.to_symbolic()
            path_json = output_path / f"{base_name}_symbolic.json"
            path_md = output_path / f"{base_name}_symbolic.md"
            symbolic.to_json(path_json)
            symbolic.export_markdown(path_md)
            results['symbolic'] = {'json': str(path_json), 'markdown': str(path_md)}
        
        if 'aiml' in formats:
            aiml = self.to_aiml()
            path = output_path / f"{base_name}.aiml"
            aiml.save_xml(path)
            results['aiml'] = str(path)
        
        if 'atomspace' in formats:
            atomspace = self.to_atomspace()
            path = output_path / f"{base_name}_atomspace.scm"
            atomspace.save_scheme(path)
            results['atomspace'] = str(path)
        
        if 'toml' in formats:
            toml_hg = self.to_toml_hypergraph(include_weights=include_weights)
            path = output_path / f"{base_name}_hypergraph.toml"
            toml_hg.save_toml(path)
            results['toml'] = str(path)
        
        # Save summary
        summary_path = output_path / f"{base_name}_conversion_summary.json"
        summary = {
            "source_file": str(self.gguf_path),
            "model_info": self.model_info,
            "formats_generated": results,
            "include_weights": include_weights
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return results
