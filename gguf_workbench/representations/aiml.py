"""
AIML (Artificial Intelligence Markup Language) representation for transformer models.

AIML is a pattern-matching language used by Pandorabots and other chatbot systems.
This module represents the tiny transformer as AIML categories that can encode
knowledge from the model's structure and behavior.
"""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom


class AIMLRepresentation:
    """
    AIML representation of a transformer model.
    
    Represents the model structure as AIML categories with patterns for:
    - Model architecture queries
    - Layer information
    - Weight information
    - Token embeddings
    - Inference patterns
    
    This format is useful for:
    - Creating chatbot interfaces to explain the model
    - Pattern-based model inspection
    - Educational demonstrations
    - Integration with Pandorabot systems
    """
    
    def __init__(self):
        """Initialize empty AIML representation."""
        self.categories: List[Dict[str, str]] = []
        self.metadata: Dict[str, Any] = {}
        self.topic: str = "tinytransformer"
    
    def add_category(self, pattern: str, template: str, that: Optional[str] = None) -> None:
        """
        Add an AIML category (pattern-response pair).
        
        Args:
            pattern: Input pattern to match (user query)
            template: Response template
            that: Optional context pattern (previous bot response)
        """
        category = {
            "pattern": pattern.upper(),  # AIML patterns are uppercase
            "template": template
        }
        if that:
            category["that"] = that.upper()
        self.categories.append(category)
    
    def add_architecture_categories(self, metadata: Dict[str, Any]) -> None:
        """Add categories for architecture queries."""
        # Model name
        self.add_category(
            "WHAT IS THE MODEL NAME",
            f"The model name is {metadata.get('model_name', 'TinyTransformer')}."
        )
        
        # Architecture type
        self.add_category(
            "WHAT IS THE ARCHITECTURE",
            f"This is a {metadata.get('architecture', 'transformer')} architecture model."
        )
        
        self.add_category(
            "WHAT KIND OF MODEL IS THIS",
            f"This is a tiny transformer model with {metadata.get('total_parameters', 200)} parameters."
        )
        
        # Dimensions
        self.add_category(
            "WHAT IS THE EMBEDDING DIMENSION",
            f"The embedding dimension is {metadata.get('embedding_dim', 5)}."
        )
        
        self.add_category(
            "HOW MANY LAYERS",
            f"The model has {metadata.get('num_blocks', 1)} transformer block(s)."
        )
        
        self.add_category(
            "HOW MANY ATTENTION HEADS",
            f"The model has {metadata.get('num_heads', 1)} attention head(s)."
        )
        
        self.add_category(
            "WHAT IS THE VOCABULARY SIZE",
            f"The vocabulary size is {metadata.get('vocab_size', 10)} tokens."
        )
        
        self.add_category(
            "WHAT IS THE CONTEXT LENGTH",
            f"The maximum context length is {metadata.get('context_length', 5)} tokens."
        )
    
    def add_layer_categories(self) -> None:
        """Add categories for layer information."""
        self.add_category(
            "WHAT LAYERS DOES THE MODEL HAVE",
            "The model has an embedding layer, self-attention layer, feed-forward network, and output layer."
        )
        
        self.add_category(
            "DESCRIBE THE EMBEDDING LAYER",
            "The embedding layer maps input tokens to dense vectors. It has a 10x5 weight matrix mapping 10 tokens to 5-dimensional embeddings."
        )
        
        self.add_category(
            "DESCRIBE THE ATTENTION LAYER",
            "The self-attention layer has query, key, and value projections with a single attention head. It computes weighted combinations of token representations."
        )
        
        self.add_category(
            "DESCRIBE THE FEEDFORWARD LAYER",
            "The feed-forward network consists of two linear transformations with dimension 5. It processes each position independently."
        )
        
        self.add_category(
            "DESCRIBE THE OUTPUT LAYER",
            "The output layer projects the 5-dimensional representations back to vocabulary size (10) to produce token predictions."
        )
    
    def add_operation_categories(self) -> None:
        """Add categories for understanding operations."""
        self.add_category(
            "HOW DOES ATTENTION WORK",
            "Attention computes query, key, and value matrices, then calculates attention scores by comparing queries and keys. These scores weight the values to produce the output."
        )
        
        self.add_category(
            "WHAT IS SELF ATTENTION",
            "Self-attention allows each position in the sequence to attend to all positions, capturing dependencies between tokens."
        )
        
        self.add_category(
            "HOW DO YOU RUN INFERENCE",
            "Inference flows through: token embedding, self-attention, feed-forward network, and output projection. The result is token probabilities."
        )
        
        self.add_category(
            "WHAT IS A TRANSFORMER",
            "A transformer is a neural network architecture based on self-attention mechanisms. It processes sequences in parallel rather than sequentially."
        )
    
    def add_technical_categories(self) -> None:
        """Add technical query categories."""
        self.add_category(
            "WHAT ARE THE MODEL PARAMETERS",
            "The model has approximately 200 parameters split across: embedding (50), attention (75), feed-forward (50), and output (50) layers."
        )
        
        self.add_category(
            "WHAT FORMATS ARE SUPPORTED",
            "The model is available in GGUF, PyTorch, ONNX, JSON, TOML, AIML, OpenCog AtomSpace, and hypergraph formats."
        )
        
        self.add_category(
            "HOW CAN I USE THIS MODEL",
            "You can load it with the gguf-workbench Python API, use it for educational purposes, or study its structure in various representation formats."
        )
    
    def add_conversational_categories(self) -> None:
        """Add conversational/interactive categories."""
        self.add_category(
            "HELLO",
            "Hello! I am the TinyTransformer model. Ask me about my architecture, layers, or how I work."
        )
        
        self.add_category(
            "HELP",
            "You can ask me about: my architecture, layers, attention mechanism, parameters, or supported formats. Try 'what is the architecture' or 'describe the attention layer'."
        )
        
        self.add_category(
            "THANK YOU",
            "You're welcome! Let me know if you have more questions about the model."
        )
    
    @classmethod
    def from_tiny_transformer(cls, model_path: Optional[Path] = None) -> "AIMLRepresentation":
        """
        Create AIML representation from tiny transformer model.
        
        Args:
            model_path: Optional path to model file for loading actual weights
        
        Returns:
            AIMLRepresentation of the model
        """
        aiml = cls()
        
        # Set metadata
        aiml.metadata = {
            "model_name": "TinyTransformer",
            "architecture": "transformer",
            "vocab_size": 10,
            "embedding_dim": 5,
            "context_length": 5,
            "num_blocks": 1,
            "num_heads": 1,
            "feedforward_dim": 5,
            "total_parameters": 200,
        }
        
        # Add various category groups
        aiml.add_architecture_categories(aiml.metadata)
        aiml.add_layer_categories()
        aiml.add_operation_categories()
        aiml.add_technical_categories()
        aiml.add_conversational_categories()
        
        return aiml
    
    def to_xml(self) -> str:
        """
        Convert to AIML XML format.
        
        Returns:
            AIML XML string
        """
        # Create root element
        aiml_root = ET.Element("aiml", version="2.0")
        
        for category_data in self.categories:
            # Create category element
            category = ET.SubElement(aiml_root, "category")
            
            # Add pattern
            pattern = ET.SubElement(category, "pattern")
            pattern.text = category_data["pattern"]
            
            # Add that if present (context)
            if "that" in category_data:
                that = ET.SubElement(category, "that")
                that.text = category_data["that"]
            
            # Add template (response)
            template = ET.SubElement(category, "template")
            template.text = category_data["template"]
        
        # Convert to pretty XML string
        xml_str = ET.tostring(aiml_root, encoding="unicode")
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent="  ")
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert to JSON representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "metadata": self.metadata,
            "topic": self.topic,
            "categories": self.categories,
            "category_count": len(self.categories)
        }
    
    def save_xml(self, path: Path) -> None:
        """Save AIML to XML file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_xml())
    
    def save_json(self, path: Path) -> None:
        """Save AIML categories to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)
