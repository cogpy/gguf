"""
Symbolic/algebraic representation for transformer models.

Represents the model as mathematical expressions and equations,
making the computational structure explicit and analyzable.
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class SymbolicExpression:
    """A symbolic mathematical expression."""

    name: str
    expression: str
    dependencies: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class Parameter:
    """A symbolic parameter."""

    name: str
    shape: tuple
    dtype: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


class SymbolicRepresentation:
    """
    Symbolic/algebraic representation of a transformer model.

    Represents the model as a system of mathematical equations,
    showing how each tensor is computed from others.

    Advantages:
    - Mathematical clarity and rigor
    - Easy to understand computational relationships
    - Enables symbolic manipulation and analysis
    - Good for documentation and teaching
    - Facilitates mathematical optimization

    Disadvantages:
    - Less detailed than graph representations
    - Harder to represent implementation details
    - Limited computational tool support
    - Notation can be complex for large models
    """

    def __init__(self):
        """Initialize empty symbolic representation."""
        self.parameters: Dict[str, Parameter] = {}
        self.expressions: Dict[str, SymbolicExpression] = {}
        self.metadata: Dict[str, Any] = {}
        self.notation: Dict[str, str] = {}

    def add_parameter(self, param: Parameter) -> None:
        """Add a parameter to the representation."""
        self.parameters[param.name] = param

    def add_expression(self, expr: SymbolicExpression) -> None:
        """Add an expression to the representation."""
        self.expressions[expr.name] = expr

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metadata": self.metadata,
            "notation": self.notation,
            "parameters": {name: param.to_dict() for name, param in self.parameters.items()},
            "expressions": {name: expr.to_dict() for name, expr in self.expressions.items()},
            "statistics": self.get_statistics(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Compute statistics."""
        total_params = sum(
            int(p.shape[0]) * int(p.shape[1]) if len(p.shape) == 2 else int(p.shape[0])
            for p in self.parameters.values()
        )

        return {
            "parameter_count": len(self.parameters),
            "expression_count": len(self.expressions),
            "total_parameters": total_params,
        }

    def to_json(self, filepath: Optional[Path] = None, indent: int = 2) -> str:
        """Export to JSON format."""
        json_str = json.dumps(self.to_dict(), indent=indent)
        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
        return json_str

    def to_latex(self, filepath: Optional[Path] = None) -> str:
        """
        Export to LaTeX format for mathematical documentation.
        """
        lines = ["\\documentclass{article}"]
        lines.append("\\usepackage{amsmath}")
        lines.append("\\usepackage{amssymb}")
        lines.append("\\begin{document}")
        lines.append("")
        lines.append("\\section{TinyTransformer Symbolic Representation}")
        lines.append("")

        # Notation
        lines.append("\\subsection{Notation}")
        lines.append("\\begin{itemize}")
        for symbol, meaning in self.notation.items():
            lines.append(f"  \\item ${symbol}$: {meaning}")
        lines.append("\\end{itemize}")
        lines.append("")

        # Parameters
        lines.append("\\subsection{Parameters}")
        lines.append("\\begin{align}")
        for param in self.parameters.values():
            shape_str = " \\times ".join(str(d) for d in param.shape)
            lines.append(
                f"  {param.name} &\\in \\mathbb{{R}}^{{{shape_str}}} "
                f"\\quad \\text{{{param.description}}} \\\\"
            )
        lines.append("\\end{align}")
        lines.append("")

        # Expressions
        lines.append("\\subsection{Forward Pass}")
        lines.append("\\begin{align}")
        for expr in self.expressions.values():
            lines.append(f"  {expr.name} &= {expr.expression} \\\\")
        lines.append("\\end{align}")
        lines.append("")

        lines.append("\\end{document}")
        latex_str = "\n".join(lines)

        if filepath:
            with open(filepath, "w") as f:
                f.write(latex_str)

        return latex_str

    @classmethod
    def from_tiny_transformer(cls, model_path: Optional[Path] = None) -> "SymbolicRepresentation":
        """
        Create symbolic representation from tiny transformer model.
        """
        sr = cls()

        sr.metadata = {
            "model_name": "TinyTransformer",
            "architecture": "transformer",
            "representation": "symbolic",
            "vocab_size": 10,
            "embedding_dim": 5,
            "context_length": 5,
            "num_blocks": 1,
            "num_heads": 1,
        }

        # Notation guide
        sr.notation = {
            "x": "Input token IDs, x ∈ {0,1,...,9}^L where L=5",
            "E": "Embedding matrix (vocab_size × d_model)",
            "h": "Hidden state / embeddings",
            "Q, K, V": "Query, Key, Value matrices",
            "W^Q, W^K, W^V": "Query, Key, Value projection weights",
            "W^O": "Attention output projection",
            "A": "Attention weights (after softmax)",
            "W^{ff}_1, W^{ff}_2": "Feed-forward network weights",
            "W^{out}": "Output projection to vocabulary",
            "LN": "Layer normalization",
            "d_k": "Dimension of keys (= d_model / num_heads = 5)",
            "L": "Sequence length (= 5)",
            "d_{model}": "Model dimension (= 5)",
            "d_{ff}": "Feed-forward dimension (= 5)",
            "V": "Vocabulary size (= 10)",
        }

        # Parameters
        sr.add_parameter(
            Parameter(
                name="E", shape=(10, 5), dtype="float32", description="Token embedding matrix"
            )
        )

        sr.add_parameter(
            Parameter(
                name="W^Q", shape=(5, 5), dtype="float32", description="Query projection weights"
            )
        )

        sr.add_parameter(
            Parameter(
                name="W^K", shape=(5, 5), dtype="float32", description="Key projection weights"
            )
        )

        sr.add_parameter(
            Parameter(
                name="W^V", shape=(5, 5), dtype="float32", description="Value projection weights"
            )
        )

        sr.add_parameter(
            Parameter(
                name="W^O", shape=(5, 5), dtype="float32", description="Attention output projection"
            )
        )

        sr.add_parameter(
            Parameter(
                name="\\gamma_1", shape=(5,), dtype="float32", description="Layer norm 1 scale"
            )
        )

        sr.add_parameter(
            Parameter(name="\\beta_1", shape=(5,), dtype="float32", description="Layer norm 1 bias")
        )

        sr.add_parameter(
            Parameter(
                name="W^{ff}_1",
                shape=(5, 5),
                dtype="float32",
                description="FFN first layer weights",
            )
        )

        sr.add_parameter(
            Parameter(
                name="W^{ff}_2",
                shape=(5, 5),
                dtype="float32",
                description="FFN second layer weights",
            )
        )

        sr.add_parameter(
            Parameter(
                name="W^{out}",
                shape=(5, 10),
                dtype="float32",
                description="Output projection to vocabulary",
            )
        )

        # Expressions (forward pass)
        sr.add_expression(
            SymbolicExpression(
                name="h_0",
                expression="E[x]",
                dependencies=["E", "x"],
                properties={
                    "description": "Embedding lookup",
                    "shape": "(batch, L, d_model)",
                    "latex": "h_0 = E[x] \\in \\mathbb{R}^{L \\times d_{model}}",
                },
            )
        )

        sr.add_expression(
            SymbolicExpression(
                name="Q",
                expression="h_0 W^Q",
                dependencies=["h_0", "W^Q"],
                properties={
                    "description": "Query projection",
                    "shape": "(batch, L, d_k)",
                    "latex": "Q = h_0 W^Q",
                },
            )
        )

        sr.add_expression(
            SymbolicExpression(
                name="K",
                expression="h_0 W^K",
                dependencies=["h_0", "W^K"],
                properties={
                    "description": "Key projection",
                    "shape": "(batch, L, d_k)",
                    "latex": "K = h_0 W^K",
                },
            )
        )

        sr.add_expression(
            SymbolicExpression(
                name="V",
                expression="h_0 W^V",
                dependencies=["h_0", "W^V"],
                properties={
                    "description": "Value projection",
                    "shape": "(batch, L, d_k)",
                    "latex": "V = h_0 W^V",
                },
            )
        )

        sr.add_expression(
            SymbolicExpression(
                name="S",
                expression="QK^T / sqrt(d_k)",
                dependencies=["Q", "K"],
                properties={
                    "description": "Scaled attention scores",
                    "shape": "(batch, L, L)",
                    "latex": "S = \\frac{QK^T}{\\sqrt{d_k}}",
                },
            )
        )

        sr.add_expression(
            SymbolicExpression(
                name="A",
                expression="softmax(S)",
                dependencies=["S"],
                properties={
                    "description": "Attention weights",
                    "shape": "(batch, L, L)",
                    "latex": "A = \\text{softmax}(S)",
                },
            )
        )

        sr.add_expression(
            SymbolicExpression(
                name="h_attn",
                expression="(AV)W^O",
                dependencies=["A", "V", "W^O"],
                properties={
                    "description": "Attention output",
                    "shape": "(batch, L, d_model)",
                    "latex": "h_{\\text{attn}} = (AV)W^O",
                },
            )
        )

        sr.add_expression(
            SymbolicExpression(
                name="h_1",
                expression="LN(h_0 + h_attn; gamma_1, beta_1)",
                dependencies=["h_0", "h_attn", "\\gamma_1", "\\beta_1"],
                properties={
                    "description": "Post-attention with residual and layer norm",
                    "shape": "(batch, L, d_model)",
                    "latex": "h_1 = \\text{LN}(h_0 + h_{\\text{attn}}; \\gamma_1, \\beta_1)",
                },
            )
        )

        sr.add_expression(
            SymbolicExpression(
                name="h_ff",
                expression="GELU(h_1 W^{ff}_1) W^{ff}_2",
                dependencies=["h_1", "W^{ff}_1", "W^{ff}_2"],
                properties={
                    "description": "Feed-forward network",
                    "shape": "(batch, L, d_model)",
                    "latex": "h_{\\text{ff}} = \\text{GELU}(h_1 W^{ff}_1) W^{ff}_2",
                },
            )
        )

        sr.add_expression(
            SymbolicExpression(
                name="h_2",
                expression="h_1 + h_ff",
                dependencies=["h_1", "h_ff"],
                properties={
                    "description": "Post-FFN with residual",
                    "shape": "(batch, L, d_model)",
                    "latex": "h_2 = h_1 + h_{\\text{ff}}",
                },
            )
        )

        sr.add_expression(
            SymbolicExpression(
                name="logits",
                expression="h_2 W^{out}",
                dependencies=["h_2", "W^{out}"],
                properties={
                    "description": "Output logits over vocabulary",
                    "shape": "(batch, L, vocab_size)",
                    "latex": "\\text{logits} = h_2 W^{\\text{out}}",
                },
            )
        )

        sr.add_expression(
            SymbolicExpression(
                name="y",
                expression="argmax(logits)",
                dependencies=["logits"],
                properties={
                    "description": "Predicted tokens",
                    "shape": "(batch, L)",
                    "latex": "y = \\arg\\max(\\text{logits})",
                },
            )
        )

        return sr

    def export_markdown(self, filepath: Optional[Path] = None) -> str:
        """Export to Markdown format for documentation."""
        lines = ["# TinyTransformer - Symbolic Representation"]
        lines.append("")
        lines.append("## Model Architecture")
        lines.append("")
        for key, value in self.metadata.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

        lines.append("## Notation")
        lines.append("")
        for symbol, meaning in self.notation.items():
            lines.append(f"- `{symbol}`: {meaning}")
        lines.append("")

        lines.append("## Parameters")
        lines.append("")
        lines.append("| Parameter | Shape | Type | Description |")
        lines.append("|-----------|-------|------|-------------|")
        for param in self.parameters.values():
            shape_str = " × ".join(str(d) for d in param.shape)
            lines.append(f"| {param.name} | {shape_str} | {param.dtype} | {param.description} |")
        lines.append("")

        lines.append("## Forward Pass Equations")
        lines.append("")
        for i, expr in enumerate(self.expressions.values(), 1):
            lines.append(f"{i}. **{expr.name}** = `{expr.expression}`")
            if "description" in expr.properties:
                lines.append(f"   - {expr.properties['description']}")
            if "shape" in expr.properties:
                lines.append(f"   - Shape: {expr.properties['shape']}")
            lines.append("")

        md_str = "\n".join(lines)

        if filepath:
            with open(filepath, "w") as f:
                f.write(md_str)

        return md_str
