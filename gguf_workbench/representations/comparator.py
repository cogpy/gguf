"""
Comparator for different model representations.

Provides analysis and comparison of different representation formats,
evaluating them on completeness, transparency, and efficiency.
"""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class RepresentationMetrics:
    """Metrics for evaluating a representation."""
    name: str
    completeness_score: float  # 0-1, captures all model information
    transparency_score: float  # 0-1, human understandability
    efficiency_score: float  # 0-1, computational/storage efficiency
    expressiveness_score: float  # 0-1, ability to represent complex relationships
    tool_support_score: float  # 0-1, availability of analysis tools
    
    storage_size_kb: Optional[float] = None
    node_count: Optional[int] = None
    edge_count: Optional[int] = None
    
    strengths: List[str] = None
    weaknesses: List[str] = None
    use_cases: List[str] = None

    def __post_init__(self):
        if self.strengths is None:
            self.strengths = []
        if self.weaknesses is None:
            self.weaknesses = []
        if self.use_cases is None:
            self.use_cases = []

    def overall_score(self) -> float:
        """Compute overall weighted score."""
        return (
            self.completeness_score * 0.3 +
            self.transparency_score * 0.2 +
            self.efficiency_score * 0.2 +
            self.expressiveness_score * 0.2 +
            self.tool_support_score * 0.1
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['overall_score'] = self.overall_score()
        return d


class RepresentationComparator:
    """
    Compare different model representation formats.
    
    Evaluates representations on multiple dimensions:
    - Completeness: Does it capture all model information?
    - Transparency: Is it human-readable and understandable?
    - Efficiency: Storage size and computational overhead
    - Expressiveness: Can it represent complex relationships?
    - Tool Support: Availability of analysis and visualization tools
    """

    def __init__(self):
        """Initialize comparator."""
        self.metrics: Dict[str, RepresentationMetrics] = {}

    def add_metrics(self, metrics: RepresentationMetrics) -> None:
        """Add metrics for a representation."""
        self.metrics[metrics.name] = metrics

    def compare_all(self) -> Dict[str, Any]:
        """Generate comprehensive comparison."""
        comparison = {
            "representations": {},
            "rankings": {},
            "summary": {},
        }
        
        for name, metrics in self.metrics.items():
            comparison["representations"][name] = metrics.to_dict()
        
        # Rankings by dimension
        dimensions = [
            "completeness_score",
            "transparency_score",
            "efficiency_score",
            "expressiveness_score",
            "tool_support_score",
            "overall_score",
        ]
        
        for dim in dimensions:
            if dim == "overall_score":
                sorted_metrics = sorted(
                    self.metrics.items(),
                    key=lambda x: x[1].overall_score(),
                    reverse=True
                )
            else:
                sorted_metrics = sorted(
                    self.metrics.items(),
                    key=lambda x: getattr(x[1], dim),
                    reverse=True
                )
            comparison["rankings"][dim] = [name for name, _ in sorted_metrics]
        
        # Summary
        comparison["summary"] = self._generate_summary()
        
        return comparison

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comparison summary."""
        if not self.metrics:
            return {}
        
        best_overall = max(
            self.metrics.items(),
            key=lambda x: x[1].overall_score()
        )
        
        best_completeness = max(
            self.metrics.items(),
            key=lambda x: x[1].completeness_score
        )
        
        best_transparency = max(
            self.metrics.items(),
            key=lambda x: x[1].transparency_score
        )
        
        best_efficiency = max(
            self.metrics.items(),
            key=lambda x: x[1].efficiency_score
        )
        
        return {
            "best_overall": best_overall[0],
            "best_completeness": best_completeness[0],
            "best_transparency": best_transparency[0],
            "best_efficiency": best_efficiency[0],
            "total_representations": len(self.metrics),
        }

    def to_json(self, filepath: Optional[Path] = None, indent: int = 2) -> str:
        """Export comparison to JSON."""
        json_str = json.dumps(self.compare_all(), indent=indent)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        return json_str

    def to_markdown(self, filepath: Optional[Path] = None) -> str:
        """Export comparison to Markdown."""
        lines = ["# Model Representation Comparison"]
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        lines.append("Comparison of different representation formats for the TinyTransformer model.")
        lines.append("")
        
        # Summary table
        lines.append("## Score Summary")
        lines.append("")
        lines.append("| Representation | Completeness | Transparency | Efficiency | Expressiveness | Tool Support | Overall |")
        lines.append("|----------------|--------------|--------------|------------|----------------|--------------|---------|")
        
        for name, metrics in self.metrics.items():
            lines.append(
                f"| {name} | "
                f"{metrics.completeness_score:.2f} | "
                f"{metrics.transparency_score:.2f} | "
                f"{metrics.efficiency_score:.2f} | "
                f"{metrics.expressiveness_score:.2f} | "
                f"{metrics.tool_support_score:.2f} | "
                f"{metrics.overall_score():.2f} |"
            )
        
        lines.append("")
        
        # Detailed analysis
        lines.append("## Detailed Analysis")
        lines.append("")
        
        for name, metrics in self.metrics.items():
            lines.append(f"### {name}")
            lines.append("")
            
            if metrics.storage_size_kb:
                lines.append(f"**Storage Size**: {metrics.storage_size_kb:.2f} KB")
                lines.append("")
            
            if metrics.node_count:
                lines.append(f"**Node Count**: {metrics.node_count}")
                if metrics.edge_count:
                    lines.append(f"**Edge Count**: {metrics.edge_count}")
                lines.append("")
            
            if metrics.strengths:
                lines.append("**Strengths:**")
                for strength in metrics.strengths:
                    lines.append(f"- {strength}")
                lines.append("")
            
            if metrics.weaknesses:
                lines.append("**Weaknesses:**")
                for weakness in metrics.weaknesses:
                    lines.append(f"- {weakness}")
                lines.append("")
            
            if metrics.use_cases:
                lines.append("**Best Use Cases:**")
                for use_case in metrics.use_cases:
                    lines.append(f"- {use_case}")
                lines.append("")
        
        # Rankings
        comparison = self.compare_all()
        lines.append("## Rankings")
        lines.append("")
        
        for dimension, ranking in comparison["rankings"].items():
            dim_name = dimension.replace("_", " ").title()
            lines.append(f"**{dim_name}:**")
            for i, name in enumerate(ranking, 1):
                lines.append(f"{i}. {name}")
            lines.append("")
        
        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        summary = comparison["summary"]
        lines.append(f"- **For overall use**: {summary['best_overall']}")
        lines.append(f"- **For complete information**: {summary['best_completeness']}")
        lines.append(f"- **For human readability**: {summary['best_transparency']}")
        lines.append(f"- **For storage/performance**: {summary['best_efficiency']}")
        lines.append("")
        
        md_str = "\n".join(lines)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(md_str)
        
        return md_str

    @classmethod
    def create_default_comparison(cls) -> 'RepresentationComparator':
        """Create comparison with default metrics for common formats."""
        comp = cls()
        
        # Hypergraph representation
        comp.add_metrics(RepresentationMetrics(
            name="Hypergraph",
            completeness_score=1.0,
            transparency_score=0.7,
            efficiency_score=0.6,
            expressiveness_score=1.0,
            tool_support_score=0.5,
            strengths=[
                "Naturally represents multi-input/multi-output operations",
                "Captures complex dependencies in transformer architecture",
                "Enables advanced analysis (hypergraph cuts, clustering)",
                "Most expressive for neural network operations",
                "Single edge can represent attention operation (Q, K, V → output)",
            ],
            weaknesses=[
                "More complex than simple directed graphs",
                "Visualization can be challenging",
                "Limited tool support compared to standard graphs",
                "Some algorithms don't have hypergraph equivalents",
                "Steeper learning curve",
            ],
            use_cases=[
                "Analyzing complex multi-tensor operations",
                "Advanced model analysis and optimization",
                "Research on neural architecture search",
                "Understanding attention mechanism dependencies",
                "Graph-theoretic analysis of model structure",
            ],
        ))
        
        # DAG (Standard Graph) representation
        comp.add_metrics(RepresentationMetrics(
            name="DAG (Directed Graph)",
            completeness_score=0.95,
            transparency_score=0.85,
            efficiency_score=0.8,
            expressiveness_score=0.8,
            tool_support_score=0.95,
            strengths=[
                "Well-understood graph algorithms available",
                "Easy to visualize with standard tools (Graphviz, etc.)",
                "Topological ordering shows execution order",
                "Broad tool ecosystem (NetworkX, graph databases)",
                "Balance between simplicity and expressiveness",
            ],
            weaknesses=[
                "Multi-input operations require operation nodes",
                "More verbose than hypergraph (more nodes needed)",
                "Less natural for attention mechanism",
                "Operation semantics split between nodes and edges",
            ],
            use_cases=[
                "Standard model visualization and documentation",
                "Execution order analysis (topological sort)",
                "Dependency tracking and analysis",
                "Integration with existing graph tools",
                "Teaching and explaining model architecture",
            ],
        ))
        
        # Symbolic/Algebraic representation
        comp.add_metrics(RepresentationMetrics(
            name="Symbolic/Algebraic",
            completeness_score=0.9,
            transparency_score=0.95,
            efficiency_score=0.9,
            expressiveness_score=0.85,
            tool_support_score=0.7,
            strengths=[
                "Mathematical clarity and rigor",
                "Excellent for documentation and papers",
                "Easy to understand computational relationships",
                "Compact representation",
                "Natural for mathematical analysis and proofs",
                "Best for human understanding of computations",
            ],
            weaknesses=[
                "Less detailed for implementation specifics",
                "Harder to represent control flow",
                "Limited computational tool support",
                "Notation can be ambiguous",
                "Doesn't capture tensor shapes as clearly",
            ],
            use_cases=[
                "Academic papers and documentation",
                "Teaching transformer architecture",
                "Mathematical analysis and optimization",
                "Theoretical understanding of model",
                "Comparing model variants mathematically",
            ],
        ))
        
        # GGUF Binary format
        comp.add_metrics(RepresentationMetrics(
            name="GGUF (Binary)",
            completeness_score=1.0,
            transparency_score=0.2,
            efficiency_score=1.0,
            expressiveness_score=0.6,
            tool_support_score=0.6,
            strengths=[
                "Most compact storage format",
                "Fastest to load and process",
                "Complete weight information",
                "Optimized for inference",
                "Single-file deployment",
            ],
            weaknesses=[
                "Not human-readable",
                "Requires specialized tools to inspect",
                "Hard to understand structure",
                "Limited expressiveness for relationships",
                "Opaque to manual inspection",
            ],
            use_cases=[
                "Production model deployment",
                "Efficient model storage",
                "Inference with llama.cpp",
                "Model distribution",
                "Minimizing storage and bandwidth",
            ],
        ))
        
        # JSON representation
        comp.add_metrics(RepresentationMetrics(
            name="JSON",
            completeness_score=0.95,
            transparency_score=0.85,
            efficiency_score=0.5,
            expressiveness_score=0.7,
            tool_support_score=0.9,
            strengths=[
                "Human-readable text format",
                "Widely supported by tools and languages",
                "Easy to parse and generate",
                "Good for debugging and inspection",
                "Flexible structure",
            ],
            weaknesses=[
                "Verbose (large file size)",
                "Slower to parse than binary",
                "Limited type system",
                "Can be unwieldy for large models",
                "No schema enforcement (without JSON Schema)",
            ],
            use_cases=[
                "Model inspection and debugging",
                "Data exchange between tools",
                "Web APIs and services",
                "Configuration files",
                "Testing and validation",
            ],
        ))
        
        # TOML representation
        comp.add_metrics(RepresentationMetrics(
            name="TOML",
            completeness_score=0.9,
            transparency_score=0.9,
            efficiency_score=0.6,
            expressiveness_score=0.65,
            tool_support_score=0.7,
            strengths=[
                "Very human-readable",
                "Simpler than JSON for configuration",
                "Good for manual editing",
                "Clear structure with sections",
                "Better for configuration-style data",
            ],
            weaknesses=[
                "Less widely supported than JSON",
                "Not ideal for deeply nested data",
                "Limited tooling compared to JSON",
                "Verbose for large arrays",
                "Not as common in ML ecosystem",
            ],
            use_cases=[
                "Configuration files",
                "Manual model specification",
                "Human-editable model definitions",
                "Small to medium models",
                "Documentation and examples",
            ],
        ))
        
        # PyTorch format
        comp.add_metrics(RepresentationMetrics(
            name="PyTorch (.pth)",
            completeness_score=1.0,
            transparency_score=0.3,
            efficiency_score=0.85,
            expressiveness_score=0.7,
            tool_support_score=0.95,
            strengths=[
                "Native PyTorch format",
                "Excellent tool support in PyTorch ecosystem",
                "Efficient binary storage",
                "Complete model information",
                "Supports training and inference",
            ],
            weaknesses=[
                "PyTorch-specific (not portable)",
                "Binary format (not human-readable)",
                "Pickle-based (security concerns)",
                "Large file size compared to optimized formats",
                "Requires PyTorch to load",
            ],
            use_cases=[
                "PyTorch training and fine-tuning",
                "PyTorch-based inference",
                "Transfer learning",
                "Model checkpointing",
                "PyTorch ecosystem integration",
            ],
        ))
        
        # ONNX format
        comp.add_metrics(RepresentationMetrics(
            name="ONNX",
            completeness_score=0.95,
            transparency_score=0.6,
            efficiency_score=0.8,
            expressiveness_score=0.8,
            tool_support_score=0.85,
            strengths=[
                "Framework-independent",
                "Optimized computation graph",
                "Good tool support (ONNX Runtime, Netron)",
                "Production-ready",
                "Cross-platform deployment",
            ],
            weaknesses=[
                "Binary format (less readable than text)",
                "Can be complex for dynamic models",
                "Optimization can obscure original structure",
                "Learning curve for ONNX tools",
                "May lose some framework-specific features",
            ],
            use_cases=[
                "Cross-framework deployment",
                "Production inference with ONNX Runtime",
                "Model optimization",
                "Framework-agnostic model exchange",
                "Mobile and edge deployment",
            ],
        ))
        
        return comp
