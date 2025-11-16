"""
Inference tracer for detailed step-by-step execution visualization.

This module provides utilities to trace and visualize how the tiny transformer
processes input, showing exactly how weights are applied at each step.
"""

from typing import List, Dict, Any, Optional
import json


class InferenceTracer:
    """
    Tracer that records and displays every step of inference.
    
    This is useful for understanding:
    - How indexed weights are applied to inputs
    - How attention combines information across positions
    - How the vocabulary relates to outputs
    
    Example:
        >>> from gguf_workbench.inference import TinyTransformerListBased
        >>> tracer = InferenceTracer()
        >>> model = TinyTransformerListBased.from_json('tinytf/tiny_model_gguf.json')
        >>> tracer.trace_forward(model, [0, 1, 2])
        >>> print(tracer.get_report())
    """
    
    def __init__(self):
        """Initialize empty tracer."""
        self.steps = []
        self.current_step = 0
    
    def reset(self):
        """Clear all recorded steps."""
        self.steps = []
        self.current_step = 0
    
    def record_step(
        self, 
        step_name: str, 
        description: str, 
        data: Dict[str, Any]
    ):
        """Record a computation step."""
        self.steps.append({
            'step_number': self.current_step,
            'name': step_name,
            'description': description,
            'data': data,
        })
        self.current_step += 1
    
    def trace_embedding_lookup(
        self,
        token_ids: List[int],
        embedding_weights: List[List[float]],
    ):
        """Trace the embedding lookup operation."""
        for i, token_id in enumerate(token_ids):
            embedding = embedding_weights[token_id]
            
            self.record_step(
                step_name=f"embed_token_{i}",
                description=f"Look up embedding for token_{token_id} at position {i}",
                data={
                    'position': i,
                    'token_id': token_id,
                    'token_name': f'token_{token_id}',
                    'operation': 'embedding_matrix[token_id]',
                    'weight_row_index': token_id,
                    'embedding_vector': embedding,
                    'weight_application': 'Indexed lookup - select row from matrix',
                }
            )
    
    def trace_matrix_vector_multiply(
        self,
        step_name: str,
        matrix: List[List[float]],
        vector: List[float],
        result: List[float],
        description: str = "",
    ):
        """Trace a matrix-vector multiplication."""
        # Show how each output element is computed
        computations = []
        for i, row in enumerate(matrix):
            dot_prod = sum(a * b for a, b in zip(row, vector))
            computations.append({
                'output_index': i,
                'computation': f'sum(row[{i}] * vector)',
                'row': row,
                'vector': vector,
                'result': dot_prod,
                'detailed_steps': [
                    f'{row[j]:.4f} * {vector[j]:.4f} = {row[j] * vector[j]:.4f}'
                    for j in range(len(vector))
                ]
            })
        
        self.record_step(
            step_name=step_name,
            description=description or "Matrix-vector multiplication",
            data={
                'operation': 'matrix @ vector',
                'matrix_shape': f'[{len(matrix)}, {len(matrix[0]) if matrix else 0}]',
                'vector_shape': f'[{len(vector)}]',
                'result_shape': f'[{len(result)}]',
                'result': result,
                'computations': computations,
                'weight_application': 'Each output[i] = sum(matrix[i][j] * vector[j])',
            }
        )
    
    def trace_attention_scores(
        self,
        position: int,
        query: List[float],
        keys: List[List[float]],
        scores: List[float],
        scale: float,
    ):
        """Trace attention score computation."""
        score_details = []
        for j, key in enumerate(keys):
            dot_prod = sum(q * k for q, k in zip(query, key))
            scaled_score = dot_prod * scale
            
            score_details.append({
                'key_position': j,
                'dot_product': dot_prod,
                'scale_factor': scale,
                'scaled_score': scaled_score,
                'computation': f'dot(query, key[{j}]) * {scale:.4f} = {scaled_score:.4f}',
            })
        
        self.record_step(
            step_name=f"attention_scores_pos_{position}",
            description=f"Compute attention scores for position {position}",
            data={
                'position': position,
                'query': query,
                'keys': keys,
                'scores': scores,
                'scale_factor': scale,
                'score_details': score_details,
                'weight_application': 'Query-Key similarity with scaling',
            }
        )
    
    def trace_softmax(
        self,
        position: int,
        scores: List[float],
        weights: List[float],
    ):
        """Trace softmax computation."""
        import math
        
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        
        softmax_steps = []
        for i, (score, exp_score, weight) in enumerate(zip(scores, exp_scores, weights)):
            softmax_steps.append({
                'position': i,
                'raw_score': score,
                'shifted_score': score - max_score,
                'exp_value': exp_score,
                'softmax_weight': weight,
                'computation': f'exp({score:.4f} - {max_score:.4f}) / {sum_exp:.4f} = {weight:.4f}',
            })
        
        self.record_step(
            step_name=f"softmax_pos_{position}",
            description=f"Apply softmax to attention scores at position {position}",
            data={
                'position': position,
                'raw_scores': scores,
                'max_score': max_score,
                'exp_scores': exp_scores,
                'sum_exp': sum_exp,
                'weights': weights,
                'softmax_steps': softmax_steps,
                'weight_application': 'Normalize scores to probabilities',
            }
        )
    
    def trace_weighted_sum(
        self,
        position: int,
        values: List[List[float]],
        weights: List[float],
        result: List[float],
    ):
        """Trace weighted sum of values."""
        combination_details = []
        for j, (value, weight) in enumerate(zip(values, weights)):
            weighted_value = [v * weight for v in value]
            combination_details.append({
                'value_position': j,
                'weight': weight,
                'value': value,
                'weighted_value': weighted_value,
                'contribution': f'{weight:.4f} * value[{j}]',
            })
        
        self.record_step(
            step_name=f"weighted_sum_pos_{position}",
            description=f"Compute weighted sum of values at position {position}",
            data={
                'position': position,
                'weights': weights,
                'values': values,
                'result': result,
                'combination_details': combination_details,
                'weight_application': 'Combine values using attention weights',
            }
        )
    
    def get_step(self, step_number: int) -> Dict[str, Any]:
        """Get a specific step by number."""
        if 0 <= step_number < len(self.steps):
            return self.steps[step_number]
        return None
    
    def get_step_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get first step matching the name."""
        for step in self.steps:
            if step['name'] == name:
                return step
        return None
    
    def get_all_steps(self) -> List[Dict[str, Any]]:
        """Get all recorded steps."""
        return self.steps
    
    def get_report(self, verbose: bool = True) -> str:
        """
        Generate a formatted report of all steps.
        
        Args:
            verbose: Include detailed computations
            
        Returns:
            Formatted string report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("INFERENCE TRACE REPORT")
        lines.append("=" * 80)
        lines.append(f"\nTotal steps: {len(self.steps)}")
        lines.append("")
        
        for step in self.steps:
            lines.append("-" * 80)
            lines.append(f"Step {step['step_number']}: {step['name']}")
            lines.append(f"Description: {step['description']}")
            lines.append("")
            
            data = step['data']
            
            # Show key information
            if 'operation' in data:
                lines.append(f"Operation: {data['operation']}")
            
            if 'weight_application' in data:
                lines.append(f"Weight Application: {data['weight_application']}")
            
            # Show result
            if 'result' in data:
                result = data['result']
                if isinstance(result, list) and len(result) <= 10:
                    lines.append(f"Result: [{', '.join(f'{v:.4f}' for v in result)}]")
                elif isinstance(result, (int, float)):
                    lines.append(f"Result: {result:.4f}")
            
            if verbose:
                # Show detailed computations for some step types
                if 'computations' in data:
                    lines.append("\nDetailed computations:")
                    for comp in data['computations'][:3]:  # Show first 3
                        lines.append(f"  Output[{comp['output_index']}]:")
                        for detail in comp['detailed_steps'][:3]:  # Show first 3
                            lines.append(f"    {detail}")
                    if len(data['computations']) > 3:
                        lines.append(f"  ... and {len(data['computations']) - 3} more")
                
                if 'score_details' in data:
                    lines.append("\nScore details:")
                    for detail in data['score_details']:
                        lines.append(f"  {detail['computation']}")
                
                if 'softmax_steps' in data:
                    lines.append("\nSoftmax computation:")
                    for step_detail in data['softmax_steps']:
                        lines.append(f"  Position {step_detail['position']}: {step_detail['computation']}")
            
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def export_json(self, filepath: str):
        """Export trace to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                'total_steps': len(self.steps),
                'steps': self.steps,
            }, f, indent=2)
    
    def visualize_attention_flow(
        self,
        attention_weights: List[List[float]],
        token_names: List[str],
    ) -> str:
        """
        Create ASCII visualization of attention flow.
        
        Args:
            attention_weights: [seq_len, seq_len] attention matrix
            token_names: Names of tokens at each position
            
        Returns:
            ASCII art showing attention flow
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ATTENTION FLOW VISUALIZATION")
        lines.append("=" * 80)
        lines.append("")
        
        seq_len = len(attention_weights)
        
        for i in range(seq_len):
            lines.append(f"\nPosition {i} ({token_names[i]}) attends to:")
            weights = attention_weights[i]
            
            # Create bar chart
            for j in range(seq_len):
                weight = weights[j]
                bar_len = int(weight * 50)  # Scale to 50 chars max
                bar = "█" * bar_len
                lines.append(f"  {token_names[j]:12s} [{weight:.4f}] {bar}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def show_weight_application_summary(self) -> str:
        """
        Show summary of how weights are applied throughout inference.
        
        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("WEIGHT APPLICATION SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        
        # Group steps by weight application type
        applications = {}
        for step in self.steps:
            if 'weight_application' in step['data']:
                app_type = step['data']['weight_application']
                if app_type not in applications:
                    applications[app_type] = []
                applications[app_type].append(step['name'])
        
        for app_type, step_names in applications.items():
            lines.append(f"\n{app_type}:")
            lines.append(f"  Used in {len(step_names)} step(s)")
            lines.append(f"  Examples: {', '.join(step_names[:3])}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
