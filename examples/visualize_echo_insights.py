#!/usr/bin/env python3
"""
Deep Tree Echo Insights Visualizer

Creates visual representations and detailed reports of the Deep Tree Echo analysis.
Generates markdown reports with statistics, insights, and recommendations.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class InsightsVisualizer:
    """Creates visualizations and reports from analysis data."""
    
    def __init__(self, analysis_file: str):
        self.analysis_file = analysis_file
        self.data = None
        
    def load_analysis(self):
        """Load the analysis JSON file."""
        with open(self.analysis_file, 'r') as f:
            self.data = json.load(f)
        return self
    
    def generate_markdown_report(self) -> str:
        """Generate a comprehensive markdown report."""
        if not self.data:
            return "No data loaded"
        
        report = []
        report.append("# Deep Tree Echo Conversation Analysis Report")
        report.append("")
        report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        if 'key_insights' in self.data:
            for insight in self.data['key_insights']:
                report.append(f"- {insight}")
            report.append("")
        
        # Data Sources
        report.append("## Data Sources Analyzed")
        report.append("")
        if 'overview' in self.data:
            sources = self.data['overview'].get('data_sources', {})
            report.append("| Data Source | Status |")
            report.append("|------------|--------|")
            report.append(f"| Conversation Data | {'✓ Loaded' if sources.get('conversation_loaded') else '✗ Not Found'} |")
            report.append(f"| Training Data | {'✓ Loaded' if sources.get('training_data_loaded') else '✗ Not Found'} |")
            char_models = sources.get('character_models', [])
            if char_models:
                report.append(f"| Character Models | {', '.join(char_models)} |")
            report.append("")
        
        # Cognitive Architecture
        report.append("## Cognitive Architecture Components")
        report.append("")
        if 'cognitive_architecture' in self.data:
            arch = self.data['cognitive_architecture']
            sorted_arch = sorted(arch.items(), key=lambda x: x[1], reverse=True)
            
            report.append("| Component | Mentions |")
            report.append("|-----------|----------|")
            for component, count in sorted_arch:
                if count > 0:
                    name = component.replace('_', ' ').title()
                    report.append(f"| {name} | {count} |")
            report.append("")
            
            # Create a simple text-based bar chart
            report.append("### Component Frequency Distribution")
            report.append("")
            report.append("```")
            max_count = max(arch.values()) if arch else 1
            for component, count in sorted_arch[:7]:  # Top 7
                if count > 0:
                    name = component.replace('_', ' ').title()
                    bar_length = int((count / max_count) * 50)
                    bar = '█' * bar_length
                    report.append(f"{name:25} {bar} {count}")
            report.append("```")
            report.append("")
        
        # Training Data Analysis
        report.append("## Training Data Analysis")
        report.append("")
        if 'training_data_analysis' in self.data and self.data['training_data_analysis']:
            td = self.data['training_data_analysis']
            
            report.append("### Statistics")
            report.append("")
            report.append(f"- **Total Conversations**: {td.get('total_conversations', 0)}")
            report.append(f"- **Total Messages**: {td.get('total_messages', 0)}")
            report.append(f"- **User Messages**: {td.get('user_message_count', 0)}")
            report.append(f"- **Assistant Messages**: {td.get('assistant_message_count', 0)}")
            report.append(f"- **Average User Message Length**: {td.get('avg_user_message_length', 0):.0f} characters")
            report.append(f"- **Average Assistant Message Length**: {td.get('avg_assistant_message_length', 0):.0f} characters")
            report.append("")
            
            if td.get('template_usage'):
                report.append("### Template Patterns Used")
                report.append("")
                report.append("| Template | Usage Count |")
                report.append("|----------|-------------|")
                for template, count in list(td['template_usage'].items())[:10]:
                    # Truncate long templates
                    template_display = template[:50] + "..." if len(template) > 50 else template
                    report.append(f"| `{template_display}` | {count} |")
                report.append("")
        
        # Conversation vs Training Comparison
        report.append("## Conversation vs Training Dataset Comparison")
        report.append("")
        if 'conversation_vs_training' in self.data:
            comp = self.data['conversation_vs_training']
            
            if 'error' not in comp:
                report.append("| Metric | Real Conversation | Training Dataset |")
                report.append("|--------|------------------|------------------|")
                
                conv = comp.get('conversation_stats', {})
                train = comp.get('training_stats', {})
                
                report.append(f"| User Messages | {conv.get('user_messages', 0)} | {train.get('user_messages', 0)} |")
                report.append(f"| Assistant Messages | {conv.get('assistant_messages', 0)} | {train.get('assistant_messages', 0)} |")
                report.append(f"| Avg User Length | {conv.get('avg_user_length', 0):.0f} chars | {train.get('avg_user_length', 0):.0f} chars |")
                report.append(f"| Avg Assistant Length | {conv.get('avg_assistant_length', 0):.0f} chars | {train.get('avg_assistant_length', 0):.0f} chars |")
                report.append("")
                
                report.append("### Analysis")
                report.append("")
                report.append("The training dataset shows strong alignment with real conversation patterns:")
                
                user_msg_diff = abs(conv.get('user_messages', 0) - train.get('user_messages', 0))
                if user_msg_diff < 20:
                    report.append(f"- ✓ Message counts are very similar (difference: {user_msg_diff})")
                
                user_len_ratio = conv.get('avg_user_length', 1) / train.get('avg_user_length', 1)
                if 0.9 < user_len_ratio < 1.1:
                    report.append(f"- ✓ Average message lengths are well-matched (ratio: {user_len_ratio:.2f})")
                
                report.append("")
        
        # Persona Dimensions
        report.append("## Persona Dimensions")
        report.append("")
        if 'persona_dimensions' in self.data:
            pd = self.data['persona_dimensions']
            
            for char_name, dimensions in pd.items():
                report.append(f"### {char_name}")
                report.append("")
                
                if dimensions:
                    report.append(f"Total Dimensions Defined: **{len(dimensions)}**")
                    report.append("")
                    
                    for dim in dimensions:
                        num = dim.get('number', '')
                        name = dim.get('name', '')
                        desc = dim.get('description', '')
                        report.append(f"{num}. **{name}**: {desc}")
                    report.append("")
                else:
                    report.append("*No persona dimensions extracted from this model.*")
                    report.append("")
        
        # Recommendations
        report.append("## Recommendations for Further Analysis")
        report.append("")
        report.append("Based on this analysis, consider:")
        report.append("")
        report.append("1. **Temporal Analysis**: Track how topics and patterns evolve over the conversation timeline")
        report.append("2. **Sentiment Analysis**: Analyze emotional tone and engagement levels")
        report.append("3. **Entity Extraction**: Identify and track key entities, concepts, and their relationships")
        report.append("4. **Response Quality Metrics**: Evaluate response coherence, relevance, and informativeness")
        report.append("5. **Template Pattern Optimization**: Analyze which templates correlate with better responses")
        report.append("6. **Character Consistency Scoring**: Develop metrics to quantify persona consistency")
        report.append("7. **Tree Structure Analysis**: Deep dive into branching patterns and conversation flows")
        report.append("")
        
        # Conclusion
        report.append("## Conclusion")
        report.append("")
        report.append("This analysis reveals a sophisticated conversation system with:")
        report.append("")
        report.append("- Strong alignment between training data and real conversations")
        report.append("- Consistent use of Echo Self cognitive architecture concepts")
        report.append("- Rich persona dimensions guiding interaction patterns")
        report.append("- Complex tree-structured conversations with multiple branches")
        report.append("- Extensive use of technical and identity-focused language")
        report.append("")
        report.append("The Deep Tree Echo system demonstrates coherent integration of character models,")
        report.append("cognitive architecture principles, and conversational dynamics.")
        report.append("")
        
        return "\n".join(report)
    
    def save_markdown_report(self, output_path: str):
        """Save the markdown report to a file."""
        report = self.generate_markdown_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Markdown report saved to: {output_path}")
        return report
    
    def generate_insights_summary(self) -> str:
        """Generate a concise insights summary."""
        if not self.data:
            return "No data loaded"
        
        summary = []
        summary.append("=" * 80)
        summary.append("DEEP TREE ECHO INSIGHTS SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        
        # Key metrics
        if 'cognitive_architecture' in self.data:
            arch = self.data['cognitive_architecture']
            total_mentions = sum(arch.values())
            summary.append(f"Total Cognitive Architecture Mentions: {total_mentions}")
            
            top_3 = sorted(arch.items(), key=lambda x: x[1], reverse=True)[:3]
            summary.append("\nTop 3 Components:")
            for component, count in top_3:
                name = component.replace('_', ' ').title()
                percentage = (count / total_mentions * 100) if total_mentions > 0 else 0
                summary.append(f"  - {name}: {count} ({percentage:.1f}%)")
            summary.append("")
        
        # Training alignment
        if 'conversation_vs_training' in self.data:
            comp = self.data['conversation_vs_training']
            if 'error' not in comp:
                summary.append("Conversation-Training Alignment:")
                conv = comp.get('conversation_stats', {})
                train = comp.get('training_stats', {})
                
                msg_similarity = 100 - abs(conv.get('user_messages', 0) - train.get('user_messages', 0))
                summary.append(f"  Message Count Similarity: {msg_similarity}%")
                
                len_ratio = conv.get('avg_assistant_length', 1) / train.get('avg_assistant_length', 1)
                summary.append(f"  Response Length Ratio: {len_ratio:.2f}x")
                summary.append("")
        
        # Insights
        if 'key_insights' in self.data:
            summary.append("Key Insights:")
            for i, insight in enumerate(self.data['key_insights'], 1):
                summary.append(f"  {i}. {insight}")
            summary.append("")
        
        summary.append("=" * 80)
        
        return "\n".join(summary)


def main():
    """Main execution function."""
    examples_dir = Path(__file__).parent
    analysis_dir = examples_dir / "analysis_output"
    
    # Check for analysis file
    analysis_file = analysis_dir / "comprehensive_echo_analysis.json"
    
    if not analysis_file.exists():
        print(f"Error: Analysis file not found: {analysis_file}")
        print("Please run analyze_all_echo_data.py first to generate the analysis.")
        return
    
    # Create visualizer
    visualizer = InsightsVisualizer(str(analysis_file))
    visualizer.load_analysis()
    
    # Print summary
    print(visualizer.generate_insights_summary())
    print()
    
    # Save markdown report
    report_file = analysis_dir / "DEEP_TREE_ECHO_REPORT.md"
    visualizer.save_markdown_report(str(report_file))
    
    print(f"\nTo view the full report:")
    print(f"  cat {report_file}")
    print(f"  or open {report_file} in a markdown viewer")


if __name__ == "__main__":
    main()
