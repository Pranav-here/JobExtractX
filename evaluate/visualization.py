"""
Visualization utilities for JobExtractX evaluation results.

This module provides functions to create visualizations
of evaluation metrics and results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_field_accuracies(field_accuracies, save_path=None):
    """
    Plot accuracies for each field.
    
    Args:
        field_accuracies: Dictionary mapping field names to accuracy percentages
        save_path: Optional path to save the figure
    """
    # Sort fields by accuracy
    sorted_fields = sorted(field_accuracies.items(), key=lambda x: x[1], reverse=True)
    fields = [item[0] for item in sorted_fields]
    accuracies = [item[1] for item in sorted_fields]
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(fields)))
    
    bars = plt.barh(fields, accuracies, color=colors)
    
    # Add accuracy values at the end of each bar
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{accuracies[i]:.1f}%', va='center')
    
    plt.title('Field Accuracy Percentages')
    plt.xlabel('Accuracy (%)')
    plt.xlim(0, 110)  # Add some space for the text
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Field accuracies chart saved to {save_path}")
    else:
        plt.show()

def plot_metrics_comparison(metrics, save_path=None):
    """
    Plot a comparison of key metrics.
    
    Args:
        metrics: Dictionary with keys 'json_valid_rate', 'avg_coverage', 'avg_accuracy'
        save_path: Optional path to save the figure
    """
    metric_names = ['JSON Validity Rate', 'Field Coverage', 'Field Accuracy']
    metric_values = [
        metrics.get('json_valid_rate', 0),
        metrics.get('avg_coverage', 0),
        metrics.get('avg_accuracy', 0)
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=['#2166AC', '#4393C3', '#92C5DE'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.title('Evaluation Metrics Comparison')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 110)  # Give some space for labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Metrics comparison chart saved to {save_path}")
    else:
        plt.show()

def visualize_evaluation_results(results_file, output_dir="evaluation_results"):
    """
    Create a suite of visualizations from an evaluation results file.
    
    Args:
        results_file: Path to the JSON results file
        output_dir: Directory to save visualization outputs
    
    Returns:
        dict: Paths to saved visualization files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract results
    coverage_values = [r.get('field_coverage', 0) for r in results if r.get('json_valid', False)]
    accuracy_values = [r.get('field_accuracy', 0) for r in results if r.get('json_valid', False)]
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate metrics
    metrics = {
        'json_valid_rate': sum(1 for r in results if r.get('json_valid', False)) / len(results) * 100,
        'avg_coverage': np.mean(coverage_values) if coverage_values else 0,
        'avg_accuracy': np.mean(accuracy_values) if accuracy_values else 0
    }
    
    # Aggregate field accuracies
    field_accuracies = {}
    for result in results:
        if result.get('json_valid', False):
            for field, acc in result.get('field_accuracies', {}).items():
                if field not in field_accuracies:
                    field_accuracies[field] = []
                field_accuracies[field].append(acc)
    
    field_avg_accuracies = {field: np.mean(accs) for field, accs in field_accuracies.items()}
    
    # Create visualizations
    metrics_chart_path = os.path.join(output_dir, f"metrics_comparison_{timestamp}.png")
    field_acc_chart_path = os.path.join(output_dir, f"field_accuracies_{timestamp}.png")
    
    plot_metrics_comparison(metrics, metrics_chart_path)
    plot_field_accuracies(field_avg_accuracies, field_acc_chart_path)
    
    return {
        'metrics_chart': metrics_chart_path,
        'field_accuracies_chart': field_acc_chart_path
    }