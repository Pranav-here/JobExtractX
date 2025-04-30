"""
Utility functions for JobExtractX evaluation.

This module provides helper functions for loading, saving,
and processing evaluation data and results.
"""

import os
import json
import glob
import sys
from datetime import datetime

# Add the project root to the path so we can import from model directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def ensure_results_dir(dir_path="evaluation_results"):
    """
    Ensure the evaluation results directory exists.
    
    Args:
        dir_path: Path to the results directory
        
    Returns:
        str: Path to the results directory
    """
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def generate_results_filename(prefix="evaluation", results_dir="evaluation_results"):
    """
    Generate a timestamped filename for results.
    
    Args:
        prefix: Prefix for the filename
        results_dir: Directory for results files
        
    Returns:
        str: Full path to the results file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    return os.path.join(results_dir, filename)

def save_results(results, filename=None, results_dir="evaluation_results"):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Results data to save
        filename: Filename to save to (if None, generate one)
        results_dir: Directory for results files
        
    Returns:
        str: Path to the saved file
    """
    ensure_results_dir(results_dir)
    
    if filename is None:
        filename = generate_results_filename(results_dir=results_dir)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    return filename

def load_results(filename):
    """
    Load results from a JSON file.
    
    Args:
        filename: Path to the results file
        
    Returns:
        dict: The loaded results data
    """
    with open(filename, 'r') as f:
        return json.load(f)

def find_latest_results(results_dir="evaluation_results", prefix="evaluation"):
    """
    Find the most recent evaluation results file.
    
    Args:
        results_dir: Directory containing results files
        prefix: File prefix to search for
        
    Returns:
        str: Path to the most recent results file, or None if not found
    """
    ensure_results_dir(results_dir)
    pattern = os.path.join(results_dir, f"{prefix}_*.json")
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        return None
    
    # Sort by modification time (newest first)
    latest_file = max(matching_files, key=os.path.getmtime)
    return latest_file

def find_model_checkpoint(base_dir="."):
    """
    Find available model checkpoints in the workspace.
    
    Args:
        base_dir: Base directory to search from
        
    Returns:
        list: List of found model checkpoint paths
    """
    # Common patterns for model checkpoints
    patterns = [
        "**/checkpoint-*",
        "**/best_model",
        "**/model_final"
    ]
    
    checkpoints = []
    for pattern in patterns:
        full_pattern = os.path.join(base_dir, pattern)
        matches = glob.glob(full_pattern, recursive=True)
        checkpoints.extend(matches)
    
    return checkpoints

def print_summary(metrics, field_metrics=None):
    """
    Print a summary of evaluation metrics.
    
    Args:
        metrics: Dictionary of overall metrics
        field_metrics: Dictionary of field-specific metrics
    """
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    print(f"JSON validity rate: {metrics.get('json_valid_rate', 0):.2f}%")
    print(f"Average field coverage: {metrics.get('avg_coverage', 0):.2f}%")
    print(f"Average field accuracy: {metrics.get('avg_accuracy', 0):.2f}%")
    
    if field_metrics:
        print("\nField-specific accuracy:")
        # Sort fields by accuracy (highest first)
        sorted_fields = sorted(field_metrics.items(), key=lambda x: x[1], reverse=True)
        for field, accuracy in sorted_fields:
            print(f"  {field}: {accuracy:.2f}%")
    
    print("="*50)