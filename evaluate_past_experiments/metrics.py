"""
Metrics for evaluating JobExtractX model performance.

This module provides functions to calculate various metrics
for evaluating the performance of JSON extraction models.
"""

import json
import numpy as np

def calculate_field_coverage(expected_json, generated_json):
    """Calculate field coverage between expected and generated JSON.
    
    Args:
        expected_json: The expected JSON output (ground truth)
        generated_json: The model-generated JSON output
        
    Returns:
        float: Percentage of fields in expected JSON that are present in generated JSON
    """
    expected_fields = set(expected_json.keys())
    generated_fields = set(generated_json.keys())
    common_fields = expected_fields.intersection(generated_fields)
    
    if expected_fields:
        return len(common_fields) / len(expected_fields) * 100
    return 0

def calculate_field_accuracy(expected_json, generated_json):
    """Calculate field-by-field accuracy between expected and generated JSON.
    
    Args:
        expected_json: The expected JSON output (ground truth)
        generated_json: The model-generated JSON output
        
    Returns:
        tuple: (average_accuracy, field_accuracy_dict)
            - average_accuracy: Average accuracy across all fields
            - field_accuracy_dict: Dictionary mapping field names to their accuracy scores
    """
    expected_fields = set(expected_json.keys())
    generated_fields = set(generated_json.keys())
    common_fields = expected_fields.intersection(generated_fields)
    
    field_accuracy = {}
    for field in common_fields:
        expected_value = expected_json[field]
        generated_value = generated_json[field]
        
        if isinstance(expected_value, list) and isinstance(generated_value, list):
            # For list fields, calculate overlap
            expected_set = set(str(item).lower() for item in expected_value if item)
            generated_set = set(str(item).lower() for item in generated_value if item)
            
            if expected_set:
                # Calculate overlap percentage
                overlap = len(expected_set.intersection(generated_set)) / len(expected_set) * 100
            else:
                overlap = 100 if not generated_set else 0
            field_accuracy[field] = overlap
        elif isinstance(expected_value, dict) and isinstance(generated_value, dict):
            # For nested dictionaries, calculate sub-field coverage
            sub_coverage = calculate_field_coverage(expected_value, generated_value)
            field_accuracy[field] = sub_coverage
        else:
            # For scalar values, check exact match
            match = 100 if str(expected_value).lower() == str(generated_value).lower() else 0
            field_accuracy[field] = match
    
    # Calculate average accuracy across all fields
    if field_accuracy:
        avg_accuracy = sum(field_accuracy.values()) / len(field_accuracy)
    else:
        avg_accuracy = 0
    
    return avg_accuracy, field_accuracy

def calculate_json_validity_rate(results):
    """Calculate JSON validity rate from a list of evaluation results.
    
    Args:
        results: List of result dictionaries, each containing a 'json_valid' field
        
    Returns:
        float: Percentage of results with valid JSON
    """
    if not results:
        return 0
    
    valid_count = sum(1 for r in results if r.get('json_valid', False))
    return (valid_count / len(results)) * 100

def aggregate_field_metrics(results):
    """Aggregate field-specific metrics across multiple evaluation results.
    
    Args:
        results: List of result dictionaries with field-specific metrics
        
    Returns:
        dict: Dictionary mapping field names to their average accuracy scores
    """
    field_avg_accuracies = {}
    
    for result in results:
        if result.get("json_valid", False):
            field_accs = result.get("field_accuracies", {})
            for field, acc in field_accs.items():
                if field not in field_avg_accuracies:
                    field_avg_accuracies[field] = []
                field_avg_accuracies[field].append(acc)
    
    # Calculate average accuracy for each field
    return {field: np.mean(accs) for field, accs in field_avg_accuracies.items()}