import json
import os
import glob
import re
from sklearn.metrics import accuracy_score
import numpy as np

def find_latest_results():
    """Find the most recent evaluation results file."""
    results_dir = "evaluation_results"
    result_files = glob.glob(os.path.join(results_dir, "evaluation_results_*.json"))
    if not result_files:
        print("No evaluation results found.")
        return None
    
    # Sort by modification time (newest first)
    latest_file = max(result_files, key=os.path.getmtime)
    return latest_file

def calculate_json_match_rate(results):
    """Calculate the percentage of outputs that are valid JSON."""
    valid_json_count = sum(1 for r in results if "Both outputs are valid JSON" in r["json_match"])
    return (valid_json_count / len(results)) * 100

def calculate_field_coverage(results):
    """Calculate how many expected fields are present in the generated output."""
    field_coverage_rates = []
    
    for result in results:
        try:
            # Try to parse the expected and generated outputs as JSON
            expected_json = json.loads(result["expected_output"])
            
            # Handle case where generated output might not be valid JSON
            try:
                generated_text = result["generated_output"]
                # Try to complete the JSON if it's truncated
                if generated_text.startswith('"') and not generated_text.endswith('"'):
                    generated_text += '"'
                if not (generated_text.startswith('{') or generated_text.endswith('}')):
                    generated_text = '{' + generated_text + '}'
                
                generated_json = json.loads(generated_text)
                
                # Count fields that are present in both
                expected_fields = set(expected_json.keys())
                generated_fields = set(generated_json.keys())
                common_fields = expected_fields.intersection(generated_fields)
                
                # Calculate coverage rate
                coverage_rate = len(common_fields) / len(expected_fields) * 100
                field_coverage_rates.append(coverage_rate)
            except json.JSONDecodeError:
                # If we can't parse the generated output as JSON, coverage is 0%
                field_coverage_rates.append(0)
        except json.JSONDecodeError:
            # If we can't parse the expected output as JSON, skip this example
            continue
    
    if field_coverage_rates:
        return {
            "avg_field_coverage": np.mean(field_coverage_rates),
            "min_field_coverage": np.min(field_coverage_rates),
            "max_field_coverage": np.max(field_coverage_rates)
        }
    else:
        return {
            "avg_field_coverage": 0,
            "min_field_coverage": 0,
            "max_field_coverage": 0
        }

def analyze_results(results_file):
    """Analyze the evaluation results and print metrics."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    num_examples = len(results)
    print(f"Analyzing {num_examples} examples from {results_file}")
    
    # Calculate metrics
    json_match_rate = calculate_json_match_rate(results)
    field_coverage = calculate_field_coverage(results)
    
    # Print summary
    print("\n===== MODEL EVALUATION SUMMARY =====")
    print(f"Total examples analyzed: {num_examples}")
    print(f"JSON match rate: {json_match_rate:.2f}%")
    print(f"Average field coverage: {field_coverage['avg_field_coverage']:.2f}%")
    print(f"Min field coverage: {field_coverage['min_field_coverage']:.2f}%")
    print(f"Max field coverage: {field_coverage['max_field_coverage']:.2f}%")
    
    # Analyze common issues
    print("\n===== COMMON ISSUES =====")
    truncation_count = sum(1 for r in results if len(r["generated_output"]) < 50)
    print(f"Severely truncated outputs: {truncation_count} ({truncation_count/num_examples*100:.2f}%)")
    
    # Check for any examples with good performance
    good_examples = [i for i, r in enumerate(results) if "Both outputs are valid JSON" in r["json_match"]]
    if good_examples:
        print(f"\nExamples with valid JSON output: {len(good_examples)} ({len(good_examples)/num_examples*100:.2f}%)")
        print(f"Example IDs: {good_examples}")
    
    print("\n===== RECOMMENDATIONS =====")
    if json_match_rate < 50:
        print("- The model struggles to produce complete, valid JSON. Consider fine-tuning with a focus on JSON completion.")
    if field_coverage['avg_field_coverage'] < 50:
        print("- Field coverage is low. The model may be underfitting or not understanding the expected output format.")
    if truncation_count > num_examples * 0.3:
        print("- Many outputs are truncated. Consider increasing max token length for generation.")

if __name__ == "__main__":
    latest_results = find_latest_results()
    if latest_results:
        analyze_results(latest_results)
    else:
        print("Please run model/evaluate.py first to generate results.") 