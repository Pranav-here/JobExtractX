import json
import os
import re
import glob
import numpy as np
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from data_preprocessing import prepare_data
from post_processor import fix_json

# Create results directory if it doesn't exist
results_dir = "evaluation_results"
os.makedirs(results_dir, exist_ok=True)

def calculate_field_coverage(expected_json, generated_json):
    """Calculate field coverage between expected and generated JSON."""
    expected_fields = set(expected_json.keys())
    generated_fields = set(generated_json.keys())
    common_fields = expected_fields.intersection(generated_fields)
    
    if expected_fields:
        return len(common_fields) / len(expected_fields) * 100
    return 0

def calculate_field_accuracy(expected_json, generated_json):
    """Calculate field-by-field accuracy between expected and generated JSON."""
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

def run_evaluation(num_examples=10, max_length=1024):
    """Run a comprehensive evaluation of the model with post-processing."""
    # Initialize the pipeline
    print("Initializing model pipeline...")
    pipe = pipeline("text2text-generation", model="alexlanxy/flan_t5_large_linkedin_no_prompt", max_length=max_length)
    
    # Load data
    print("Loading data...")
    data_pairs = prepare_data()
    print(f"Loaded {len(data_pairs)} data pairs for evaluation")
    
    # Create a timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"comprehensive_eval_{timestamp}.json")
    
    # Evaluation results
    results = []
    
    # Process subset for examples
    sample_count = min(num_examples, len(data_pairs))
    print(f"\nGenerating and evaluating outputs for {sample_count} examples...")
    
    # Track metrics
    json_valid_count = 0
    coverage_values = []
    accuracy_values = []
    
    for i, pair in enumerate(data_pairs[:sample_count]):
        print(f"\nExample {i+1}/{sample_count}")
        input_text = pair[0]
        expected_output = pair[1]
        
        # Generate output
        print("Generating output...")
        model_output = pipe(input_text)
        generated_text = model_output[0]["generated_text"]
        
        # Post-process to fix JSON
        print("Post-processing output...")
        processed_output = fix_json(generated_text)
        
        # Calculate metrics
        try:
            expected_json = json.loads(expected_output)
            try:
                generated_json = json.loads(processed_output)
                json_valid = True
                json_valid_count += 1
                
                # Calculate coverage and accuracy
                coverage = calculate_field_coverage(expected_json, generated_json)
                accuracy, field_accuracies = calculate_field_accuracy(expected_json, generated_json)
                
                coverage_values.append(coverage)
                accuracy_values.append(accuracy)
                
                print(f"Field coverage: {coverage:.2f}%")
                print(f"Field accuracy: {accuracy:.2f}%")
                
                # Show accuracies for top fields
                top_fields = sorted(field_accuracies.items(), key=lambda x: x[1], reverse=True)[:3]
                print("Top field accuracies:")
                for field, acc in top_fields:
                    print(f"  {field}: {acc:.2f}%")
                
            except json.JSONDecodeError:
                json_valid = False
                coverage = 0
                accuracy = 0
                field_accuracies = {}
                print("Failed to parse processed output as JSON")
                
        except json.JSONDecodeError:
            json_valid = False
            coverage = 0
            accuracy = 0
            field_accuracies = {}
            print("Failed to parse expected output as JSON")
        
        # Store result
        results.append({
            "example_id": i,
            "input": input_text,
            "expected_output": expected_output,
            "generated_output": generated_text,
            "processed_output": processed_output,
            "json_valid": json_valid,
            "field_coverage": coverage,
            "field_accuracy": accuracy,
            "field_accuracies": field_accuracies
        })
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Compute overall metrics
    avg_coverage = np.mean(coverage_values) if coverage_values else 0
    avg_accuracy = np.mean(accuracy_values) if accuracy_values else 0
    json_valid_rate = (json_valid_count / sample_count) * 100
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total examples evaluated: {sample_count}")
    print(f"Valid JSON rate after post-processing: {json_valid_rate:.2f}%")
    print(f"Average field coverage: {avg_coverage:.2f}%")
    print(f"Average field accuracy: {avg_accuracy:.2f}%")
    
    # Field-specific analysis
    print("\nAccuracy by field:")
    field_avg_accuracies = {}
    
    for result in results:
        if result.get("json_valid", False):
            field_accs = result.get("field_accuracies", {})
            for field, acc in field_accs.items():
                if field not in field_avg_accuracies:
                    field_avg_accuracies[field] = []
                field_avg_accuracies[field].append(acc)
    
    # Calculate average accuracy for each field
    for field, accs in field_avg_accuracies.items():
        avg = np.mean(accs)
        print(f"  {field}: {avg:.2f}%")
    
    print(f"\nResults saved to {results_file}")
    
    return {
        "results_file": results_file,
        "json_valid_rate": json_valid_rate,
        "avg_coverage": avg_coverage,
        "avg_accuracy": avg_accuracy,
        "field_avg_accuracies": field_avg_accuracies
    }

if __name__ == "__main__":
    # Run with different example counts
    for num_examples in [5, 10]:
        print("\n" + "="*70)
        print(f"Running comprehensive evaluation with {num_examples} examples")
        print("="*70)
        
        run_evaluation(num_examples=num_examples, max_length=1024) 