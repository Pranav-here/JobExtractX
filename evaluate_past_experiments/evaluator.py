"""
Main evaluation runner for JobExtractX models.

This module provides the primary evaluation functionality to assess
model performance on job description information extraction tasks.
"""

import json
import os
import sys
import argparse
import numpy as np
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.data_preprocessing import prepare_data
from model.post_processor import fix_json
from evaluate.metrics import (calculate_field_coverage, calculate_field_accuracy, 
                             calculate_json_validity_rate, aggregate_field_metrics)
from evaluate.utils import (ensure_results_dir, save_results, load_results,
                          find_latest_results, generate_results_filename, print_summary)
from evaluate.visualization import visualize_evaluation_results


def run_evaluation(model_name_or_path, num_examples=10, max_length=1024, 
                  results_dir="evaluation_results", save_visualizations=True, 
                  use_prompt=False):
    """
    Run a comprehensive evaluation of the model with post-processing.
    
    Args:
        model_name_or_path: Model name on HF hub or path to local model
        num_examples: Number of examples to evaluate
        max_length: Maximum output length for generation
        results_dir: Directory to store evaluation results
        save_visualizations: Whether to generate and save visualizations
        use_prompt: Whether to use prompts with schema in the input
        
    Returns:
        dict: Dictionary containing evaluation summary and paths to results
    """
    # Ensure results directory exists
    ensure_results_dir(results_dir)
    
    # Initialize the pipeline
    print("Initializing model pipeline...")
    try:
        pipe = pipeline("text2text-generation", model=model_name_or_path, max_length=max_length)
        print(f"Successfully loaded model: {model_name_or_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    # Load data with appropriate prompt setting based on the model
    print(f"Loading data with use_prompt={use_prompt}...")
    data_pairs = prepare_data(prompt=use_prompt)
    print(f"Loaded {len(data_pairs)} data pairs")
    
    # Use only the last 10% of data for evaluation (ensuring it's different from training data)
    eval_start_idx = int(len(data_pairs) * 0.9)
    evaluation_data = data_pairs[eval_start_idx:]
    print(f"Using {len(evaluation_data)} examples from the last 10% of data for evaluation")
    
    # Create a timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"comprehensive_eval_{timestamp}.json")
    
    # Evaluation results
    results = []
    
    # Process subset for examples
    sample_count = min(num_examples, len(evaluation_data))
    print(f"\nGenerating and evaluating outputs for {sample_count} examples...")
    
    # Track metrics
    json_valid_count = 0
    coverage_values = []
    accuracy_values = []
    
    for i, pair in enumerate(evaluation_data[:sample_count]):
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
    save_path = save_results(results, results_file)
    print(f"\nResults saved to {save_path}")
    
    # Compute overall metrics
    avg_coverage = np.mean(coverage_values) if coverage_values else 0
    avg_accuracy = np.mean(accuracy_values) if accuracy_values else 0
    json_valid_rate = (json_valid_count / sample_count) * 100
    
    # Aggregate field metrics
    field_avg_accuracies = aggregate_field_metrics(results)
    
    # Create metric summary
    metrics = {
        "json_valid_rate": json_valid_rate,
        "avg_coverage": avg_coverage,
        "avg_accuracy": avg_accuracy,
        "used_prompt": use_prompt
    }
    
    # Print summary
    print_summary(metrics, field_avg_accuracies)
    
    # Generate visualizations
    viz_paths = {}
    if save_visualizations:
        print("\nGenerating visualizations...")
        viz_paths = visualize_evaluation_results(save_path, results_dir)
    
    return {
        "results_file": save_path,
        "metrics": metrics,
        "field_metrics": field_avg_accuracies,
        "visualizations": viz_paths
    }


def main():
    """
    Main entry point for running evaluations from the command line.
    """
    parser = argparse.ArgumentParser(description="Evaluate JobExtractX models")
    
    parser.add_argument("--model", type=str, default="alexlanxy/flan_t5_large_linkedin_no_prompt",
                        help="Model name or path to evaluate (default: alexlanxy/flan_t5_large_linkedin_no_prompt)")
    parser.add_argument("--num-examples", type=int, default=10,
                        help="Number of examples to evaluate (default: 10)")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Maximum output length for generation (default: 1024)")
    parser.add_argument("--results-dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results (default: evaluation_results)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization generation")
    parser.add_argument("--use-prompt", action="store_true",
                        help="Use schema prompts in the input (for prompt-based models)")
    
    args = parser.parse_args()
    
    run_evaluation(
        model_name_or_path=args.model,
        num_examples=args.num_examples,
        max_length=args.max_length,
        results_dir=args.results_dir,
        save_visualizations=not args.no_viz,
        use_prompt=args.use_prompt
    )


if __name__ == "__main__":
    main()