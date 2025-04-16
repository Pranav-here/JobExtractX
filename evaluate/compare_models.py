"""
Compare multiple JobExtractX models on evaluation metrics.

This script allows comparing the performance of multiple models
on the job description information extraction task.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from evaluate.evaluator import run_evaluation
from evaluate.utils import ensure_results_dir


def compare_models(models, model_prompts=None, num_examples=10, max_length=1024, 
                  results_dir="evaluation_results"):
    """
    Compare multiple models on the same evaluation set.
    
    Args:
        models: List of model names or paths to evaluate
        model_prompts: Dictionary mapping model names to whether they use prompts (True/False)
        num_examples: Number of examples to evaluate for each model
        max_length: Maximum output length for generation
        results_dir: Directory to store evaluation results
        
    Returns:
        dict: Dictionary containing comparison results
    """
    ensure_results_dir(results_dir)
    
    # Default all models to no prompt if not specified
    if model_prompts is None:
        model_prompts = {}
    
    # Results for all models
    all_results = {}
    
    # Evaluate each model
    for model in models:
        print("\n" + "="*70)
        print(f"Evaluating model: {model}")
        
        # Check if this model uses prompts
        use_prompt = model_prompts.get(model, False)
        print(f"Using prompt: {use_prompt}")
        print("="*70)
        
        result = run_evaluation(
            model_name_or_path=model,
            num_examples=num_examples,
            max_length=max_length,
            results_dir=results_dir,
            save_visualizations=False,  # We'll create our own comparison visualizations
            use_prompt=use_prompt
        )
        
        if result:
            all_results[model] = result
    
    # Create comparison visualizations
    model_names = list(all_results.keys())
    
    if model_names:
        # Extract metrics for comparison
        json_valid_rates = [all_results[model]["metrics"]["json_valid_rate"] for model in model_names]
        coverages = [all_results[model]["metrics"]["avg_coverage"] for model in model_names]
        accuracies = [all_results[model]["metrics"]["avg_accuracy"] for model in model_names]
        
        # Create comparison chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(model_names))
        width = 0.25
        
        ax.bar(x - width, json_valid_rates, width, label='JSON Validity Rate')
        ax.bar(x, coverages, width, label='Field Coverage')
        ax.bar(x + width, accuracies, width, label='Field Accuracy')
        
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        
        # Create labels with model name and prompt info
        labels = []
        for model in model_names:
            basename = os.path.basename(model)
            prompt_used = all_results[model]["metrics"].get("used_prompt", False)
            labels.append(f"{basename}\n({'with' if prompt_used else 'no'} prompt)")
            
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        plt.tight_layout()
        
        # Save the comparison chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_chart_path = os.path.join(results_dir, f"model_comparison_{timestamp}.png")
        plt.savefig(comparison_chart_path)
        
        print(f"\nComparison chart saved to: {comparison_chart_path}")
        
        # Save full comparison results
        comparison_results = {
            "models": model_names,
            "model_prompts": {model: all_results[model]["metrics"].get("used_prompt", False) for model in model_names},
            "metrics": {
                "json_valid_rate": dict(zip(model_names, json_valid_rates)),
                "field_coverage": dict(zip(model_names, coverages)),
                "field_accuracy": dict(zip(model_names, accuracies)),
            },
            "detailed_results": all_results
        }
        
        comparison_results_path = os.path.join(results_dir, f"model_comparison_{timestamp}.json")
        with open(comparison_results_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"Comparison results saved to: {comparison_results_path}")
        
        return {
            "chart": comparison_chart_path,
            "results": comparison_results_path
        }
    else:
        print("No valid model results to compare.")
        return None


def main():
    """
    Main entry point for comparing models from the command line.
    """
    parser = argparse.ArgumentParser(description="Compare multiple JobExtractX models")
    
    parser.add_argument("--models", nargs="+", required=True,
                        help="Space-separated list of model names or paths to evaluate")
    parser.add_argument("--prompts", nargs="+", 
                        help="Space-separated list of 'true' or 'false' values corresponding to whether each model uses prompts")
    parser.add_argument("--num-examples", type=int, default=10,
                        help="Number of examples to evaluate (default: 10)")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Maximum output length for generation (default: 1024)")
    parser.add_argument("--results-dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results (default: evaluation_results)")
    
    args = parser.parse_args()
    
    # Create dictionary mapping models to prompt settings
    model_prompts = {}
    if args.prompts:
        if len(args.prompts) != len(args.models):
            print("Error: Number of prompt settings must match number of models")
            return
        
        for model, prompt_setting in zip(args.models, args.prompts):
            model_prompts[model] = prompt_setting.lower() == "true"
    
    compare_models(
        models=args.models,
        model_prompts=model_prompts,
        num_examples=args.num_examples,
        max_length=args.max_length,
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()