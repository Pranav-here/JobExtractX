import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

# Configure visualizations
plt.style.use('ggplot')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = [12, 8]

def load_evaluation_results(directory="evaluation_results/flan_t5_xl_lora_batch_4_eval"):
    """Load all evaluation result files in the directory."""
    result_files = glob.glob(f"{directory}/flan_t5_xl_lora_eval_*.json")
    
    if not result_files:
        print(f"No result files found in {directory}")
        return None
    
    # Sort by timestamp (newest first)
    result_files.sort(reverse=True)
    print(f"Found {len(result_files)} result files. Using most recent: {os.path.basename(result_files[0])}")
    
    # Load the most recent file
    with open(result_files[0], 'r') as f:
        data = json.load(f)
    
    return data

def visualize_metrics(data):
    """Create visualizations for the evaluation metrics."""
    if not data or 'metrics' not in data:
        print("No metrics found in data")
        return
    
    metrics = data['metrics']
    
    # Create output directory for visualizations
    output_dir = "evaluation_results/flan_t5_xl_lora_batch_4_eval/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Overall Metrics
    overall = metrics.get('overall', {})
    
    # Create bar chart for overall metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = {
        'Parse Rate': overall.get('parse_rate', 0),
        'Mean F1 Score': overall.get('mean_f1_score', 0)
    }
    
    plt.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color=['#3498db', '#2ecc71'])
    plt.ylim(0, 1.0)
    plt.title('Overall Extraction Performance', fontsize=16)
    plt.ylabel('Score (0-1)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(metrics_to_plot.values()):
        plt.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_metrics_{timestamp}.png")
    print(f"Saved overall metrics visualization to {output_dir}/overall_metrics_{timestamp}.png")
    
    # 2. Field-level metrics visualization
    field_metrics = metrics.get('field_metrics', {})
    
    if field_metrics:
        # Prepare data for field-level metrics
        fields = []
        presence_rates = []
        accuracy_rates = []
        
        for field, field_data in field_metrics.items():
            fields.append(field)
            presence_rates.append(field_data.get('presence_rate', 0))
            accuracy_rates.append(field_data.get('accuracy_rate', 0))
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Field': fields,
            'Presence Rate': presence_rates,
            'Accuracy Rate': accuracy_rates
        })
        
        # Sort by accuracy rate
        df = df.sort_values('Accuracy Rate', ascending=True)
        
        # Plot field metrics
        plt.figure(figsize=(12, max(6, len(fields) * 0.5)))
        
        # Plot horizontal bars
        ax = plt.subplot(1, 1, 1)
        barwidth = 0.35
        y_pos = np.arange(len(df['Field']))
        
        ax.barh(y_pos - barwidth/2, df['Presence Rate'], barwidth, color='#3498db', label='Presence Rate')
        ax.barh(y_pos + barwidth/2, df['Accuracy Rate'], barwidth, color='#2ecc71', label='Accuracy Rate')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['Field'])
        ax.set_xlim(0, 1.0)
        ax.set_title('Field-level Metrics', fontsize=16)
        ax.set_xlabel('Score (0-1)', fontsize=14)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.legend()
        
        # Add values to the bars
        for i, v in enumerate(df['Presence Rate']):
            ax.text(v + 0.05, i - barwidth/2, f'{v:.2f}', va='center', fontsize=10)
        
        for i, v in enumerate(df['Accuracy Rate']):
            ax.text(v + 0.05, i + barwidth/2, f'{v:.2f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/field_metrics_{timestamp}.png")
        print(f"Saved field metrics visualization to {output_dir}/field_metrics_{timestamp}.png")
    
    # 3. List field metrics (Precision, Recall, F1)
    list_metrics = metrics.get('list_field_metrics', {})
    
    if list_metrics:
        # Prepare data
        fields = []
        precision_values = []
        recall_values = []
        f1_values = []
        
        for field, field_data in list_metrics.items():
            fields.append(field)
            precision_values.append(field_data.get('precision', 0))
            recall_values.append(field_data.get('recall', 0))
            f1_values.append(field_data.get('f1_score', 0))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Field': fields,
            'Precision': precision_values,
            'Recall': recall_values,
            'F1 Score': f1_values
        })
        
        # Sort by F1 score
        df = df.sort_values('F1 Score', ascending=True)
        
        # Plot
        plt.figure(figsize=(14, max(6, len(fields) * 0.6)))
        
        # Plot horizontal bars grouped by metric
        ax = plt.subplot(1, 1, 1)
        barwidth = 0.25
        y_pos = np.arange(len(df['Field']))
        
        ax.barh(y_pos - barwidth, df['Precision'], barwidth, color='#3498db', label='Precision')
        ax.barh(y_pos, df['Recall'], barwidth, color='#2ecc71', label='Recall')
        ax.barh(y_pos + barwidth, df['F1 Score'], barwidth, color='#e74c3c', label='F1 Score')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['Field'])
        ax.set_xlim(0, 1.0)
        ax.set_title('List Field Extraction Performance', fontsize=16)
        ax.set_xlabel('Score (0-1)', fontsize=14)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/list_field_metrics_{timestamp}.png")
        print(f"Saved list field metrics visualization to {output_dir}/list_field_metrics_{timestamp}.png")
    
    # 4. Create combined summary visualization
    plt.figure(figsize=(15, 10))
    
    # Create a combined visualization showing key metrics across all categories
    summary_data = {}
    
    # Add overall metrics
    summary_data['Parse Rate'] = overall.get('parse_rate', 0)
    summary_data['Mean F1'] = overall.get('mean_f1_score', 0)
    
    # Add top field metrics (accuracy only)
    for field, metrics in field_metrics.items():
        summary_data[f"{field} (acc)"] = metrics.get('accuracy_rate', 0)
    
    # Add F1 scores for list fields
    for field, metrics in list_metrics.items():
        if not field.startswith('skills_'):  # Skip nested skill fields to reduce clutter
            summary_data[f"{field} (F1)"] = metrics.get('f1_score', 0)
    
    # Sort by value
    summary_data = {k: v for k, v in sorted(summary_data.items(), key=lambda item: item[1], reverse=True)}
    
    # Plot
    plt.figure(figsize=(14, 8))
    plt.bar(summary_data.keys(), summary_data.values(), color=plt.cm.viridis(np.linspace(0, 1, len(summary_data))))
    plt.ylim(0, 1.0)
    plt.title('Extraction Performance Summary', fontsize=16)
    plt.ylabel('Score (0-1)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    for i, (k, v) in enumerate(summary_data.items()):
        plt.text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_summary_{timestamp}.png")
    print(f"Saved performance summary visualization to {output_dir}/performance_summary_{timestamp}.png")
    
    # Display total samples count
    print(f"\nTotal evaluated samples: {overall.get('total_samples', 0)}")
    print(f"Successful parses: {overall.get('successful_parses', 0)}")

def main():
    print("Loading evaluation results...")
    data = load_evaluation_results()
    
    if data:
        print("Creating visualizations...")
        visualize_metrics(data)
        print("\nVisualization complete! Check the 'visualizations' directory for results.")
    else:
        print("No data to visualize.")

if __name__ == "__main__":
    main()
