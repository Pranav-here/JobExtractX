import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Load model evaluation data
t5_large_file = "evaluation_results/flan_t5_large_eval/flan_t5_large_eval_20250418_092005.json"
t5_xl_lora_batch_2_file = "evaluation_results/flan_t5_xl_lora_batch_2_eval/flan_t5_xl_lora_eval_20250418_131856.json"
t5_xl_lora_batch_4_file = "evaluation_results/flan_t5_xl_lora_batch_4_eval/flan_t5_xl_lora_eval_20250418_114438.json"
mistral_lora_file = "evaluation_results/mistral_lora_eval/mistral_lora_eval_20250418_132938.json"

# Ensure output directory exists
os.makedirs("evaluation_results/comparison", exist_ok=True)

# Load all model data
with open(t5_large_file, 'r') as f:
    t5_large_data = json.load(f)

with open(t5_xl_lora_batch_2_file, 'r') as f:
    t5_xl_lora_batch_2_data = json.load(f)

with open(t5_xl_lora_batch_4_file, 'r') as f:
    t5_xl_lora_batch_4_data = json.load(f)

with open(mistral_lora_file, 'r') as f:
    mistral_lora_data = json.load(f)

# Model names
models = ["T5 Large", "T5-XL LoRA Batch 2", "T5-XL LoRA Batch 4", "Mistral LoRA"]
model_data = [t5_large_data, t5_xl_lora_batch_2_data, t5_xl_lora_batch_4_data, mistral_lora_data]

# Create timestamp for filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Overall Metrics Comparison
metrics = ['parse_rate', 'mean_f1_score']
all_metrics = []

for data in model_data:
    model_metrics = [
        data["metrics"]["overall"]["parse_rate"],
        data["metrics"]["overall"]["mean_f1_score"]
    ]
    all_metrics.append(model_metrics)

x = np.arange(len(metrics))
width = 0.2  # Narrower bars for 4 models

fig, ax = plt.subplots(figsize=(12, 8))
for i, model_metric in enumerate(all_metrics):
    offset = width * (i - 1.5)
    rects = ax.bar(x + offset, model_metric, width, label=models[i])
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Score')
ax.set_title('Overall Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig(f'evaluation_results/comparison/overall_metrics_comparison_{timestamp}.png')

# 2. Field Metrics Comparison - Accuracy Rate
# Find common fields across all models
all_field_keys = set()
for data in model_data:
    if "field_metrics" in data["metrics"]:
        all_field_keys.update(data["metrics"]["field_metrics"].keys())

fields = sorted(list(all_field_keys))

# Get accuracy rates for each model and field
all_field_metrics = []
for data in model_data:
    model_field_metrics = []
    for field in fields:
        if "field_metrics" in data["metrics"] and field in data["metrics"]["field_metrics"]:
            model_field_metrics.append(data["metrics"]["field_metrics"][field]["accuracy_rate"])
        else:
            model_field_metrics.append(0)
    all_field_metrics.append(model_field_metrics)

fig, ax = plt.subplots(figsize=(16, 10))
x = np.arange(len(fields))
width = 0.2

for i, model_metric in enumerate(all_field_metrics):
    offset = width * (i - 1.5)
    rects = ax.bar(x + offset, model_metric, width, label=models[i])

ax.set_ylabel('Accuracy Rate')
ax.set_title('Field Accuracy Rate Comparison')
ax.set_xticks(x)
ax.set_xticklabels(fields, rotation=45, ha='right')
ax.legend()

plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig(f'evaluation_results/comparison/field_accuracy_comparison_{timestamp}.png')

# 3. List Field Metrics Comparison - F1 Scores
# Find common list fields across all models
all_list_field_keys = set()
for data in model_data:
    if "list_field_metrics" in data["metrics"]:
        all_list_field_keys.update(data["metrics"]["list_field_metrics"].keys())

list_fields = sorted(list(all_list_field_keys))

# Get F1 scores for each model and list field
all_list_field_metrics = []
for data in model_data:
    model_list_field_metrics = []
    for field in list_fields:
        if "list_field_metrics" in data["metrics"] and field in data["metrics"]["list_field_metrics"]:
            model_list_field_metrics.append(data["metrics"]["list_field_metrics"][field]["f1_score"])
        else:
            model_list_field_metrics.append(0)
    all_list_field_metrics.append(model_list_field_metrics)

fig, ax = plt.subplots(figsize=(18, 12))
x = np.arange(len(list_fields))
width = 0.2

for i, model_metric in enumerate(all_list_field_metrics):
    offset = width * (i - 1.5)
    rects = ax.bar(x + offset, model_metric, width, label=models[i])

ax.set_ylabel('F1 Score')
ax.set_title('List Field F1 Score Comparison')
ax.set_xticks(x)
ax.set_xticklabels(list_fields, rotation=45, ha='right')
ax.legend()

plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig(f'evaluation_results/comparison/list_field_f1_comparison_{timestamp}.png')

# 4. Sample Size Comparison
samples = [data["metrics"]["overall"]["total_samples"] for data in model_data]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, samples)

ax.set_ylabel('Number of Samples')
ax.set_title('Total Samples Comparison')
plt.xticks(rotation=15)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f'evaluation_results/comparison/sample_size_comparison_{timestamp}.png')

print(f"Visualizations created with timestamp: {timestamp} in evaluation_results/comparison/ directory") 