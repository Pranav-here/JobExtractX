import sys
import os
import json
import torch
import random
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re
import numpy as np
from collections import Counter

# Create results directory if it doesn't exist
results_dir = "evaluation_results"
os.makedirs(results_dir, exist_ok=True)

# Create specific subdirectory for mistral lora evaluations
mistral_lora_results_dir = os.path.join(results_dir, "mistral_lora_eval")
os.makedirs(mistral_lora_results_dir, exist_ok=True)

# Set your Hugging Face token as an environment variable
# You need to accept the model's terms of use on the Hugging Face website first
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"  # Your token
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load your fine-tuned model from Hugging Face
model_name = "alexlanxy/mistral_7b_lora_batch_1_single_server"
print(f"Loading model {model_name}...")

# Initialize tokenizer with proper padding settings - same as in training
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print(f"Using pad_token_id: {tokenizer.pad_token_id}")

# Setup quantization config (load in 8-bit to save memory) - same as in training
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# Load model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"  # Automatically places on correct GPU
)

# Maximum sequence length for generation
max_length = 1536  # Same as in training script

# Load data from JSON file
print("Loading data from prepared_data.json...")
with open("prepared_data.json", "r") as f:
    json_data = json.load(f)

# Format the data into input-output pairs
data_pairs = []
for item in json_data:
    # For the prompt format
    input_text = f"Label the following job posting in pure JSON format based on this example schema. If no information for a field, leave the field blank.\n\nExample schema:\n{{\n    \"experience_level\": \"\",\n    \"employment_status\": [],\n    \"work_location\": \"\",\n    \"salary\": {{\"min\": \"\", \"max\": \"\", \"period\": \"\", \"currency\": \"\"}},\n    \"benefits\": [],\n    \"job_functions\": [],\n    \"required_skills\": {{\n        \"programming_languages\": [],\n        \"tools\": [],\n        \"frameworks\": [],\n        \"databases\": [],\n        \"other\": []\n    }},\n    \"required_certifications\": [],\n    \"required_minimum_degree\": \"\",\n    \"required_experience\": \"\",\n    \"industries\": [],\n    \"additional_keywords\": []\n}}\n\nJob posting:\n{item['source']}"
    data_pairs.append((input_text, item['target']))

print(f"Loaded {len(data_pairs)} data pairs")

# Print the first data pair to understand structure
print("\nSample input data structure:")
print(f"Input (first 200 chars): {data_pairs[0][0][:200]}...")
print(f"Expected output (first 200 chars): {data_pairs[0][1][:200]}...")

# Use only the last 10% of the data for evaluation (since first 90% was used for training)
total_samples = len(data_pairs)
test_start_idx = int(total_samples * 0.9)  # Start at 90% mark
data_pairs = data_pairs[test_start_idx:]
print(f"Using last 10% of data ({len(data_pairs)} items) for evaluation")

# Select random samples for evaluation from the test set
# Use a fixed seed for reproducibility
random.seed(42)
num_eval_samples = min(2, len(data_pairs))  # Use up to 10 samples
eval_samples = random.sample(data_pairs, num_eval_samples)
print(f"Selected {len(eval_samples)} samples for evaluation")

# Let's enhance the comparison function for better metrics
def calculate_metrics(results):
    """Calculate comprehensive metrics for the evaluation"""
    if not results:
        return {}
    
    metrics = {
        "overall": {
            "total_samples": len(results),
            "successful_parses": sum(1 for r in results if r["parsed_successfully"]),
            "parse_rate": 0.0
        },
        "field_metrics": {},
        "list_field_metrics": {},
        "nested_field_metrics": {}
    }
    
    # Calculate parse rate
    metrics["overall"]["parse_rate"] = metrics["overall"]["successful_parses"] / metrics["overall"]["total_samples"]
    
    # Skip metrics if no successful parses
    if metrics["overall"]["successful_parses"] == 0:
        return metrics
    
    # Track metrics for each field
    field_presence = {}  # How often field was populated vs empty
    field_accuracy = {}  # How often field matched exactly
    field_counts = {}    # How many samples had this field
    
    # For list fields 
    list_precision = {}  # How many generated items were correct
    list_recall = {}     # How many expected items were found
    list_f1 = {}         # Harmonic mean of precision and recall
    
    # Process each result
    for result in results:
        if not result["parsed_successfully"]:
            continue
            
        try:
            generated = json.loads(result["generated_json"])
            expected = json.loads(result["expected_json"])
            
            # Process simple string fields
            string_fields = ["experience_level", "work_location", "required_minimum_degree", "required_experience"]
            for field in string_fields:
                if field not in field_counts:
                    field_counts[field] = 0
                    field_presence[field] = 0
                    field_accuracy[field] = 0
                
                field_counts[field] += 1
                
                # Track if field has content
                if field in generated and generated[field] and generated[field] != "":
                    field_presence[field] += 1
                
                # Track exact matches
                if (field in generated and field in expected and 
                    str(generated[field]).lower() == str(expected[field]).lower()):
                    field_accuracy[field] += 1
            
            # Process list fields
            list_fields = ["employment_status", "benefits", "job_functions", 
                          "required_certifications", "industries", "additional_keywords"]
            
            for field in list_fields:
                if field not in list_precision:
                    list_precision[field] = []
                    list_recall[field] = []
                    list_f1[field] = []
                
                # Convert to sets for easier comparison, case-insensitive
                gen_set = set(item.lower() for item in generated.get(field, []) if item)
                exp_set = set(item.lower() for item in expected.get(field, []) if item)
                
                # Calculate metrics
                if gen_set or exp_set:  # Only if at least one set has items
                    true_positives = len(gen_set.intersection(exp_set))
                    
                    precision = true_positives / len(gen_set) if gen_set else 0
                    recall = true_positives / len(exp_set) if exp_set else 0
                    
                    # F1 score
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    list_precision[field].append(precision)
                    list_recall[field].append(recall)
                    list_f1[field].append(f1)
            
            # Special handling for nested structure: required_skills
            if "required_skills" in generated and "required_skills" in expected:
                gen_skills = generated["required_skills"]
                exp_skills = expected["required_skills"]
                
                skill_categories = ["programming_languages", "tools", "frameworks", "databases", "other"]
                
                for category in skill_categories:
                    field_key = f"skills_{category}"
                    if field_key not in list_precision:
                        list_precision[field_key] = []
                        list_recall[field_key] = []
                        list_f1[field_key] = []
                    
                    # Convert to sets for comparison
                    gen_set = set(item.lower() for item in gen_skills.get(category, []) if item)
                    exp_set = set(item.lower() for item in exp_skills.get(category, []) if item)
                    
                    # Calculate metrics
                    if gen_set or exp_set:
                        true_positives = len(gen_set.intersection(exp_set))
                        
                        precision = true_positives / len(gen_set) if gen_set else 0
                        recall = true_positives / len(exp_set) if exp_set else 0
                        
                        # F1 score
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        
                        list_precision[field_key].append(precision)
                        list_recall[field_key].append(recall)
                        list_f1[field_key].append(f1)
                
                # Special handling for salary field
                if "salary" in generated and "salary" in expected:
                    # Count number of salary components that match
                    salary_fields = ["min", "max", "period", "currency"]
                    for sal_field in salary_fields:
                        field_key = f"salary_{sal_field}"
                        if field_key not in field_counts:
                            field_counts[field_key] = 0
                            field_presence[field_key] = 0
                            field_accuracy[field_key] = 0
                        
                        field_counts[field_key] += 1
                        
                        gen_val = generated["salary"].get(sal_field, "")
                        exp_val = expected["salary"].get(sal_field, "")
                        
                        if gen_val and gen_val != "":
                            field_presence[field_key] += 1
                        
                        if str(gen_val).lower() == str(exp_val).lower():
                            field_accuracy[field_key] += 1
                
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            continue
    
    # Calculate averages for string fields
    for field in field_counts:
        if field_counts[field] > 0:
            presence_rate = field_presence[field] / field_counts[field]
            accuracy_rate = field_accuracy[field] / field_counts[field]
            
            metrics["field_metrics"][field] = {
                "presence_rate": presence_rate,
                "accuracy_rate": accuracy_rate,
                "count": field_counts[field]
            }
    
    # Calculate averages for list fields
    for field in list_precision:
        if list_precision[field]:
            avg_precision = np.mean(list_precision[field])
            avg_recall = np.mean(list_recall[field])
            avg_f1 = np.mean(list_f1[field])
            
            metrics["list_field_metrics"][field] = {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1_score": avg_f1,
                "count": len(list_precision[field])
            }
    
    # Calculate overall extraction quality (average F1 across all list fields)
    all_f1_scores = []
    for field_metrics in metrics["list_field_metrics"].values():
        all_f1_scores.append(field_metrics["f1_score"])
    
    if all_f1_scores:
        metrics["overall"]["mean_f1_score"] = np.mean(all_f1_scores)
    else:
        metrics["overall"]["mean_f1_score"] = 0.0
    
    return metrics

# Results container
results = []

# Evaluate each sample
print("\nStarting evaluation...")
for i, (input_text, expected_output) in enumerate(eval_samples):
    print(f"\nProcessing example {i+1}/10:")
    print(f"Input text length: {len(input_text)} characters")
    
    # Format messages following Mistral's chat format - exactly as in training
    messages = [
        {"role": "user", "content": input_text},
    ]
    
    # Print the formatted message
    print("\nFormatted input message:")
    print(f"Role: {messages[0]['role']}")
    print(f"Content (first 200 chars): {messages[0]['content'][:200]}...")
    
    # Use tokenizer to encode and handle message formatting
    chat_input = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    # Print tokenizer template
    print(f"\nTokenizer chat template: {tokenizer.chat_template}")
    
    # Print encoded chat input (first few tokens)
    print(f"Encoded chat input shape: {chat_input.shape}")
    print(f"First few tokens: {chat_input[0, :10]}")
    
    # Move to the same device as the model
    chat_input = chat_input.to(model.device)
    
    # Generate text
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=chat_input,
            max_new_tokens=384,
            temperature=0.2,  # Lower temperature for more deterministic outputs
            top_p=0.9,
            do_sample=True
        )
    
    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Print full generated text for debugging
    print("\nFull generated text:")
    print(generated_text)
    
    # Extract just the model's response (not the input prompt)
    assistant_start = generated_text.find("<|assistant|>")
    if assistant_start != -1:
        generated_json = generated_text[assistant_start + len("<|assistant|>"):].strip()
        print(f"\nFound assistant marker at position {assistant_start}")
    else:
        generated_json = generated_text
        print("\nNo assistant marker found in output")
    
    # Clean up the generated JSON (extract just the JSON part)
    try:
        # The model output appears to have the expected JSON appended at the end
        # Extract all valid JSON-like blocks from the text
        json_blocks = []
        brace_count = 0
        start_index = -1
        
        # Find JSON-like sections by tracking braces
        for i, char in enumerate(generated_json):
            if char == '{':
                if brace_count == 0:
                    start_index = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_index != -1:
                    # Found a potential JSON block
                    json_block = generated_json[start_index:i+1]
                    json_blocks.append(json_block)
        
        print(f"\nFound {len(json_blocks)} potential JSON blocks")
        
        # Try each block, preferring the last one
        valid_json = None
        for i, block in enumerate(reversed(json_blocks)):
            try:
                # Check if this looks like our expected schema
                if '"experience_level"' in block:
                    parsed = json.loads(block)
                    valid_json = block
                    print(f"Successfully parsed JSON block {len(json_blocks)-i} (counting from end)")
                    break
            except json.JSONDecodeError:
                # Try to clean up common issues with the JSON
                try:
                    # Fix trailing/malformed characters
                    clean_block = block.rstrip('}]')
                    if clean_block[-1] != '}':
                        clean_block += '}'
                    parsed = json.loads(clean_block)
                    valid_json = clean_block
                    print(f"Successfully parsed cleaned JSON block {len(json_blocks)-i} (counting from end)")
                    break
                except:
                    continue
        
        if valid_json:
            generated_json = valid_json
            parsed = True
            generated_json_parsed = json.loads(generated_json)
        else:
            print("No valid JSON blocks found")
            parsed = False
            generated_json_parsed = {}
    except Exception as e:
        print(f"Error in JSON extraction: {str(e)}")
        parsed = False
        generated_json_parsed = {}
    
    # Try to parse expected output
    try:
        expected_json = json.loads(expected_output)
    except:
        expected_json = {}
    
    # Store results
    result = {
        "example_id": i,
        "input_text": input_text[:200] + "...",  # Truncated for readability
        "generated_json": generated_json,
        "expected_json": expected_output,
        "parsed_successfully": parsed
    }
    
    results.append(result)
    
    # Print some information about the current example
    print(f"Example {i+1} processed")
    print(f"Parsed successfully: {parsed}")
    if parsed:
        # Print a few fields as examples
        print("\nSample of extracted fields:")
        fields_to_show = ["experience_level", "work_location", "required_minimum_degree"]
        for field in fields_to_show:
            if field in generated_json_parsed:
                print(f"  {field}: {generated_json_parsed.get(field, '')}")
    
    print("-" * 80)

# Calculate overall statistics
successful_parses = sum(1 for r in results if r["parsed_successfully"])
print(f"\nEvaluation complete!")
print(f"Successfully parsed JSON: {successful_parses}/{len(results)} ({successful_parses/len(results)*100:.1f}%)")

# After completing evaluation, calculate and print metrics
overall_metrics = calculate_metrics(results)

# Print metrics summary
print("\n===== EVALUATION METRICS =====")
print(f"Total samples: {overall_metrics['overall']['total_samples']}")
print(f"Parse success rate: {overall_metrics['overall']['parse_rate']:.2f}")
if 'mean_f1_score' in overall_metrics['overall']:
    print(f"Overall extraction quality (Mean F1): {overall_metrics['overall']['mean_f1_score']:.2f}")

# Print field-level metrics
print("\nField-level metrics:")
for field, metrics in overall_metrics.get('field_metrics', {}).items():
    print(f"  {field}:")
    print(f"    - Populated rate: {metrics['presence_rate']:.2f}")
    print(f"    - Accuracy: {metrics['accuracy_rate']:.2f}")

# Print metrics for list fields (with F1 scores)
print("\nList field metrics:")
for field, metrics in overall_metrics.get('list_field_metrics', {}).items():
    print(f"  {field}:")
    print(f"    - Precision: {metrics['precision']:.2f}")
    print(f"    - Recall: {metrics['recall']:.2f}")
    print(f"    - F1 Score: {metrics['f1_score']:.2f}")

# Save metrics to results file
results_with_metrics = {
    "evaluation_results": results,
    "metrics": overall_metrics
}

# Save results and metrics
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"{mistral_lora_results_dir}/mistral_lora_eval_{timestamp}.json"
with open(results_file, "w") as f:
    json.dump(results_with_metrics, f, indent=2)

print(f"Results and metrics saved to {results_file}")

# Print some examples of generated vs expected JSON
print("\nExamples of model outputs vs expected outputs:")
for i in range(min(3, len(results))):
    result = results[i]
    print(f"\nExample {i+1}:")
    print(f"Generated (truncated):\n{result['generated_json'][:200]}...")
    print(f"Expected (truncated):\n{result['expected_json'][:200]}...")
    print("-" * 80)

print("\nEvaluation complete!") 