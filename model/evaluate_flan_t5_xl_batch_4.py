import json
import torch
import datetime
import re
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from collections import Counter

def fix_json_format(json_str):
    """
    Fix common JSON formatting issues in the model's output.
    Uses a more robust approach to handling JSON formatting errors.
    """
    # Check if already properly formatted
    try:
        json.loads(json_str)
        return json_str
    except:
        pass
    
    # Create a template object with the expected structure
    fixed_json = {
        "experience_level": "",
        "employment_status": [],
        "work_location": "",
        "salary": {"min": "", "max": "", "period": "", "currency": ""},
        "benefits": [],
        "job_functions": [],
        "required_skills": {
            "programming_languages": [],
            "tools": [],
            "frameworks": [],
            "databases": [],
            "other": []
        },
        "required_certifications": [],
        "required_minimum_degree": "",
        "required_experience": "",
        "industries": [],
        "additional_keywords": []
    }
    
    # Add outer braces if missing
    if not json_str.strip().startswith('{'):
        json_str = '{' + json_str + '}'
    
    # Extract experience_level
    exp_match = re.search(r'"experience_level":\s*"([^"]*)"', json_str)
    if exp_match:
        fixed_json["experience_level"] = exp_match.group(1)
    
    # Extract employment_status
    emp_match = re.search(r'"employment_status":\s*\[(.*?)\]', json_str)
    if emp_match:
        emp_items = re.findall(r'"([^"]+)"', emp_match.group(1))
        fixed_json["employment_status"] = emp_items
    
    # Extract work_location
    loc_match = re.search(r'"work_location":\s*"([^"]*)"', json_str)
    if loc_match:
        fixed_json["work_location"] = loc_match.group(1)
    
    # Extract salary components
    min_match = re.search(r'"min":\s*"([^"]*)"', json_str)
    max_match = re.search(r'"max":\s*"([^"]*)"', json_str)
    period_match = re.search(r'"period":\s*"([^"]*)"', json_str)
    currency_match = re.search(r'"currency":\s*"([^"]*)"', json_str)
    
    if min_match:
        fixed_json["salary"]["min"] = min_match.group(1)
    if max_match:
        fixed_json["salary"]["max"] = max_match.group(1)
    if period_match:
        fixed_json["salary"]["period"] = period_match.group(1)
    if currency_match:
        fixed_json["salary"]["currency"] = currency_match.group(1)
    
    # Extract benefits
    ben_match = re.search(r'"benefits":\s*\[(.*?)\]', json_str)
    if ben_match:
        ben_items = re.findall(r'"([^"]+)"', ben_match.group(1))
        fixed_json["benefits"] = ben_items
    
    # Extract job_functions
    job_match = re.search(r'"job_functions":\s*\[(.*?)\]', json_str)
    if job_match:
        job_items = re.findall(r'"([^"]+)"', job_match.group(1))
        fixed_json["job_functions"] = job_items
    
    # Extract required_skills
    # Programming languages
    prog_match = re.search(r'"programming_languages":\s*\[(.*?)\]', json_str)
    if prog_match:
        prog_items = re.findall(r'"([^"]+)"', prog_match.group(1))
        fixed_json["required_skills"]["programming_languages"] = prog_items
    
    # Tools
    tools_match = re.search(r'"tools":\s*\[(.*?)\]', json_str)
    if tools_match:
        tools_items = re.findall(r'"([^"]+)"', tools_match.group(1))
        fixed_json["required_skills"]["tools"] = tools_items
    
    # Frameworks
    frame_match = re.search(r'"frameworks":\s*\[(.*?)\]', json_str)
    if frame_match:
        frame_items = re.findall(r'"([^"]+)"', frame_match.group(1))
        fixed_json["required_skills"]["frameworks"] = frame_items
    
    # Databases
    db_match = re.search(r'"databases":\s*\[(.*?)\]', json_str)
    if db_match:
        db_items = re.findall(r'"([^"]+)"', db_match.group(1))
        fixed_json["required_skills"]["databases"] = db_items
    
    # Other skills
    other_match = re.search(r'"other":\s*\[(.*?)\]', json_str)
    if other_match:
        other_items = re.findall(r'"([^"]+)"', other_match.group(1))
        fixed_json["required_skills"]["other"] = other_items
    
    # Extract required_certifications
    cert_match = re.search(r'"required_certifications":\s*\[(.*?)\]', json_str)
    if cert_match:
        cert_items = re.findall(r'"([^"]+)"', cert_match.group(1))
        fixed_json["required_certifications"] = cert_items
    
    # Extract required_minimum_degree
    degree_match = re.search(r'"required_minimum_degree":\s*"([^"]*)"', json_str)
    if degree_match:
        fixed_json["required_minimum_degree"] = degree_match.group(1)
    
    # Extract required_experience
    exp_match = re.search(r'"required_experience":\s*"([^"]*)"', json_str)
    if exp_match:
        fixed_json["required_experience"] = exp_match.group(1)
    
    # Extract industries
    ind_match = re.search(r'"industries":\s*\[(.*?)\]', json_str)
    if ind_match:
        ind_items = re.findall(r'"([^"]+)"', ind_match.group(1))
        fixed_json["industries"] = ind_items
    
    # Extract additional_keywords
    key_match = re.search(r'"additional_keywords":\s*\[(.*?)\]', json_str)
    if key_match:
        key_items = re.findall(r'"([^"]+)"', key_match.group(1))
        fixed_json["additional_keywords"] = key_items
    
    # Convert back to JSON string
    return json.dumps(fixed_json)

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
            generated = json.loads(result["fixed_json"])
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

# Function to evaluate model on test examples
def evaluate_model(model, tokenizer, test_data, max_source_length=1536, max_target_length=384, num_examples=1000):
    results = []
    
    for i, data_item in tqdm(enumerate(test_data[:num_examples]), total=min(num_examples, len(test_data)), desc="Evaluating model"):
        # Extract source and target based on data structure
        source_text = data_item["source"]
        expected_json = data_item["target"]
        
        # Create the prompt with schema as in training
        schema = """
        {
            "experience_level": "",
            "employment_status": [],
            "work_location": "",
            "salary": {"min": "", "max": "", "period": "", "currency": ""},
            "benefits": [],
            "job_functions": [],
            "required_skills": {
                "programming_languages": [],
                "tools": [],
                "frameworks": [],
                "databases": [],
                "other": []
            },
            "required_certifications": [],
            "required_minimum_degree": "",
            "required_experience": "",
            "industries": [],
            "additional_keywords": []
            }
        """
        
        prompt_text = (
            f"Label the following job posting in pure JSON format based on this example schema. "
            f"If no information for a field, leave the field blank.\n\n"
            f"Example schema:\n{schema}\n\n"
            f"Job posting:\n{source_text}"
        )
        
        # Truncate input text for display in results
        truncated_input = prompt_text[:150] + "..."
        
        # Tokenize and generate
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_source_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate text
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_target_length,
            num_beams=4,
            early_stopping=True,
            use_cache=True
        )
        
        # Decode output
        generated_json = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Apply post-processing to fix JSON formatting
        fixed_json = fix_json_format(generated_json)
        
        # Check if generated JSON is valid
        try:
            json.loads(fixed_json)
            parsed_successfully = True
        except json.JSONDecodeError:
            parsed_successfully = False
            print(f"Failed to parse JSON: {generated_json[:50]}...")
        
        # Add to results
        results.append({
            "example_id": i,
            # "input_text": truncated_input,
            "generated_json": generated_json,
            "fixed_json": fixed_json,
            "expected_json": expected_json,
            "parsed_successfully": parsed_successfully
        })
    
    return results

if __name__ == "__main__":
    # Load test data from prepared_data.json
    with open('prepared_data.json', 'r') as f:
        data_pairs = json.load(f)
    
    # Use only the last 10% of the data for evaluation (since first 90% was used for training)
    total_samples = len(data_pairs)
    test_start_idx = int(total_samples * 0.9)  # Start at 90% mark
    test_data = data_pairs[test_start_idx:]
    
    print(f"Loaded {len(data_pairs)} examples, using {len(test_data)} for testing (last 10%)")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer from Hugging Face
    model_name = "alexlanxy/flan_t5_xl_lora_prompt_bf16_batch_4"
    print(f"Loading model from {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load PEFT model 
    config = PeftConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, model_name)
    model.eval()
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, tokenizer, test_data)
    
    # Calculate metrics
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
    
    # Create timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create combined results with metrics
    results_with_metrics = {
        "evaluation_results": results,
        "metrics": overall_metrics
    }
    
    # Ensure directory exists
    import os
    os.makedirs("evaluation_results/flan_t5_xl_lora_batch_4_eval", exist_ok=True)
    
    # Save results to file
    output_file = f"evaluation_results/flan_t5_xl_lora_batch_4_eval/flan_t5_xl_lora_eval_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results_with_metrics, f, indent=2)
    
    print(f"Evaluation completed. Results saved to {output_file}")
    
    # Print examples
    print("\nEvaluation Examples:")
    for i, result in enumerate(results):
        print(f"\nExample {i}:")
        print(f"Generated JSON: {result['generated_json'][:100]}...")
        print(f"Parsed successfully: {result['parsed_successfully']}")
        if i >= 2:  # Only show first 3 examples
            break 