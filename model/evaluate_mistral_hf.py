import json
import os
import random
import numpy as np
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Create results directory if it doesn't exist
results_dir = "evaluation_results"
os.makedirs(results_dir, exist_ok=True)

# Set your Hugging Face token as an environment variable
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"  # Your token
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_from_prepared_data(num_samples=10):
    """Load data from prepared_data.json file."""
    print("Loading data from prepared_data.json...")
    try:
        with open("prepared_data.json", "r") as f:
            data = json.load(f)
            print(f"Successfully loaded {len(data)} examples from prepared_data.json")
            
            # Select random samples for evaluation
            if len(data) > num_samples:
                # Set random seed for reproducibility
                random.seed(42)
                samples = random.sample(data, num_samples)
            else:
                samples = data
                
            return samples
    except FileNotFoundError:
        print("ERROR: prepared_data.json file not found!")
        return []
    except json.JSONDecodeError:
        print("ERROR: prepared_data.json has invalid JSON format!")
        return []

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

def evaluate_mistral_model(num_examples=10):
    """Evaluate the Mistral 7B model from HuggingFace."""
    # Load the model
    model_name = "alexlanxy/mistral_7b_lora_batch_1_single_server"
    print(f"Loading model {model_name}...")
    
    # Initialize tokenizer with proper padding settings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Using pad_token_id: {tokenizer.pad_token_id}")
    
    # Setup quantization config (load in 8-bit to save memory)
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
    
    # Load evaluation data
    eval_data = load_from_prepared_data(num_examples)
    if not eval_data:
        print("No data available for evaluation!")
        return None
    
    # Define the schema used during training
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
    
    # Create a timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"mistral_hf_eval_{timestamp}.json")
    
    # Results container
    results = []
    
    # Track metrics
    json_valid_count = 0
    coverage_values = []
    accuracy_values = []
    
    # Evaluate each sample
    print("\nStarting evaluation...")
    for i, data_item in enumerate(eval_data):
        input_text = data_item[0]
        expected_output = data_item[1]
        
        print(f"\nProcessing example {i+1}/{len(eval_data)}:")
        print(f"Input text length: {len(input_text)} characters")
        
        # Format messages following Mistral's chat format - exactly as in training
        messages = [
            {"role": "user", "content": f"Label the following job posting in pure JSON format based on this example schema. If no information for a field, leave the field blank.\n\nExample schema:\n{schema}\n\nJob posting:\n{input_text}"}
        ]
        
        # Use tokenizer to encode and handle message formatting
        chat_input = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
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
        
        # Extract just the model's response (not the input prompt)
        assistant_start = generated_text.find("<|assistant|>")
        if assistant_start != -1:
            generated_json = generated_text[assistant_start + len("<|assistant|>"):].strip()
        else:
            generated_json = generated_text
        
        # Clean up the generated JSON (extract just the JSON part)
        try:
            # Find the first { and the last }
            start_idx = generated_json.find('{')
            end_idx = generated_json.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                generated_json = generated_json[start_idx:end_idx]
            
            # Parse the JSON to validate it
            generated_json_parsed = json.loads(generated_json)
            json_valid = True
            json_valid_count += 1
            
            # Calculate coverage and accuracy
            try:
                expected_json = json.loads(expected_output)
                coverage = calculate_field_coverage(expected_json, generated_json_parsed)
                accuracy, field_accuracies = calculate_field_accuracy(expected_json, generated_json_parsed)
                
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
                print("Failed to parse expected output as JSON")
                coverage = 0
                accuracy = 0
                field_accuracies = {}
        except Exception as e:
            print(f"Error parsing generated JSON: {str(e)}")
            json_valid = False
            coverage = 0
            accuracy = 0
            field_accuracies = {}
            generated_json_parsed = {}
        
        # Store result
        result = {
            "example_id": i,
            "input_text": input_text[:200] + "...",  # Truncated for readability
            "expected_output": expected_output,
            "generated_json": generated_json,
            "json_valid": json_valid,
            "field_coverage": coverage if 'coverage' in locals() else 0,
            "field_accuracy": accuracy if 'accuracy' in locals() else 0,
            "field_accuracies": field_accuracies if 'field_accuracies' in locals() else {}
        }
        
        results.append(result)
        
        # Print some information about the current example
        print(f"Example {i+1} processed")
        print(f"JSON valid: {json_valid}")
        
        print("-" * 80)
    
    # Calculate overall statistics
    avg_coverage = np.mean(coverage_values) if coverage_values else 0
    avg_accuracy = np.mean(accuracy_values) if accuracy_values else 0
    json_valid_rate = (json_valid_count / len(eval_data)) * 100
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total examples evaluated: {len(eval_data)}")
    print(f"Valid JSON rate: {json_valid_rate:.2f}%")
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
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return {
        "results_file": results_file,
        "json_valid_rate": json_valid_rate,
        "avg_coverage": avg_coverage,
        "avg_accuracy": avg_accuracy,
        "field_avg_accuracies": field_avg_accuracies
    }

if __name__ == "__main__":
    # Run evaluation with 10 examples
    evaluate_mistral_model(num_examples=10)