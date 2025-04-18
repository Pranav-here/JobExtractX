import json
import torch
import datetime
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm

# Function to evaluate model on test examples
def evaluate_model(model, tokenizer, test_data, max_source_length=1536, max_target_length=384, num_examples=2):
    results = []
    
    for i, data_item in enumerate(test_data[:num_examples]):
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
        
        # Check if generated JSON is valid
        try:
            json.loads(generated_json)
            parsed_successfully = True
        except json.JSONDecodeError:
            parsed_successfully = False
        
        # Add to results
        results.append({
            "example_id": i,
            "input_text": truncated_input,
            "generated_json": generated_json,
            "expected_json": expected_json,
            "parsed_successfully": parsed_successfully
        })
    
    return results

if __name__ == "__main__":
    # Load test data from prepared_data.json
    with open('prepared_data.json', 'r') as f:
        data_pairs = json.load(f)
    
    # Use last 10 examples as test data
    test_data = data_pairs[-10:]
    print(f"Loaded {len(data_pairs)} examples, using {len(test_data)} for testing")
    
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
    
    # Create timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results/flan_t5_xl_lora_eval_{timestamp}.json"
    
    # Ensure directory exists
    import os
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Save results to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed. Results saved to {output_file}")
    
    # Print examples
    print("\nEvaluation Examples:")
    for result in results:
        print(f"\nExample {result['example_id']}:")
        print(f"Generated JSON: {result['generated_json'][:100]}...")
        print(f"Parsed successfully: {result['parsed_successfully']}") 