import sys
import os
import json
import torch
import random
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from data_preprocessing import prepare_data

# Create results directory if it doesn't exist
results_dir = "evaluation_results"
os.makedirs(results_dir, exist_ok=True)

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

# Load just a small subset of data for testing
print("Loading data...")
data_pairs = prepare_data(prompt=True)
print(f"Loaded {len(data_pairs)} data pairs")

# Limit to first 20 items for faster testing
data_pairs = data_pairs[:20]
print(f"Using first {len(data_pairs)} items for evaluation")

# Select 10 random samples for evaluation
# Use a fixed seed for reproducibility
random.seed(42)
eval_samples = random.sample(data_pairs, 2)
print(f"Selected {len(eval_samples)} samples for evaluation")


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
        parsed = True
    except Exception as e:
        print(f"Error parsing generated JSON: {str(e)}")
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

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"{results_dir}/mistral_lora_eval_{timestamp}.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {results_file}")

# Print some examples of generated vs expected JSON
print("\nExamples of model outputs vs expected outputs:")
for i in range(min(3, len(results))):
    result = results[i]
    print(f"\nExample {i+1}:")
    print(f"Generated (truncated):\n{result['generated_json'][:200]}...")
    print(f"Expected (truncated):\n{result['expected_json'][:200]}...")
    print("-" * 80)

print("\nEvaluation complete!")