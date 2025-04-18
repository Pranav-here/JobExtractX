import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Set Hugging Face token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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" 

# Load model and tokenizer
print("Loading pretrained Mistral-7B-Instruct-v0.3...")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Initialize tokenizer with proper padding settings
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Setup quantization config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# Load model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Model and tokenizer loaded successfully!")

# Simple job data
job_example = {
    "title": "Software Engineer",
    "description": "We are looking for a Python developer with 3+ years of experience. Remote work available. Bachelor's degree required."
}

# Create a simple, structured prompt
prompt = f"""
I need to extract job information from this posting into JSON format.

Job posting:
{job_example}

Extract the following fields (leave blank if information not present):
- experience_level (entry/mid/senior)
- work_location (remote/onsite/hybrid)
- required_skills.programming_languages (list)
- required_minimum_degree
- required_experience (years)

Format as valid JSON.
"""

# Format as a chat message
messages = [{"role": "user", "content": prompt}]

# Generate with the model
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=200,
        temperature=0.1,
        do_sample=True
    )

# Decode the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nModel response:")
print("-" * 80)
print(response)
print("-" * 80)

# Try to extract JSON
print("\nAttempting to find and extract the JSON object...")

# Look for lines after the prompt that might be JSON
content_lines = response.split('\n')
json_start = None
json_end = None
for i, line in enumerate(content_lines):
    # Skip initial prompt
    if "Format as valid JSON" in line:
        json_start = i + 1  # Start on next line
        break

if json_start is not None:
    # Extract from where the JSON likely starts
    potential_json_lines = []
    for i in range(json_start, len(content_lines)):
        line = content_lines[i].strip()
        if line:  # Only include non-empty lines
            potential_json_lines.append(line)
    
    # Join all the potential JSON lines
    potential_json = ' '.join(potential_json_lines)
    
    # Try to extract just the JSON object
    try:
        # Find first { and last }
        first_brace = potential_json.find('{')
        last_brace = potential_json.rfind('}')
        
        if first_brace != -1 and last_brace != -1:
            json_str = potential_json[first_brace:last_brace+1]
            
            # Try to parse this as JSON
            parsed_json = json.loads(json_str)
            
            print("\nSuccessfully extracted JSON:")
            print(json.dumps(parsed_json, indent=2))
            
            print("\nThe pretrained model correctly extracted:")
            print(f"- Experience level: {parsed_json.get('experience_level', 'Not found')}")
            print(f"- Work location: {parsed_json.get('work_location', 'Not found')}")
            if 'required_skills' in parsed_json and 'programming_languages' in parsed_json['required_skills']:
                print(f"- Programming languages: {', '.join(parsed_json['required_skills']['programming_languages'])}")
            print(f"- Required degree: {parsed_json.get('required_minimum_degree', 'Not found')}")
            print(f"- Years of experience: {parsed_json.get('required_experience', 'Not found')}")
        else:
            print("Couldn't find JSON object markers { } in the response.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print("Raw potential JSON string:")
        print(potential_json)
else:
    print("Couldn't find the start of JSON content in the response.") 