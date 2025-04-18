import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re

# Set Hugging Face token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load model and tokenizer
print("Loading pretrained Mistral-7B-Instruct-v0.3...")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Setup quantization config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# Initialize tokenizer with proper padding settings
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Model and tokenizer loaded successfully!")

# Define schema similar to your training examples
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

# Sample job data from our tests
job_data = """
{'title': 'Software Engineer', 'description': 'We are looking for a software engineer with Python experience. The role is remote with competitive salary. Bachelor's degree required.'}
"""

# Format the prompt with explicit instructions
prompt = f"""
Extract job information from the posting into structured JSON. Follow this exact schema, leaving values blank or empty lists if information is not present.

Schema:
{schema}

Job Posting:
{job_data}

Return ONLY the completed JSON with the extracted information.
"""

# Format messages following Mistral's chat format
messages = [
    {"role": "user", "content": prompt}
]

# Encode messages for model
input_ids = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

print("\nGenerating response...")
# Generate text
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        temperature=0.1,  # Lower temperature for more deterministic outputs
        top_p=0.9,
        do_sample=True
    )

# Decode the generated output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated output:")
print("-" * 80)
print(generated_text)
print("-" * 80)

# Find the last empty line before the JSON response
lines = generated_text.split('\n')
response_lines = []
found_empty_line = False

for line in reversed(lines):
    if not line.strip() and not found_empty_line:
        found_empty_line = True
        continue
    
    if found_empty_line:
        response_lines.insert(0, line)

response_text = '\n'.join(response_lines).strip()
print("\nExtracted response:")
print("-" * 80)
print(response_text)
print("-" * 80)

# Try to extract valid JSON from the response
# Pattern to find a JSON object
json_pattern = r'(\{\s*"experience_level"\s*:.*?\})'
json_matches = re.findall(json_pattern, response_text, re.DOTALL)

if json_matches:
    json_str = json_matches[0]
    print("\nFound JSON match:")
    print(json_str)
    
    try:
        # Try to parse the JSON
        parsed_json = json.loads(json_str)
        print("\nValid JSON extracted:")
        print(json.dumps(parsed_json, indent=2))
        print("\nSuccess! The pretrained model returned valid JSON.")
    except json.JSONDecodeError as e:
        print(f"\nJSON parsing error: {str(e)}")
        
        # Try manual cleanup
        json_str = json_str.replace("\n", " ").replace("  ", " ")
        try:
            parsed_json = json.loads(json_str)
            print("\nValid JSON extracted after cleanup:")
            print(json.dumps(parsed_json, indent=2))
            print("\nSuccess after cleanup!")
        except:
            print("Failed to parse even after cleanup.")
else:
    print("\nNo JSON pattern found in the response.") 