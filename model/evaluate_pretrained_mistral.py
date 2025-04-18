import sys
import os
import json
import torch
import random
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Create results directory if it doesn't exist
results_dir = "evaluation_results"
os.makedirs(results_dir, exist_ok=True)

# Set your Hugging Face token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Use the pretrained Mistral model instead of fine-tuned model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
print(f"Loading pretrained model {model_name}...")

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

# Load data from JSON file
print("Loading data from prepared_data.json...")
with open("prepared_data.json", "r") as f:
    json_data = json.load(f)

# Define the schema for extraction
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

# Format the data into input-output pairs
data_pairs = []
for item in json_data:
    job_posting = item['source']
    expected_output = item['target']
    
    # Enhanced prompt with specific structured guidelines and examples
    input_text = f"""You are JobExtractX, a specialized job information extraction assistant. Your task is to carefully analyze job postings and extract ALL relevant information into structured JSON.

Job Posting:
{job_posting}

CRITICAL INSTRUCTIONS:
1. CAREFULLY READ the entire job posting
2. EXTRACT ACTUAL VALUES for each field below - empty JSON is NOT acceptable
3. For each field, you MUST find and extract the specific information:
   - experience_level: Extract "Entry-level", "Mid-level", "Senior", etc.
   - employment_status: Extract ALL that apply from ["Full-time", "Part-time", "Contract", "Freelance", "Internship"]
   - work_location: Extract "Remote", "Hybrid", or "On-site"
   - salary: Extract EXACT numbers for min/max, period (yearly/monthly/hourly), and currency
   - benefits: Extract ALL mentioned benefits (health insurance, vacation, etc.)
   - job_functions: Extract ALL responsibilities and duties
   - required_skills: Categorize ALL technical and non-technical skills
   - required_certifications: Extract ANY required certifications
   - required_minimum_degree: Extract exact education requirement
   - required_experience: Extract years or level of experience required
   - industries: Extract ALL mentioned industries
   - additional_keywords: Extract other important terms

4. FOCUS ON INFORMATION EXTRACTION - Look thoroughly through the entire posting for any mention of these attributes
5. MAKE EDUCATED INFERENCES when information is implied but not explicitly stated
6. If information truly cannot be found after thorough search, ONLY THEN use empty values
7. YOUR PRIMARY GOAL IS INFORMATION EXTRACTION - not just creating valid JSON structure

RESPOND ONLY with the extracted information in this exact JSON schema:
{schema}

JSON MUST contain ACTUAL EXTRACTED VALUES, not empty fields. Empty JSON is considered a failed extraction."""
    
    data_pairs.append((input_text, expected_output))

print(f"Loaded {len(data_pairs)} data pairs")

# Limit to first 20 items for faster testing
data_pairs = data_pairs[:20]
print(f"Using first {len(data_pairs)} items for evaluation")

# Select random samples for evaluation
# Use a fixed seed for reproducibility
random.seed(42)
eval_samples = random.sample(data_pairs, 2)
print(f"Selected {len(eval_samples)} samples for evaluation")

# Helper function to extract JSON from model output
def extract_json_from_response(response_text):
    # Try to extract valid JSON from the response
    try:
        # Find JSON object by looking for braces
        content_lines = response_text.split('\n')
        potential_json_lines = []
        
        # Skip empty lines and collect potential JSON
        for line in content_lines:
            if line.strip():
                potential_json_lines.append(line.strip())
        
        # Join all non-empty lines
        potential_json = ' '.join(potential_json_lines)
        
        # Find the opening and closing braces of the JSON object
        first_brace = potential_json.find('{')
        last_brace = potential_json.rfind('}')
        
        if first_brace != -1 and last_brace != -1:
            json_str = potential_json[first_brace:last_brace+1]
            
            # Try to parse this as JSON
            parsed_json = json.loads(json_str)
            return json_str, parsed_json, True
        else:
            return None, None, False
    except Exception as e:
        print(f"Error extracting JSON: {str(e)}")
        return None, None, False

# Results container
results = []

# Let's add a function to compare the extracted fields with expected
def compare_json_fields(generated, expected):
    """Compare key fields between generated and expected JSON"""
    if not generated or not expected:
        return {}
    
    comparison = {}
    
    # Key fields to compare
    fields = [
        "experience_level",
        "employment_status",
        "work_location",
        "required_minimum_degree",
        "required_experience",
    ]
    
    # Compare each field
    for field in fields:
        gen_value = generated.get(field, None)
        exp_value = expected.get(field, None)
        
        # Handle special case for lists
        if isinstance(gen_value, list) and isinstance(exp_value, list):
            # Check for any overlap
            overlap = any(item in exp_value for item in gen_value)
            comparison[field] = {
                "generated": gen_value,
                "expected": exp_value,
                "match": overlap
            }
        else:
            # Direct comparison
            comparison[field] = {
                "generated": gen_value,
                "expected": exp_value,
                "match": gen_value == exp_value
            }
    
    return comparison

# Evaluate each sample
print("\nStarting evaluation...")
for i, (input_text, expected_output) in enumerate(eval_samples):
    print(f"\nProcessing example {i+1}/{len(eval_samples)}:")
    print(f"Input text length: {len(input_text)} characters")
    
    # Format messages for Mistral's chat format
    messages = [
        {"role": "user", "content": input_text}
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
            max_new_tokens=1024,  # Increased for more complete JSON
            temperature=0.1,      # Lower temperature for more deterministic outputs
            top_p=0.9,
            do_sample=True
        )
    
    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Generated response length: {len(generated_text)} characters")
    
    # Print a small snippet of the output for debugging
    print("\nGenerated text snippet:")
    print("-" * 80)
    print(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)
    print("-" * 80)
    
    # Extract JSON from the response using an improved method
    def extract_json_from_text(text):
        try:
            # Look for JSON-like patterns across multiple lines
            # Start with finding all opening braces and their positions
            opening_positions = [pos for pos, char in enumerate(text) if char == '{']
            
            for start_pos in opening_positions:
                # Track nested braces
                stack = 1
                for i in range(start_pos + 1, len(text)):
                    if text[i] == '{':
                        stack += 1
                    elif text[i] == '}':
                        stack -= 1
                    
                    # When stack is empty, we've found a complete JSON object
                    if stack == 0:
                        potential_json = text[start_pos:i+1]
                        
                        # Check if this looks like our schema
                        if '"experience_level"' in potential_json:
                            try:
                                parsed_json = json.loads(potential_json)
                                return potential_json, parsed_json, True
                            except json.JSONDecodeError:
                                # Try some basic cleanup
                                clean_json = potential_json.replace('\n', ' ').replace('  ', ' ')
                                try:
                                    parsed_json = json.loads(clean_json)
                                    return clean_json, parsed_json, True
                                except:
                                    # Failed to parse this candidate, continue to next
                                    pass
            
            # If we get here, no valid JSON was found
            return None, None, False
        except Exception as e:
            print(f"Error in JSON extraction: {str(e)}")
            return None, None, False
    
    # Try to extract JSON from the response
    generated_json, generated_json_parsed, parsed = extract_json_from_text(generated_text)
    
    if not parsed:
        print("First extraction attempt failed, trying to locate specific patterns...")
        # Try to find just a JSON section with key fields we're expecting
        # Look for a simpler pattern with the exact schema we provided
        import re
        pattern = r'({[\s\S]*?"experience_level"[\s\S]*?"employment_status"[\s\S]*?})'
        matches = re.findall(pattern, generated_text)
        
        if matches:
            print(f"Found {len(matches)} potential JSON matches with regex")
            for match in matches:
                try:
                    # Try to clean and parse
                    clean_match = match.replace('\n', ' ').replace('  ', ' ')
                    parsed_json = json.loads(clean_match)
                    generated_json = clean_match
                    generated_json_parsed = parsed_json
                    parsed = True
                    print("Successfully parsed JSON after cleanup!")
                    break
                except json.JSONDecodeError as e:
                    print(f"Failed to parse match: {str(e)}")
    
    # Try to parse expected output
    try:
        expected_json = json.loads(expected_output)
    except:
        expected_json = {}
    
    # Store results
    result = {
        "example_id": i,
        "input_text": input_text[:200] + "...",  # Truncated for readability
        "generated_json": generated_json if generated_json else generated_text[:200] + "...",
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
        fields_to_show = ["experience_level", "work_location", "required_minimum_degree", "employment_status"]
        for field in fields_to_show:
            if field in generated_json_parsed:
                value = generated_json_parsed.get(field, "")
                if isinstance(value, list):
                    value = ", ".join(value) if value else "(empty list)"
                print(f"  {field}: {value}")
                
        # Compare with expected output
        if expected_json:
            print("\nComparison with expected output:")
            comparison = compare_json_fields(generated_json_parsed, expected_json)
            for field, result in comparison.items():
                match_status = "✓" if result["match"] else "✗"
                print(f"  {field}: {match_status}")
                if not result["match"]:
                    print(f"    - Generated: {result['generated']}")
                    print(f"    - Expected: {result['expected']}")
    
    print("-" * 80)

# Calculate overall statistics
successful_parses = sum(1 for r in results if r["parsed_successfully"])
print(f"\nEvaluation complete!")
print(f"Successfully parsed JSON: {successful_parses}/{len(results)} ({successful_parses/len(results)*100:.1f}%)")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"{results_dir}/pretrained_mistral_eval_{timestamp}.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {results_file}")

# Print some examples of generated vs expected JSON
print("\nExamples of model outputs vs expected outputs:")
for i in range(min(3, len(results))):
    result = results[i]
    print(f"\nExample {i+1}:")
    
    # Show generated JSON if available, otherwise show the start of the generated text
    if result["parsed_successfully"]:
        # Format the JSON nicely for display
        try:
            generated_obj = json.loads(result["generated_json"])
            print(f"Generated (truncated):\n{json.dumps(generated_obj, indent=2)[:200]}...")
        except:
            print(f"Generated (truncated):\n{result['generated_json'][:200]}...")
    else:
        print(f"Generated (failed to parse):\n{result['generated_json'][:200]}...")
    
    # Show expected JSON
    try:
        expected_obj = json.loads(result["expected_json"])
        print(f"Expected (truncated):\n{json.dumps(expected_obj, indent=2)[:200]}...")
    except:
        print(f"Expected (truncated):\n{result['expected_json'][:200]}...")
    
    print("-" * 80)

print("\nEvaluation complete!") 