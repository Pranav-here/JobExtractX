import json
import re
import os
import glob
from datetime import datetime

def fix_json(text):
    """
    Attempt to fix truncated or malformed JSON from model output.
    """
    if not text:
        return "{}"
    
    # First, handle the case where the output starts with a field name
    if text.strip().startswith('"'):
        text = "{" + text + "}"
    
    # Fix missing commas between fields
    text = re.sub(r'"\s*"', '", "', text)
    
    # Fix malformed nested objects
    # Look for patterns like "programming_languages": ["Python"] without proper nesting
    nested_field_pattern = r'"([^"]+)":\s*"([^"]+)":\s*(\[[^\]]*\])'
    if re.search(nested_field_pattern, text):
        # Replace with proper nesting
        text = re.sub(nested_field_pattern, r'"\1": {"\2": \3}', text)
    
    # Add closing brackets for arrays if needed
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    if open_brackets > close_brackets:
        text += ']' * (open_brackets - close_brackets)
    
    # Balance quotes
    quotes = text.count('"')
    if quotes % 2 != 0:
        text += '"'
    
    # Handle special case where required_skills field is incorrectly formatted
    skills_pattern = r'"required_skills":\s*"([^"]+)":\s*(\[[^\]]*\])'
    if re.search(skills_pattern, text):
        text = re.sub(skills_pattern, r'"required_skills": {"\1": \2}', text)
    
    # Try to parse the fixed JSON
    try:
        json_obj = json.loads(text)
        return json.dumps(json_obj)
    except json.JSONDecodeError as e:
        # If we still can't parse it, try a more aggressive approach
        try:
            # Parse line by line and reconstruct
            fields = {}
            
            # Extract field names and values using regex
            field_pattern = r'"([^"]+)":\s*(\[[^\]]*\]|"[^"]*"|[^,}\]]+)'
            matches = re.findall(field_pattern, text)
            
            for field_name, field_value in matches:
                # Process nested fields
                if field_name == "required_skills":
                    # Attempt to extract nested skills
                    skills_dict = {}
                    nested_matches = re.findall(r'"([^"]+)":\s*(\[[^\]]*\])', text)
                    for nested_name, nested_value in nested_matches:
                        if nested_name != "required_skills" and nested_name != "additional_keywords":
                            try:
                                # Convert string representation of list to actual list
                                value_str = nested_value.strip()
                                if value_str.startswith('[') and value_str.endswith(']'):
                                    list_items = re.findall(r'"([^"]+)"', value_str)
                                    skills_dict[nested_name] = list_items
                            except:
                                pass
                    
                    if skills_dict:
                        fields[field_name] = skills_dict
                    else:
                        fields[field_name] = {}
                else:
                    # Process regular fields
                    value_str = field_value.strip()
                    
                    # Handle lists
                    if value_str.startswith('[') and value_str.endswith(']'):
                        try:
                            list_items = re.findall(r'"([^"]+)"', value_str)
                            fields[field_name] = list_items
                        except:
                            fields[field_name] = []
                    # Handle strings
                    elif value_str.startswith('"') and value_str.endswith('"'):
                        fields[field_name] = value_str.strip('"')
                    # Handle other values
                    else:
                        fields[field_name] = value_str
            
            # Return the reconstructed JSON
            return json.dumps(fields)
        except Exception as ex:
            print(f"Failed to fix JSON: {ex}")
            return "{}"

def process_results_file(file_path):
    """
    Process all model outputs in a results file and fix the JSON formatting.
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    fixed_count = 0
    for result in results:
        generated_text = result.get('generated_output', '')
        fixed_json = fix_json(generated_text)
        
        # Update the result with the fixed JSON
        result['processed_output'] = fixed_json
        
        # Check if we successfully fixed it
        try:
            json.loads(fixed_json)
            if fixed_json != "{}":
                fixed_count += 1
                result['json_fixed'] = True
            else:
                result['json_fixed'] = False
        except:
            result['json_fixed'] = False
    
    # Save the updated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.path.dirname(file_path), f"processed_results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} results")
    print(f"Successfully fixed {fixed_count} outputs ({fixed_count/len(results)*100:.2f}%)")
    print(f"Saved to {output_file}")
    
    return output_file

def find_latest_results():
    """Find the most recent evaluation results file."""
    results_dir = "evaluation_results"
    result_files = glob.glob(os.path.join(results_dir, "evaluation_results_*.json"))
    if not result_files:
        print("No evaluation results found.")
        return None
    
    # Sort by modification time (newest first)
    latest_file = max(result_files, key=os.path.getmtime)
    return latest_file

if __name__ == "__main__":
    latest_file = find_latest_results()
    
    if latest_file:
        print(f"Processing latest results file: {latest_file}")
        processed_file = process_results_file(latest_file)
        
        # Print a sample of fixed outputs
        with open(processed_file, 'r') as f:
            results = json.load(f)
        
        print("\n===== SAMPLE OF FIXED OUTPUTS =====")
        for i, result in enumerate(results[:3]):
            if result.get('json_fixed', False):
                print(f"\nExample {i+1}:")
                try:
                    parsed = json.loads(result['processed_output'])
                    # Print first few fields for readability
                    sample = {k: parsed[k] for k in list(parsed.keys())[:3]}
                    print(f"Fixed JSON (first 3 fields): {json.dumps(sample, indent=2)}")
                except:
                    print("Error parsing fixed JSON")
    else:
        print("No results file found. Run model/evaluate.py first.") 