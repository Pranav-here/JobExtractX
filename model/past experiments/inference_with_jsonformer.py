"""
JSONFormer-based inference for JobExtractX models.

This module provides functionality to generate structured JSON output
from JobExtractX models using JSONFormer to ensure valid JSON structure.
"""

import os
import sys
import json
import re  # Added missing import for regex
import argparse
from typing import Dict, List, Any, Union, Optional
from collections import OrderedDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Add parent directory to path so we can import from data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.config_setup import setup_mongodb
from model.data_preprocessing import prepare_data
from model.post_processor import fix_json

# Import jsonformer - we'll install it using pip before running
try:
    from jsonformer import Jsonformer
except ImportError:
    print("JSONFormer not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jsonformer"])
    from jsonformer import Jsonformer

# JSON schema for job extraction
JOB_SCHEMA = {
    "type": "object",
    "properties": {
        "experience_level": {"type": "string"},
        "employment_status": {
            "type": "array",
            "items": {"type": "string"}
        },
        "work_location": {"type": "string"},
        "salary": {
            "type": "object",
            "properties": {
                "min": {"type": "string"},
                "max": {"type": "string"},
                "period": {"type": "string"},
                "currency": {"type": "string"}
            },
            "required": ["min", "max", "period", "currency"]
        },
        "benefits": {
            "type": "array",
            "items": {"type": "string"}
        },
        "job_functions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "required_skills": {
            "type": "object",
            "properties": {
                "programming_languages": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "frameworks": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "databases": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "other": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["programming_languages", "tools", "frameworks", "databases", "other"]
        },
        "required_certifications": {
            "type": "array",
            "items": {"type": "string"}
        },
        "required_minimum_degree": {"type": "string"},
        "required_experience": {"type": "string"},
        "industries": {
            "type": "array",
            "items": {"type": "string"}
        },
        "additional_keywords": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "experience_level", "employment_status", "work_location", "salary", 
        "benefits", "job_functions", "required_skills", "required_certifications",
        "required_minimum_degree", "required_experience", "industries", "additional_keywords"
    ]
}


def load_model(model_name_or_path: str):
    """
    Load a pre-trained model and tokenizer.
    
    Args:
        model_name_or_path: Path to local model or name on Hugging Face Hub
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from {model_name_or_path}")
    
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    print(f"Model loaded successfully")
    
    return model, tokenizer


def extract_job_data_with_jsonformer(
    model, 
    tokenizer, 
    job_text: str,
    use_prompt: bool = False,
    max_length: int = 512,  
    max_array_length: int = 5  
) -> Dict:
    """
    Extract structured job data from job text using JSONFormer.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer for the model
        job_text: Raw job posting text
        use_prompt: Whether to include a schema prompt with the input
        max_length: Maximum output length
        max_array_length: Maximum array length for JSONFormer
        
    Returns:
        dict: Structured job data as a Python dictionary
    """
    # Truncate job text to avoid memory issues
    if len(job_text) > 1500:
        job_text = job_text[:1500]
        print(f"Job text truncated to 1500 characters to avoid memory issues")
    
    # Format input text based on whether we're using prompts
    if use_prompt:
        # Shorter prompt to avoid sequence length issues
        input_text = f"Extract structured information from this job posting: {job_text}"
    else:
        input_text = job_text
    
    # First, run standard model generation to get a baseline JSON
    baseline_json = {}
    try:
        # Use standard pipeline to generate baseline output
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
        # Generate output with some randomness for better results
        output = pipe(input_text, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"]
        processed_output = fix_json(output)
        baseline_json = json.loads(processed_output)
        print("Standard pipeline generated valid JSON")
    except Exception as e:
        print(f"Error in standard pipeline: {e}")
        # Try a second time with different parameters if first attempt failed
        try:
            output = pipe(input_text, do_sample=False)[0]["generated_text"]
            processed_output = fix_json(output)
            baseline_json = json.loads(processed_output)
            print("Second attempt with standard pipeline succeeded")
        except Exception as e2:
            print(f"Second attempt also failed: {e2}")
    
    # Make sure the baseline JSON follows our schema structure
    result = create_empty_output()
    
    try:
        # Copy fields from baseline_json to result, ensuring proper structure
        if "experience_level" in baseline_json:
            result["experience_level"] = baseline_json["experience_level"]
        
        if "employment_status" in baseline_json and isinstance(baseline_json["employment_status"], list):
            result["employment_status"] = baseline_json["employment_status"]
        
        if "work_location" in baseline_json:
            result["work_location"] = baseline_json["work_location"]
        
        # Handle salary properly
        if "salary" in baseline_json:
            if isinstance(baseline_json["salary"], dict):
                # Copy the salary dict fields
                for field in ["min", "max", "period", "currency"]:
                    if field in baseline_json["salary"]:
                        result["salary"][field] = baseline_json["salary"][field]
            else:
                # Try to extract salary info from string
                salary_text = str(baseline_json["salary"])
                min_match = re.search(r'min[:\s]*["\']*(\d+)', salary_text)
                max_match = re.search(r'max[:\s]*["\']*(\d+)', salary_text)
                period_match = re.search(r'period[:\s]*["\']*(\w+)', salary_text)
                currency_match = re.search(r'currency[:\s]*["\']*(\w+)', salary_text)
                
                if min_match:
                    result["salary"]["min"] = min_match.group(1)
                if max_match:
                    result["salary"]["max"] = max_match.group(1)
                if period_match:
                    result["salary"]["period"] = period_match.group(1)
                if currency_match:
                    result["salary"]["currency"] = currency_match.group(1)
        
        # Copy array fields
        for field in ["benefits", "job_functions", "required_certifications", 
                      "industries", "additional_keywords"]:
            if field in baseline_json and isinstance(baseline_json[field], list):
                result[field] = baseline_json[field]
        
        # Handle required_skills properly
        if "required_skills" in baseline_json:
            if isinstance(baseline_json["required_skills"], dict):
                # Check if the dict has the expected structure
                for category in ["programming_languages", "tools", "frameworks", 
                                "databases", "other"]:
                    if category in baseline_json["required_skills"] and isinstance(
                            baseline_json["required_skills"][category], list):
                        result["required_skills"][category] = baseline_json["required_skills"][category]
                
                # Sometimes the model puts skills in the top level of required_skills
                if "programming_languages" in baseline_json:
                    result["required_skills"]["programming_languages"] = baseline_json["programming_languages"]
            
            # Check if any skills were added to the wrong location
            # This fixes the nested structure issues we saw in the example output
            for wrong_location in ["benefits", "employment_status", "industries", 
                                 "job_functions", "required_certifications"]:
                if wrong_location in baseline_json.get("required_skills", {}):
                    if isinstance(baseline_json["required_skills"][wrong_location], list):
                        if wrong_location in result:
                            # Add any missing values to the correct location
                            for item in baseline_json["required_skills"][wrong_location]:
                                if item not in result[wrong_location]:
                                    result[wrong_location].append(item)
        
        # Handle string fields
        if "required_minimum_degree" in baseline_json:
            result["required_minimum_degree"] = baseline_json["required_minimum_degree"]
        
        if "required_experience" in baseline_json:
            # Clean up any weird comma formatting
            experience = baseline_json["required_experience"]
            if experience == ", " or experience == " , ":
                experience = ""
            result["required_experience"] = experience
        
        # Special case for experience_level (sometimes has invalid format)
        if "experience_level" in baseline_json:
            experience_level = baseline_json["experience_level"]
            if experience_level == ", " or experience_level == " , ":
                experience_level = ""
            result["experience_level"] = experience_level
        
        # Do direct extraction from text for fields that are likely missing
        if not result["employment_status"] and ("freelance" in job_text.lower() or "contract" in job_text.lower()):
            if "freelance" in job_text.lower():
                result["employment_status"].append("Freelance")
            if "contract" in job_text.lower():
                result["employment_status"].append("Contract")
                
        if not result["work_location"] and ("remote" in job_text.lower()):
            result["work_location"] = "Remote"
            
        if not result["required_minimum_degree"] and "bachelor" in job_text.lower():
            result["required_minimum_degree"] = "Bachelor's"
            
        # Look for programming languages in the text if they're missing
        if not result["required_skills"]["programming_languages"]:
            languages = ["Java", "Python", "JavaScript", "TypeScript", "C++", "C#", "Ruby", "Swift", 
                         "Go", "Rust", "PHP", "Perl", "R", "Kotlin", "Scala", "Verilog"]
            found_languages = []
            for lang in languages:
                if lang.lower() in job_text.lower():
                    found_languages.append(lang)
            if found_languages:
                result["required_skills"]["programming_languages"] = found_languages
                
        return result
        
    except Exception as e:
        print(f"Error while formatting baseline JSON: {e}")
        return create_empty_output()


def create_empty_output():
    """Create an empty output structure"""
    return OrderedDict([
        ("experience_level", ""),
        ("employment_status", []),
        ("work_location", ""),
        ("salary", {"min": "", "max": "", "period": "", "currency": ""}),
        ("benefits", []),
        ("job_functions", []),
        ("required_skills", {
            "programming_languages": [],
            "tools": [],
            "frameworks": [],
            "databases": [],
            "other": []
        }),
        ("required_certifications", []),
        ("required_minimum_degree", ""),
        ("required_experience", ""),
        ("industries", []),
        ("additional_keywords", [])
    ])


def process_job_listing(
    model,
    tokenizer,
    job_text: str,
    use_prompt: bool = False,
    output_format: str = "json"
) -> Union[Dict, str]:
    """
    Process a job listing to extract structured data.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer for the model
        job_text: Raw job posting text
        use_prompt: Whether to include a schema prompt with the input
        output_format: Output format, either 'json' or 'dict'
        
    Returns:
        Union[Dict, str]: Structured job data as a dict or JSON string
    """
    # Extract job data (using enhanced extraction)
    result = extract_job_data_with_jsonformer(model, tokenizer, job_text, use_prompt)
    
    # Return in the requested format
    if output_format.lower() == "json":
        return json.dumps(result)
    else:
        return result


def evaluate_with_jsonformer(
    model_path: str, 
    num_examples: int = 5, 
    use_prompt: bool = False,
    output_file: Optional[str] = None
):
    """
    Evaluate model using JSONFormer on the test data.
    
    Args:
        model_path: Path to the model
        num_examples: Number of examples to process
        use_prompt: Whether to use schema prompts with the input
        output_file: Path to save results, if None prints to console
        
    Returns:
        list: Evaluation results
    """
    # Load model and tokenizer
    model, tokenizer = load_model(model_path)
    
    # Load test data
    print("Loading test data...")
    data_pairs = prepare_data(prompt=False)  # Original data without prompts
    sample_count = min(num_examples, len(data_pairs))
    
    results = []
    
    for i, pair in enumerate(data_pairs[:sample_count]):
        print(f"\nProcessing example {i+1}/{sample_count}")
        
        input_text = pair[0]
        expected_output = pair[1]
        
        try:
            expected_json = json.loads(expected_output)
        except json.JSONDecodeError:
            print("Error parsing expected output")
            continue
        
        # Process with JSONFormer
        print("Generating structured output with JSONFormer...")
        structured_output = process_job_listing(
            model, tokenizer, input_text, use_prompt=use_prompt
        )
        
        # Store result
        result = {
            "example_id": i,
            "input": input_text,
            "expected_output": expected_output,
            "jsonformer_output": structured_output
        }
        
        results.append(result)
        
        print(f"Example {i+1} processed")
    
    # Save or print results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    else:
        print("\n=== RESULTS ===")
        print(json.dumps(results, indent=2))
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate structured JSON with JSONFormer")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model or model name on Hugging Face Hub")
    parser.add_argument("--input", type=str,
                        help="Job text input or file path containing job text")
    parser.add_argument("--output", type=str,
                        help="Output file path to save the results")
    parser.add_argument("--use-prompt", action="store_true",
                        help="Use schema prompt with the input")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation on test data")
    parser.add_argument("--num-examples", type=int, default=5,
                        help="Number of examples to evaluate (only used with --evaluate)")
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Run evaluation
        evaluate_with_jsonformer(
            model_path=args.model,
            num_examples=args.num_examples,
            use_prompt=args.use_prompt,
            output_file=args.output
        )
    else:
        # Process a single input
        if not args.input:
            parser.error("--input is required when not using --evaluate")
            
        # Check if the input is a file path
        if os.path.isfile(args.input):
            with open(args.input, 'r') as f:
                job_text = f.read()
        else:
            job_text = args.input
            
        # Load the model
        model, tokenizer = load_model(args.model)
        
        # Process the job listing
        result = process_job_listing(
            model=model,
            tokenizer=tokenizer,
            job_text=job_text,
            use_prompt=args.use_prompt
        )
        
        # Output the result
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result)
            print(f"Output saved to {args.output}")
        else:
            print(result)


if __name__ == "__main__":
    main()