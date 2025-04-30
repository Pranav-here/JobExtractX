import argparse
import json
import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import re
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
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
    return json.dumps(fixed_json, indent=2)

def load_flan_t5_model(model_name):
    """Load a FLAN-T5 model (regular or LoRA)"""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check if this is a LoRA model
    if "lora" in model_name.lower():
        # Load PEFT model 
        config = PeftConfig.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(model, model_name)
    else:
        # Load regular model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    
    model.eval()
    return model, tokenizer

def load_mistral_model(model_name):
    """Load a Mistral model"""
    print(f"Loading model: {model_name}")
    
    # Initialize tokenizer with proper padding settings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Setup quantization config (load in 8-bit to save memory)
    from transformers import BitsAndBytesConfig
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
    
    model.eval()
    return model, tokenizer

def infer_flan_t5(model, tokenizer, job_text, max_source_length=1536, max_target_length=384):
    """Run inference with a FLAN-T5 model"""
    # Create the prompt with schema
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
        f"Job posting:\n{job_text}"
    )
    
    # Tokenize and generate
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_source_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
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
    
    return fixed_json, generated_json

def infer_mistral(model, tokenizer, job_text, max_length=1536):
    """Run inference with a Mistral model"""
    # Format messages following Mistral's chat format
    messages = [
        {"role": "user", "content": (
            f"Label the following job posting in pure JSON format based on this example schema. "
            f"If no information for a field, leave the field blank.\n\n"
            f"Example schema:\n"
            f'{{\n    "experience_level": "",\n    "employment_status": [],\n    "work_location": "",\n    '
            f'"salary": {{"min": "", "max": "", "period": "", "currency": ""}},\n    "benefits": [],\n    '
            f'"job_functions": [],\n    "required_skills": {{\n        "programming_languages": [],\n        '
            f'"tools": [],\n        "frameworks": [],\n        "databases": [],\n        "other": []\n    }},\n    '
            f'"required_certifications": [],\n    "required_minimum_degree": "",\n    "required_experience": "",\n    '
            f'"industries": [],\n    "additional_keywords": []\n}}\n\n'
            f"Job posting:\n{job_text}"
        )}
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
    
    # Extract JSON from the response
    json_blocks = []
    brace_count = 0
    start_index = -1
    
    # Find JSON-like sections by tracking braces
    for j, char in enumerate(generated_json):
        if char == '{':
            if brace_count == 0:
                start_index = j
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_index != -1:
                # Found a potential JSON block
                json_block = generated_json[start_index:j+1]
                json_blocks.append(json_block)
    
    # Try each block, preferring the last one
    valid_json = None
    for block in reversed(json_blocks):
        try:
            # Check if this looks like our expected schema
            if '"experience_level"' in block:
                json.loads(block)
                valid_json = block
                break
        except json.JSONDecodeError:
            # Try to clean up common issues with the JSON
            try:
                # Fix trailing/malformed characters
                clean_block = block.rstrip('}]')
                if clean_block[-1] != '}':
                    clean_block += '}'
                json.loads(clean_block)
                valid_json = clean_block
                break
            except:
                continue
    
    if valid_json:
        fixed_json = fix_json_format(valid_json)
        return fixed_json, generated_json
    else:
        return "{}", generated_json

def main():
    parser = argparse.ArgumentParser(description="Job posting information extraction with different models")
    parser.add_argument("--model", type=str, default="flan_t5_large", 
                        choices=["flan_t5_large", "flan_t5_xl_batch_2", "flan_t5_xl_batch_4", "mistral_lora"],
                        help="Model to use for inference")
    parser.add_argument("--job_text", type=str, help="Job posting text (if not provided, will use examples)")
    parser.add_argument("--job_file", type=str, help="File containing job posting text")
    args = parser.parse_args()
    
    # Model paths mapping
    model_paths = {
        "flan_t5_large": "alexlanxy/flan_t5_large_linkedin_prompt_ordered_fields",
        "flan_t5_xl_batch_2": "alexlanxy/flan_t5_xl_lora_prompt_bf16_batch_2",
        "flan_t5_xl_batch_4": "alexlanxy/flan_t5_xl_lora_prompt_bf16_batch_4",
        "mistral_lora": "alexlanxy/mistral_7b_lora_batch_1_single_server"
    }
    
    # Load appropriate model
    model_name = model_paths[args.model]
    
    if "mistral" in args.model:
        model, tokenizer = load_mistral_model(model_name)
    else:
        model, tokenizer = load_flan_t5_model(model_name)
    
    # Get job text from argument, file, or use examples
    if args.job_text:
        job_texts = [args.job_text]
    elif args.job_file:
        with open(args.job_file, 'r') as f:
            job_texts = [f.read()]
    else:
        # Example job postings
        job_texts = [
            """
            About the job
About Prophecy 



Prophecy is the data copilot company. Fortune 500 enterprises - including the largest institutions in banking, insurance, healthcare & life sciences, and technology - rely on Prophecy Data Transformation Copilot to accelerate AI and analytics by delivering data that is clean, trusted, and timely. Prophecy enables all data users and makes them productive by helping develop, deploy, and observe data pipelines on cloud data platforms. Organizations trust Prophecy for the most demanding workloads, including tens of thousands of data pipelines that deliver massive volumes of data for AI and analytics.
Prophecy is backed by top VCs and investors including Insight Partners, SignalFire, Databricks, and JPMorgan Chase.


The AI Research Engineer will:

Provide hands-on technical leadership to design, develop, optimize and deploy Generative AI features, leveraging expertise in LLMs, RAGs, NLP, and AI infrastructure to optimize model quality and performance.
Lead a strong technical team across the US and Bangalore, guiding them on state-of-the-art AI models and algorithms while establishing best data-driven development practices.
Build agents that streamline data extraction, transformations and analysis.
Define scalable evaluation strategies that balance AI with human reviewers.


What You'll Need

Track record of driving AI products from concept to customer adoption.
Hands-on expertise with GenAI, LLMs, NLP, RAG, and Knowledge Graphs across the full ML lifecycle.
Mastery of SOTA techniques to boost LLM accuracy and reliability, including fine-tuning foundation models and curating datasets.
Experience in one or more: code generation (e.g., Codex, text-to-SQL), semantic extraction systems, knowledge graphs (e.g., Neo4j), or real-time ML products.
Passion for rapid iteration and high-quality execution with an entrepreneurial edge.
Up-to-date with AI trends; bonus for publications (e.g., NeurIPS, ACL).
BS in CS or equivalent, 5-10 years of experience; MS/PhD preferred.


Benefits and Perks 

Prophecy covers 99% of employee health insurance and 75% for dependents
Very competitive compensation
We offer $200 per month towards wellness, gyms, massages, facials, and more!
Celebrate your birthday and anniversary with a day off!
Flexible PTO
Prophecy provides employees with a $500 professional development reimbursement every year
Company sponsored Long Term Disability and Life Insurance
FSA/HSA
Ability to have your fingerprint on an innovative platform
End-to-end ownership of your projects
And more!


*Benefits and perks may vary per country



Our Commitment to Diversity and Inclusion



At Prophecy, we hire for merit and foster an inclusive culture where people from diverse backgrounds can excel and do their best work. We take great care to ensure that our hiring practices are inclusive and meet equal employment opportunity standards. Individuals looking for employment at Prophecy are considered without regard to age, color, disability, ethnicity, family or marital status, gender identity or expression, language, national origin, physical and mental ability, political affiliation, race, religion, sexual orientation, socio-economic status, veteran status, and any other protected characteristics under applicable laws.
            """
        ]
    
    # Run inference on each job text
    for i, job_text in enumerate(job_texts):
        print(f"\n{'='*50}\nProcessing Job Text #{i+1}\n{'='*50}")
        print(f"Job Text (truncated): {job_text[:150]}...")
        
        if "mistral" in args.model:
            fixed_json, raw_json = infer_mistral(model, tokenizer, job_text)
        else:
            fixed_json, raw_json = infer_flan_t5(model, tokenizer, job_text)
        
        # Print results
        print(f"\n--- Raw Model Output ---\n{raw_json[:300]}..." if len(raw_json) > 300 else raw_json)
        print(f"\n--- Parsed and Fixed JSON ---\n{fixed_json}")
        
        # Check JSON validity
        try:
            parsed_json = json.loads(fixed_json)
            print("\n--- Extracted Key Information ---")
            
            # Print experience level if available
            if parsed_json.get("experience_level"):
                print(f"Experience Level: {parsed_json['experience_level']}")
            
            # Print salary range if available
            salary = parsed_json.get("salary", {})
            if salary.get("min") or salary.get("max"):
                print(f"Salary Range: {salary.get('min', '')} - {salary.get('max', '')} {salary.get('currency', '')} per {salary.get('period', '')}")
            
            # Print work location if available
            if parsed_json.get("work_location"):
                print(f"Location: {parsed_json['work_location']}")
            
            # Print required skills
            skills = parsed_json.get("required_skills", {})
            all_skills = []
            for skill_type, skill_list in skills.items():
                all_skills.extend(skill_list)
            
            if all_skills:
                print(f"Required Skills: {', '.join(all_skills[:10])}" + (" and more..." if len(all_skills) > 10 else ""))
            
            # Print benefits if available
            benefits = parsed_json.get("benefits", [])
            if benefits:
                print(f"Benefits: {', '.join(benefits[:5])}" + (" and more..." if len(benefits) > 5 else ""))
                
            print(f"\nValidation: Successfully parsed as valid JSON")
        except json.JSONDecodeError:
            print(f"\nValidation: Failed to parse as valid JSON")

if __name__ == "__main__":
    main() 