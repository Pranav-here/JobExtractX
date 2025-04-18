import sys
import os
import json
from datetime import datetime
from collections import OrderedDict

# Add parent directory to path to find modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data.config_setup import setup_mongodb

def prepare_data_after_date(date_str="2025-04-18T00:00:00.000Z"):
    """
    Fetch data from MongoDB created after the specified date and prepare it for training.
    
    Args:
        date_str: ISO format date string to filter data by created_at
    
    Returns:
        List of input-output pairs
    """
    # Connect to MongoDB
    db = setup_mongodb()
    
    # Collections
    training_collection = db["training_preparation"]
    labeled_job_collection = db["labeled_job_data"]
    
    # Create date object for comparison
    filter_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    
    # Retrieve data from MongoDB with date filter
    original_data = list(training_collection.find())
    labeled_data = list(labeled_job_collection.find({
        "created_at": {"$gt": filter_date}
    }))
    
    print(f"Found {len(labeled_data)} new labeled documents created after {date_str}")
    
    # Create a dictionary to map _id to job data
    original_data_dict = {job['_id']: job for job in original_data}
    
    # Prepare input-output pairs
    input_output_pairs = []
    for label in labeled_data:
        job_id = label['_id']
        job = original_data_dict.get(job_id)
        
        if job:
            job_copy = job.copy()
            del job_copy['_id']
            del job_copy['company_id']

            input_text = f"{job_copy}"
            
            # Make a copy of the label to avoid modifying the original
            label_copy = label.copy()
            
            # Remove fields not needed in output
            del label_copy['_id']
            del label_copy['created_at']
            del label_copy['updated_at']
            
            # Create ordered dictionary with specified field order
            ordered_label = OrderedDict([
                ("experience_level", label_copy.get("experience_level", "")),
                ("employment_status", label_copy.get("employment_status", [])),
                ("work_location", label_copy.get("work_location", "")),
                ("salary", label_copy.get("salary", {"min": "", "max": "", "period": "", "currency": ""})),
                ("benefits", label_copy.get("benefits", [])),
                ("job_functions", label_copy.get("job_functions", [])),
                ("required_skills", label_copy.get("required_skills", {
                    "programming_languages": [],
                    "tools": [],
                    "frameworks": [],
                    "databases": [],
                    "other": []
                })),
                ("required_certifications", label_copy.get("required_certifications", [])),
                ("required_minimum_degree", label_copy.get("required_minimum_degree", "")),
                ("required_experience", label_copy.get("required_experience", "")),
                ("industries", label_copy.get("industries", [])),
                ("additional_keywords", label_copy.get("additional_keywords", []))
            ])
            
            # Convert ordered label to JSON string
            output_text = json.dumps(ordered_label)
            
            # Append the pair
            input_output_pairs.append((input_text, output_text))
    
    return input_output_pairs

if __name__ == "__main__":
    # Define the date to filter by (April 18, 2025 at 00:00:00)
    filter_date = "2025-04-18T00:00:00.000Z"
    
    # Define output file path
    output_file = "prepared_data.json"
    
    # Get data pairs
    print(f"Fetching data created after {filter_date}...")
    data_pairs = prepare_data_after_date(filter_date)
    
    # Convert all data to consistent dictionary format
    formatted_data = []
    for item in data_pairs:
        if isinstance(item, (tuple, list)):
            formatted_data.append({"source": item[0], "target": item[1]})
        elif isinstance(item, dict) and "source" in item and "target" in item:
            formatted_data.append(item)  # Already in the right format
        else:
            raise TypeError(f"Unsupported data format: {type(item)}. Must be tuple/list with 2 elements or dict with 'source'/'target' keys")
    
    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(formatted_data, f)
    
    print(f"✅ Data prepared and saved to {output_file}")
    print(f"Total data pairs: {len(formatted_data)}") 