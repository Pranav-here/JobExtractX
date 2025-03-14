import logging
from config_setup import setup_deepseek, setup_mongodb
import json
from datetime import datetime, timezone
from tqdm import tqdm
import os

# Configure logging
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), 'job_labeling.log'),  # Log file name
    level=logging.DEBUG,          # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

def label_job_data(job_data):
    # Request DeepSeek to label the data
    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful AI trained to label job posting data."},
        {"role": "user", "content": f"Label the following job posting with only json format output as per the example:{example_labeled_data}, no comment or explanation, just output the json format. if no information found, just leave it blank.\n raw job data:{job_data}"}
    ],
    stream=False
    )

    # Simulated response content
    response_content = response.choices[0].message.content

    # Remove markdown code block delimiters
    if response_content.startswith("```json"):
        response_content = response_content[7:]  # Remove the starting ```json
    if response_content.endswith("```"):
        response_content = response_content[:-3]  # Remove the ending ```

    # Parse the JSON
    try:
        label_data = json.loads(response_content)
    except json.JSONDecodeError as e:
        error_message = f"Error decoding JSON for job data: {job_data}, response: {response_content}, error: {e}"
        print(error_message)
        logging.error(error_message)  # Log the error
        label_data = None
    return label_data

example_labeled_data = """
{
    "experience_level": "",  // e.g., "Entry-level", "Mid-level", etc.
    "employment_status": [], // e.g., ["Contract", "Permanent", "Freelance", "Part-time", etc.]
    "work_location": "",    // e.g., "Remote", "Hybrid", "On-site", etc.
    "salary": {
        "min": "",           // e.g., "60000"
        "max": "",           // e.g., "80000"
        "period": "",        // e.g., "hour", "month", etc.
        "currency": ""       // e.g., "USD", etc.
    },
    "benefits": [],
    "job_functions": [],        // e.g., ["Backend", "Full Stack", etc.]
    "required_skills": {
        "programming_languages": [],  // e.g., ["Python", "Java", etc.]
        "tools": [],                  // e.g., ["Git", "Docker", etc.]
        "frameworks": [],             // e.g., ["Django", "React", etc.]
        "databases": [],              // e.g., ["MongoDB", "PostgreSQL", etc.]
        "other": []                   // e.g., ["Cloud Services", etc.]
    },
    "required_certifications": [],
    "required_minimum_degree": "",   // e.g., "Bachelor's", "Master's", "PhD"
    "required_experience": "",      // e.g., "1 year", "2 years", "3 years", etc.
    "industries": []           // e.g., ["Software Development", "Healthcare", etc.]
    "additional_keywords": []  // not mentioned above
}
"""


client = setup_deepseek()
db = setup_mongodb()

labeled_job_collection = db["labeled_job_data"]

# Define the aggregation pipeline
pipeline = [
    {
        "$lookup": {
            "from": "labeled_job_data",  
            "localField": "_id",            
            "foreignField": "_id",       
            "as": "labeled_info"         
        }
    },
    {
        "$match": {
            "labeled_info": {"$eq": []}  
        }
    }
]

# Execute the pipeline
to_label_job_data = db["training_preparation"].aggregate(pipeline)

# Count the total jobs to label
total_jobs = db["training_preparation"].aggregate([
    {
        "$lookup": {
            "from": "labeled_job_data",
            "localField": "_id",
            "foreignField": "_id",
            "as": "labeled_info"
        }
    },
    {
        "$match": {
            "labeled_info": {"$eq": []}
        }
    },
    {
        "$count": "total"
    }
])

total_jobs_count = list(total_jobs)[0]["total"] if total_jobs else 0
print(f"Total jobs to label: {total_jobs_count}")

for job in tqdm(to_label_job_data, desc="Labeling job data", total=total_jobs_count):
    id = job["_id"]
    
    # Re-check if the job has been labeled
    already_labeled = labeled_job_collection.find_one({"_id": id})
    
    if already_labeled:
        print(f"Job {id} already labeled, skipping.")
        continue
    
    # Remove unnecessary fields
    del job["_id"]
    del job["company_id"]
    
    # Label the job
    labeled_job = label_job_data(job)
    
    if labeled_job:
        # Use update_one with $set and $setOnInsert
        labeled_job["updated_at"] = datetime.now(timezone.utc)
        labeled_job_collection.update_one(
            {"_id": id},  # Filter to find the document
            {
                "$set": labeled_job,
                "$setOnInsert": {
                    "created_at": datetime.now(timezone.utc)
                }
            },
            upsert=True  # Insert the document if it doesn't exist
        )



