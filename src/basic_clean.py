from pymongo import MongoClient, ASCENDING
from config_loader import get_config
from tqdm import tqdm

# Load configuration
config = get_config()

# Initialize MongoDB client
client = MongoClient(config["mongodb_uri"])
db = client["myDatabase"]

# Collections
job_raw_collection = db["job_raw"]
basic_clean_collection = db["basic_clean"]
companies_collection = db["company_raw"]

# Ensure job_id is unique in basic_clean collection
basic_clean_collection.create_index([("job_id", ASCENDING)], unique=True)

def basic_clean_pipeline(job_data, company_data):
    # Example cleaning logic: combine job and company data
    combined_data = {**job_data, **company_data}
    # Add more cleaning steps as needed
    return combined_data

# Fetch all job IDs from job_raw collection
job_ids = job_raw_collection.distinct("job_id")

# Add a progress bar to the loop
for job_id in tqdm(job_ids, desc="Processing job IDs"):
    # Check if the job ID exists in the basic_clean collection
    if not basic_clean_collection.find_one({"job_id": job_id}):
        # Fetch job and company data
        job_data = job_raw_collection.find_one({"job_id": job_id})
        company_id = job_data.get("companyDetails", {}).get("companyResolutionResult", {}).get("entityUrn", "").split(":")[-1]
        company_data = companies_collection.find_one({"company_id": company_id})

        # Run the basic clean pipeline
        cleaned_data = basic_clean_pipeline(job_data, company_data)

        # Remove the _id field to avoid duplicate key error
        cleaned_data.pop('_id', None)

        # Insert cleaned data into basic_clean collection
        basic_clean_collection.insert_one(cleaned_data)

print("Basic cleaning process completed.")