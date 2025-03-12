from config_setup import setup_config
from linkedin_utils import extract_company_id
from tqdm import tqdm

config, api, client, db = setup_config()

jobs_collection = db["job_detail_raw"]

# Fetch all job entries from job_raw collection
job_entries = jobs_collection.find()

# Process each job entry
for job in tqdm(job_entries, desc="Processing job entries"):
    # Use the utility function to update company info
    company_id = extract_company_id(job)
    jobs_collection.update_one(
            {"_id": job["_id"]},
            {"$set": {"company_id": company_id}}
        )