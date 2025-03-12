"""
This script fetches job listings and company information from LinkedIn using the unofficial LinkedIn API.
The data is then stored in MongoDB collections named 'job_raw', 'job_detail_raw', and 'company_raw'.

Configuration:
- LinkedIn credentials are stored in 'config.json'.
- MongoDB connection details are stored in 'config.json'.

Main Steps:
1. Load LinkedIn credentials and MongoDB URI from 'config.json'.
2. Initialize the LinkedIn API and MongoDB client.
3. Fetch job listings for a specified role (e.g., 'Software Engineer').
4. Process each job listing, extracting job and company details.
5. Store the extracted data in MongoDB with progress tracking using tqdm.

Note: Ensure that 'config.json' is present in the same directory with valid LinkedIn credentials and MongoDB URI.
"""

# Import necessary libraries
from linkedin_api import Linkedin
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from tqdm import tqdm
from config_loader import load_config
from datetime import datetime

# Load configuration from config file
config = load_config()

# Initialize LinkedIn API
api = Linkedin(config["linkedin_email"], config["linkedin_password"])

# Initialize MongoDB client
client = MongoClient(config["mongodb_uri"], server_api=ServerApi('1'))
db = client["myDatabase"]

# Collections
jobs_collection = db["job_raw"]
companies_collection = db["company_raw"]
job_details_collection = db["job_detail_raw"]

# Fetch job listings
print("Fetching job listings")
job_listings = api.search_jobs('Software Engineer')
print(f"Found {len(job_listings)} job listings")

# Process each job listing with a progress bar
for job in tqdm(job_listings, desc="Processing job listings"):
    # Extract job ID
    job_id = job["entityUrn"].split(":")[-1]

    # Add timestamps
    job['updated_at'] = datetime.utcnow()

    # Insert job data into job_raw collection with job_id as _id
    jobs_collection.update_one(
        {"_id": job_id},
        {"$set": job, "$setOnInsert": {"created_at": datetime.utcnow()}},
        upsert=True
    )

    # Fetch detailed job information
    job_detail = api.get_job(job_id)

    # Add timestamps
    job_detail['updated_at'] = datetime.utcnow()

    # Insert job detail data into job_detail_raw collection with job_id as _id
    job_details_collection.update_one(
        {"_id": job_id},
        {"$set": job_detail, "$setOnInsert": {"created_at": datetime.utcnow()}},
        upsert=True
    )

    # Extract company information from job details
    company_info = job_detail.get("companyDetails", {}).get("companyResolutionResult", {})
    company_id = company_info.get("entityUrn", "").split(":")[-1]

    # Add timestamps
    company_info['updated_at'] = datetime.utcnow()

    # Insert company data into company_raw collection with company_id as _id
    companies_collection.update_one(
        {"_id": company_id},
        {"$set": company_info, "$setOnInsert": {"created_at": datetime.utcnow()}},
        upsert=True
    )

print("🎉 Job, job detail, and company data have been saved to MongoDB.")