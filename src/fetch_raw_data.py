"""
This script fetches job listings and company information from LinkedIn using the unofficial LinkedIn API.
The data is then stored in MongoDB collections named 'job_raw' and 'company_raw'.

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
import json
from linkedin_api import Linkedin
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from tqdm import tqdm

# Load configuration from config file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Initialize LinkedIn API
api = Linkedin(config["linkedin_email"], config["linkedin_password"])

# Initialize MongoDB client
client = MongoClient(config["mongodb_uri"], server_api=ServerApi('1'))
db = client["myDatabase"]

# Collections
jobs_collection = db["job_raw"]
companies_collection = db["company_raw"]

# Fetch job listings
print("Fetching job listings")
job_listings = api.search_jobs('Software Engineer')
print(f"Found {len(job_listings)} job listings")

# Process each job listing with a progress bar
for job in tqdm(job_listings, desc="Processing job listings"):
    # Extract job ID and company information
    job_id = job["entityUrn"].split(":")[-1]
    company_info = job.get("companyDetails", {}).get("companyResolutionResult", {})

    # Insert job data into MongoDB
    jobs_collection.update_one(
        {"job_id": job_id},
        {"$set": job},
        upsert=True
    )

    # Insert company data into MongoDB
    company_id = company_info.get("entityUrn", "").split(":")[-1]
    companies_collection.update_one(
        {"company_id": company_id},
        {"$set": company_info},
        upsert=True
    )

print("🎉 Job and company data have been saved to MongoDB.")