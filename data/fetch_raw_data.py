from tqdm import tqdm
from config_setup import setup_linkedin_api, setup_mongodb
from datetime import datetime, timezone
from linkedin_utils import extract_company_id
from update_raw_company import update_company_info
import argparse


api = setup_linkedin_api()
db = setup_mongodb()

# Collections
jobs_collection = db["job_raw"]
companies_collection = db["company_raw"]
job_details_collection = db["job_detail_raw"]

# Set up argument parser
parser = argparse.ArgumentParser(description="Fetch job listings and store them in MongoDB.")
parser.add_argument('-s', '--skip-existing', action='store_true', help="Skip job IDs that already exist in job_raw.")
args = parser.parse_args()

# Fetch job listings
print("Fetching job listings", flush=True)
job_listings = api.search_jobs('Software Engineer')
print(f"Found {len(job_listings)} job listings")

# Determine existing job IDs in the database
existing_job_ids = set(jobs_collection.distinct("_id"))

# Create a reduced list of job listings to process
if args.skip_existing:
    reduced_job_listings = [
        job for job in job_listings
        if job["entityUrn"].split(":")[-1] not in existing_job_ids
    ]
else:
    reduced_job_listings = job_listings

print(f"Will process {len(reduced_job_listings)} new job listings.")


# Process each job listing with a progress bar
for job in tqdm(reduced_job_listings, desc="Processing job listings"):
    # Extract job ID
    job_id = job["entityUrn"].split(":")[-1]

    # Add timestamps
    job['updated_at'] = datetime.now(timezone.utc)

    # Insert job data into job_raw collection with job_id as _id
    jobs_collection.update_one(
        {"_id": job_id},
        {"$set": job, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
        upsert=True
    )

    # Fetch detailed job information
    job_detail = api.get_job(job_id)

    # Extract company ID using the utility function
    company_id = extract_company_id(job_detail)

    # Add company_id to job_detail
    job_detail['company_id'] = company_id

    # Add timestamps
    job_detail['updated_at'] = datetime.now(timezone.utc)

    # Insert job detail data into job_detail_raw collection with job_id as _id
    job_details_collection.update_one(
        {"_id": job_id},
        {"$set": job_detail, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
        upsert=True
    )

    if company_id:
        # Proceed with using company_id
        update_company_info(company_id)
    else:
        print("No valid company information found for job.") 
        print(job_detail)
print("🎉 Job, job detail, and company data have been saved to MongoDB.")