from tqdm import tqdm
from config_setup import setup_linkedin_api, setup_mongodb
from datetime import datetime, timezone
from linkedin_utils import extract_company_id
from update_raw_company import update_company_info
import argparse
import time
from requests.exceptions import ConnectionError, RequestException


api = setup_linkedin_api()
db = setup_mongodb()

# Collections
jobs_collection = db["job_raw"]
companies_collection = db["company_raw"]
job_details_collection = db["job_detail_raw"]

# Set up argument parser
parser = argparse.ArgumentParser(description="Fetch job listings and store them in MongoDB.")
parser.add_argument('-a', '--all', action='store_true', help="Process all job IDs, including those that already exist in job_raw.")
args = parser.parse_args()

# Fetch job listings
print("Fetching job listings", flush=True)
job_listings = api.search_jobs('Software Engineer')
print(f"Found {len(job_listings)} job listings")

# Determine existing job IDs in the database
existing_job_ids = set(jobs_collection.distinct("_id"))

# Create a reduced list of job listings to process
if args.all:
    reduced_job_listings = job_listings
else:
    reduced_job_listings = [
        job for job in job_listings
        if job["entityUrn"].split(":")[-1] not in existing_job_ids
    ]

print(f"Will process {len(reduced_job_listings)} new job listings.")


# Process each job listing with a progress bar
for job in tqdm(reduced_job_listings, desc="Processing job listings"):
    try:
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

        # # Add delay between requests to avoid rate limiting
        # time.sleep(0.5)  

        # Fetch detailed job information with retry logic
        max_retries = 3
        retry_delay = 3  # seconds
        
        for attempt in range(max_retries):
            try:
                job_detail = api.get_job(job_id)
                break
            except (ConnectionError, RequestException) as e:
                if attempt == max_retries - 1:  # Last attempt
                    print(f"Failed to fetch job details for job_id {job_id} after {max_retries} attempts: {str(e)}")
                    continue
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

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
            # # Add delay before company request
            # time.sleep(0.5)  
            
            # Retry logic for company info
            for attempt in range(max_retries):
                try:
                    update_company_info(company_id)
                    break
                except (ConnectionError, RequestException) as e:
                    if attempt == max_retries - 1:  # Last attempt
                        print(f"Failed to fetch company info for company_id {company_id} after {max_retries} attempts: {str(e)}")
                        continue
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
        else:
            print(f"No valid company information found for job {job_id}")

    except Exception as e:
        print(f"Error processing job {job_id}: {str(e)}")
        continue  # Continue with next job even if this one fails

print("🎉 Job, job detail, and company data have been saved to MongoDB.")