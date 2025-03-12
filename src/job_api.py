from pymongo import MongoClient
from typing import Optional, List, Dict
import re
from config_loader import get_config

# Load configuration
config = get_config()

# MongoDB connection
client = MongoClient(config["mongodb_uri"])
db = client["myDatabase"]

jobs_collection = db["jobs"]
companies_collection = db["companies"]
for job in jobs_collection.find().limit(5):
    print(job)

def search_jobs(keyword: Optional[str] = None, location: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """ Refined version of search_jobs with improved filtering logic """
    query = []

    # Compile regex patterns
    if keyword:
        keyword_pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        query.append(lambda job: bool(keyword_pattern.search(job["title"])))

    if location:
        # Allow more flexible location matching (handles partial matches)
        location_pattern = re.compile(re.escape(location), re.IGNORECASE)
        query.append(lambda job: bool(location_pattern.search(job["location"])))

    # Apply filters
    filtered_jobs = [job for job in jobs_collection.find() if all(q(job) for q in query)]

    return filtered_jobs[:limit]


def get_job(job_id: int) -> Dict:
   # Retrieve a single job posting by job_id
    job = jobs_collection.find_one({"job_id": job_id})
    if job:
        job["_id"] = str(job["_id"])
        return job
    return {"error": "Job not found"}


def get_company(company_id: int) -> Dict:
    """ Retrieve company information by company_id """

    print(f"🔍 Searching for company_id: {company_id} (Type: {type(company_id)})")
    company = companies_collection.find_one({"company_id": company_id})
    print("Query result (as int):", company)

    if company:
        print("✅ Company found:", company)
        company["_id"] = str(company["_id"])  # Convert ObjectId to string
        return company

    return {"error": "Company not found"}

