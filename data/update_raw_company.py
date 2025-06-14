from config_setup import setup_linkedin_api, setup_mongodb
from datetime import datetime
from tqdm import tqdm

api, db = setup_linkedin_api(), setup_mongodb()
companies_collection = db["company_raw"]

def update_company_info(company_id):

    # Proceed with using company_id
    company_info = api.get_company(company_id)
    company_info['updated_at'] = datetime.utcnow()
    companies_collection.update_one(
        {"_id": company_id},
        {"$set": company_info, "$setOnInsert": {"created_at": datetime.utcnow()}},
        upsert=True
    )



if __name__ == "__main__":
    jobs_collection = db["job_detail_raw"]

    # Count the number of documents in the collection
    job_count = jobs_collection.count_documents({})
    print(f"Found {job_count} job entries to process")

    # Fetch all job entries from job_raw collection
    job_entries = jobs_collection.find()

    # Process each job entry
    for job in tqdm(job_entries, desc="Processing job entries"):
        # Use the utility function to update company info
        update_company_info(job["company_id"])



    print("🎉 Company data has been updated in MongoDB.")
