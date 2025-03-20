import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.config_setup import setup_mongodb
import json


# Connect to MongoDB
db = setup_mongodb()

# Collections
training_collection = db["training_preparation"]
labeled_job_collection = db["labeled_job_data"]


def prepare_data():
    # Retrieve data from MongoDB
    original_data = list(training_collection.find())
    labeled_data = list(labeled_job_collection.find())

    # Create a dictionary to map _id to job data
    original_data_dict = {job['_id']: job for job in original_data}

    # Prepare input-output pairs
    input_output_pairs = []
    for label in labeled_data:
        job_id = label['_id']
        job = original_data_dict.get(job_id)
        
        if job:
            del job['_id']
            del job['company_id']


            input_text = f"{job}"
            
            # Convert labeled data to JSON string
            del label['_id']
            del label['created_at']
            del label['updated_at']
            output_text = json.dumps(label)
            
            # Append the pair
            input_output_pairs.append((input_text, output_text))
    
    return input_output_pairs

# Example usage
if __name__ == "__main__":
    import nltk
    nltk.download('punkt_tab')
    from nltk.tokenize import word_tokenize
    data_pairs = prepare_data()
    for pair in data_pairs[:15]:
        
        tokens_0 = word_tokenize(pair[0])
        tokens_1 = word_tokenize(pair[1])

        print(f"Number of tokens in pair[0]: {len(tokens_0)}")
        print(f"Number of tokens in pair[1]: {len(tokens_1)}")
        print("-"*10)

        # print(pair[0])
        # print("-"*10)
        # print("-"*10)
        # print(pair[1])






