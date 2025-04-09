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
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter
    
    try:
        nltk.download('punkt')
    except:
        print("Could not download punkt, will try to use existing installation")
    
    from nltk.tokenize import word_tokenize
    
    data_pairs = prepare_data()
    
    # Collect token counts
    input_token_counts = []
    output_token_counts = []
    
    for pair in data_pairs:
        input_tokens = word_tokenize(pair[0])
        output_tokens = word_tokenize(pair[1])
        
        input_token_counts.append(len(input_tokens))
        output_token_counts.append(len(output_tokens))
    
    # Calculate statistics
    print(f"Total data pairs: {len(data_pairs)}")
    
    # Input token statistics
    print("\nORIGINAL DATA TOKEN STATISTICS:")
    print(f"Min tokens: {min(input_token_counts)}")
    print(f"Max tokens: {max(input_token_counts)}")
    print(f"Average tokens: {np.mean(input_token_counts):.2f}")
    print(f"Median tokens: {np.median(input_token_counts)}")
    print(f"25th percentile: {np.percentile(input_token_counts, 25)}")
    print(f"75th percentile: {np.percentile(input_token_counts, 75)}")
    
    # Label token statistics
    print("\nLABEL DATA TOKEN STATISTICS:")
    print(f"Min tokens: {min(output_token_counts)}")
    print(f"Max tokens: {max(output_token_counts)}")
    print(f"Average tokens: {np.mean(output_token_counts):.2f}")
    print(f"Median tokens: {np.median(output_token_counts)}")
    print(f"25th percentile: {np.percentile(output_token_counts, 25)}")
    print(f"75th percentile: {np.percentile(output_token_counts, 75)}")
    
    # Plot histograms
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(input_token_counts, bins=30, alpha=0.7, color='blue')
    plt.title('Original Data Token Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(output_token_counts, bins=30, alpha=0.7, color='green')
    plt.title('Label Data Token Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('token_distribution.png')
    print("\nDistribution histogram saved as 'token_distribution.png'")






