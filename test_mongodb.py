import sys
import os
import time

# Add parent directory to path to find modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.config_setup import setup_mongodb
from model.data_preprocessing import prepare_data

print("Attempting to connect to MongoDB...")
try:
    db = setup_mongodb()
    print("Connected to MongoDB!")
    
    # Try to access the collections used in evaluate_mistral_lora.py
    training_collection = db["training_preparation"]
    labeled_job_collection = db["labeled_job_data"]
    
    # Count records
    training_count = training_collection.count_documents({})
    labeled_count = labeled_job_collection.count_documents({})
    
    print(f"Found {training_count} documents in training_preparation collection")
    print(f"Found {labeled_count} documents in labeled_job_data collection")
    
    # Test prepare_data function
    print("\nTesting prepare_data function...")
    start_time = time.time()
    print("Starting data preparation at:", start_time)
    
    # Get limited data pairs
    data_pairs = prepare_data(prompt=True)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Data preparation completed in {elapsed:.2f} seconds")
    print(f"Got {len(data_pairs)} data pairs")
    
    # Print first data pair as sample
    if data_pairs:
        print("\nSample data pair (first 200 chars of input and output):")
        input_text, output_text = data_pairs[0]
        print(f"Input: {input_text[:200]}...")
        print(f"Output: {output_text[:200]}...")
    
except Exception as e:
    print(f"Error: {str(e)}") 