# Use a pipeline as a high-level helper
from transformers import pipeline
from data_preprocessing import prepare_data
import json
import os
import re
from datetime import datetime

# Create results directory if it doesn't exist
results_dir = "evaluation_results"
os.makedirs(results_dir, exist_ok=True)

# Initialize the pipeline with a much longer max_length
pipe = pipeline("text2text-generation", model="alexlanxy/t5-Apr-8", max_length=1024)

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("alexlanxy/t5-Apr-8")
model = AutoModelForSeq2SeqLM.from_pretrained("alexlanxy/t5-Apr-8")

# Load data
print("Loading data...")
data_pairs = prepare_data()
print(f"Loaded {len(data_pairs)} data pairs")

# Use only the last 10% of data for evaluation (ensuring it's different from training data)
eval_start_idx = int(len(data_pairs) * 0.9)
evaluation_data = data_pairs[eval_start_idx:]
print(f"Using {len(evaluation_data)} examples from the last 10% of data for evaluation")

# Create a timestamp for this evaluation run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = os.path.join(results_dir, f"evaluation_results_{timestamp}.json")

# Evaluation results
results = []

# Process a subset for example (adjust as needed)
num_examples = min(10, len(evaluation_data))
print(f"\nGenerating outputs for {num_examples} examples...")

for i, pair in enumerate(evaluation_data[:num_examples]):
    print(f"\nExample {i+1}/{num_examples}")
    input_text = pair[0]
    expected_output = pair[1]
    
    # Truncate input display for readability
    display_input = input_text[:500] + "..." if len(input_text) > 500 else input_text
    print(f"\nINPUT:\n{display_input}")
    
    # Expected output (truncated for display)
    display_expected = expected_output[:500] + "..." if len(expected_output) > 500 else expected_output
    print(f"\nEXPECTED OUTPUT:\n{display_expected}")
    
    # Generate output
    model_output = pipe(input_text)
    generated_text = model_output[0]["generated_text"]
    
    # Display generated output
    display_generated = generated_text[:500] + "..." if len(generated_text) > 500 else generated_text
    print(f"\nGENERATED OUTPUT:\n{display_generated}")
    
    # Try to parse both outputs as JSON if possible
    try:
        expected_json = json.loads(expected_output)
        try:
            generated_json = json.loads(generated_text)
            # Could calculate field-by-field accuracy here
            json_match = "Both outputs are valid JSON"
        except:
            json_match = "Expected output is JSON, generated output is not"
    except:
        json_match = "Expected output is not JSON"
    print('JSON Match: ', json_match)
    # Store result
    results.append({
        "example_id": i,
        "input": input_text,
        "expected_output": expected_output,
        "generated_output": generated_text,
        "json_match": json_match
    })

# Save results
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nEvaluation complete. Results saved to {results_file}")

