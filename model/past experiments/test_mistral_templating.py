import torch
from transformers import AutoTokenizer
import os

# Set your Hugging Face token as an environment variable
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

# Initialize the tokenizer
print("Loading tokenizer...")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Example data - keeping it very short
source_text = "{'title': 'Software Engineer'}"
target_text = '{"job_functions": ["Software Engineering"]}'

# Format messages following Mistral's chat format
messages = [
    {"role": "user", "content": f"Label this job: {source_text}"}
]

# Use tokenizer to encode and handle message formatting
print("\n--- Applying Chat Template ---")
chat_input = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True,
    return_tensors=None
)

print(f"Chat input type: {type(chat_input)}")
if isinstance(chat_input, list):
    chat_input_text = tokenizer.decode(chat_input)
else:
    chat_input_text = chat_input

print(f"Chat input text (truncated): {chat_input_text[:100]}...")

# Check if <|assistant|> is in the chat template output
print(f"\nDoes chat input contain '<|assistant|>'? {'<|assistant|>' in chat_input_text}")
assistant_position = chat_input_text.find('<|assistant|>')
print(f"Position of '<|assistant|>': {assistant_position}")

# Combine with target as in training
full_text = chat_input_text + target_text + tokenizer.eos_token
print(f"\nFull text (truncated): {full_text[:200]}...")

# Tokenize inputs - short max_length to avoid large output
encoding = tokenizer(
    full_text,
    max_length=256,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

input_ids = encoding.input_ids.squeeze()
labels = input_ids.clone()

# Find where the user message ends and assistant response begins
assistant_start = full_text.find("<|assistant|>")
if assistant_start != -1:
    # Find this position in the tokenized input
    assistant_token_pos = len(tokenizer.encode(full_text[:assistant_start]))
    
    # Mask everything before the assistant token with -100
    labels[:assistant_token_pos] = -100
    
    print(f"Found '<|assistant|>' at position {assistant_start}")
    print(f"Assistant token position in tokenized input: {assistant_token_pos}")
else:
    print("WARNING: '<|assistant|>' not found in the text!")

# Also mask padding tokens
labels[labels == tokenizer.pad_token_id] = -100

# Check label masking
non_masked = (labels != -100).sum().item()
total = labels.numel()
print(f"\nLabels check: {non_masked}/{total} valid labels ({non_masked/total*100:.2f}%)")

# Show what the model is learning to predict
valid_indices = (labels != -100).nonzero(as_tuple=True)[0]
if len(valid_indices) > 0:
    start_idx = valid_indices[0].item()
    print(f"First valid label at position: {start_idx}")
    
    # What is the model actually learning?
    learning_to_predict = tokenizer.decode(input_ids[start_idx:])
    print(f"\nText the model is learning to predict: {learning_to_predict[:150]}...")
else:
    print("\nNo valid labels found! The model won't learn anything from this example.") 