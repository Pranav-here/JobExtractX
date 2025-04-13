from data_preprocessing import prepare_data
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import Dataset
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import os

# Set your Hugging Face token as an environment variable
# Replace "your_huggingface_token" with your actual token from https://huggingface.co/settings/tokens
# You need to accept the model's terms of use on the Hugging Face website first
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"  # Replace with your token

data_pairs = prepare_data()
print("data set size: ", len(data_pairs))

# Print the structure of the first data item to understand its format
if len(data_pairs) > 0:
    print("Sample data structure:", type(data_pairs[0]))
    print("First data item:", data_pairs[0])

# Initialize tokenizer and model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Setup quantization config (load in 8-bit to save memory)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# Load model in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"  # Automatically places on correct GPU
)

# Prepare for LoRA training
model = prepare_model_for_kbit_training(model)

# Define LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # For Mistral, target attention projection layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Confirm only LoRA layers are trainable

# Dataset class
class MistralDataset(Dataset):
    def __init__(self, data_pairs, tokenizer, max_length=1024):
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, index):
        # Assuming data_pairs contains tuples or dicts
        data_item = self.data_pairs[index]
        
        # Check if the data is a tuple or a dict and handle accordingly
        if isinstance(data_item, tuple):
            # Assuming (source, target) format
            source_text = data_item[0]
            target_text = data_item[1]
        elif isinstance(data_item, dict):
            # Handle dictionary format
            source_text = data_item["source"]
            target_text = data_item["target"]
        else:
            raise TypeError(f"Unsupported data type: {type(data_item)}")
        
        # Ensure source and target are not empty
        if not source_text or not target_text:
            raise ValueError(f"Empty source or target text found at index {index}")
            
        schema = """
        {
            "experience_level": "",
            "employment_status": [],
            "work_location": "",
            "salary": {"min": "", "max": "", "period": "", "currency": ""},
            "benefits": [],
            "job_functions": [],
            "required_skills": {
                "programming_languages": [],
                "tools": [],
                "frameworks": [],
                "databases": [],
                "other": []
            },
            "required_certifications": [],
            "required_minimum_degree": "",
            "required_experience": "",
            "industries": [],
            "additional_keywords": []
            }
        """
        
        # Format messages following Mistral's chat format
        messages = [
            {"role": "user", "content": f"Label the following job posting in pure JSON format based on this example schema. If no information for a field, leave the field blank.\n\nExample schema:\n{schema}\n\nJob posting:\n{source_text}"}
        ]
        
        # Use tokenizer to encode and handle message formatting
        chat_input = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            return_tensors=None
        )
        
        # For training, we want the model to generate the target text
        # We'll tokenize the entire sequence including prompt and target together
        full_prompt = chat_input
        if isinstance(full_prompt, list):
            # If it's a list of tokens, decode it to a string first
            full_prompt = tokenizer.decode(full_prompt)
        
        # Combine with the target and tokenize everything
        full_text = full_prompt + target_text + tokenizer.eos_token
        
        # Tokenize inputs
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        
        # Create labels for causal LM (copy input_ids)
        labels = input_ids.clone()
        
        # Find where the user message ends and assistant response begins
        # We don't want to predict the prompt part, only the response
        assistant_start = full_text.find("<|assistant|>")
        if assistant_start != -1:
            # Find this position in the tokenized input
            assistant_token_pos = len(tokenizer.encode(full_text[:assistant_start]))
            
            # Mask everything before the assistant token with -100
            labels[:assistant_token_pos] = -100
        
        # Also mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Split data into train and validation sets (90/10 split)
train_size = int(0.9 * len(data_pairs))
val_size = len(data_pairs) - train_size

train_data = data_pairs[:train_size]
val_data = data_pairs[train_size:]

print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")

# For Mistral, configure the model and dataset with appropriate sequence length
max_length = 1536  # Adjust based on your needs and GPU memory

print(f"Using max_length={max_length}")

# Create datasets
train_dataset = MistralDataset(train_data, tokenizer, max_length=max_length)
val_dataset = MistralDataset(val_data, tokenizer, max_length=max_length)

# Set up training arguments - matched to your Accelerate config settings
training_args = TrainingArguments(
    output_dir="./mistral_7b_lora_output",
    run_name="mistral-7b-lora-job-extraction-multi-node",
    num_train_epochs=3,
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8, 
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=2e-4,  # Slightly higher learning rate for LoRA
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="paged_adamw_8bit",  # Memory-efficient optimizer for LoRA
    # Weights & Biases integration
    report_to="wandb",
    # No deepspeed parameter - letting Accelerate handle it from the config
)

# Add debug code to check for label issues
sample_item = train_dataset[0]
print("\n=== DEBUGGING DATASET LABELS ===")
print(f"Sample input shape: {sample_item['input_ids'].shape}")
print(f"Sample attention mask shape: {sample_item['attention_mask'].shape}")
print(f"Sample labels shape: {sample_item['labels'].shape}")
print(f"Number of valid labels (not -100): {(sample_item['labels'] != -100).sum().item()}")
print(f"Percentage of valid labels: {(sample_item['labels'] != -100).sum().item() / sample_item['labels'].shape[0] * 100:.2f}%")
print("===============================\n")

if (sample_item['labels'] != -100).sum().item() == 0:
    print("WARNING: All labels are masked (-100). This will result in zero loss!")
    print("Check your tokenization and dataset processing.")

# Initialize wandb before training - only on main process
if os.environ.get("LOCAL_RANK", "0") == "0":
    wandb.init(project="flan-t5-job-extraction", name="mistral-7b-lora-training-multi-node")

# Add a custom callback to monitor training more closely
class TrainingMonitorCallback(TrainerCallback):
    """Custom callback for monitoring training process and debugging issues."""
    def __init__(self, trainer):
        self.trainer = trainer
        self.step_count = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        """Monitor loss and gradients after each step"""
        self.step_count += 1
        if self.step_count % 10 == 0:  # Check every 10 steps
            # Get the current loss
            if state.log_history:
                latest_log = state.log_history[-1] if state.log_history else {}
                loss = latest_log.get('loss', None)
                
                if loss is not None and loss == 0.0:
                    print("\nWARNING: Loss is zero! This indicates an issue with the model/data.")
                    
                    # Check a sample batch from the training dataset
                    if hasattr(self.trainer, 'train_dataset') and len(self.trainer.train_dataset) > 0:
                        sample_batch = next(iter(self.trainer.get_train_dataloader()))
                        labels = sample_batch['labels']
                        
                        # Check how many non-masked (-100) labels we have
                        non_masked = (labels != -100).sum().item()
                        total = labels.numel()
                        print(f"Batch labels check: {non_masked}/{total} valid labels "
                              f"({non_masked/total*100:.2f}%)")
                
                # Log additional info to wandb if available
                if 'wandb' in args.report_to and os.environ.get("LOCAL_RANK", "0") == "0":
                    wandb.log({
                        "non_zero_loss_steps": 1 if loss and loss > 0 else 0,
                        "custom_step": self.step_count
                    })

# Initialize Trainer with the custom callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Add our custom monitoring callback
monitor_callback = TrainingMonitorCallback(trainer)
trainer.add_callback(monitor_callback)

# Start training
print("Starting multi-node training with Accelerate DeepSpeed Plugin...")
trainer.train()

# Save the final model (only on the main process)
if os.environ.get("LOCAL_RANK", "0") == "0":
    model_path = "./mistral_7b_lora_final"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

    # Test generation with a sample from validation data
    if len(val_data) > 0:
        test_item = val_data[0]
        
        # Access source and target based on data structure
        if isinstance(test_item, tuple):
            test_source = test_item[0]
            test_target = test_item[1]
        elif isinstance(test_item, dict):
            test_source = test_item["source"]
            test_target = test_item["target"]
        else:
            test_source = str(test_item)
            test_target = "Unknown format"
        
        print(f"\nTesting generation with sample from validation data:")
        print(f"Input: {test_source[:100]}...")
        
        # Generate text with Mistral
        schema = """
        {
            "experience_level": "",
            "employment_status": [],
            "work_location": "",
            "salary": {"min": "", "max": "", "period": "", "currency": ""},
            "benefits": [],
            "job_functions": [],
            "required_skills": {
                "programming_languages": [],
                "tools": [],
                "frameworks": [],
                "databases": [],
                "other": []
            },
            "required_certifications": [],
            "required_minimum_degree": "",
            "required_experience": "",
            "industries": [],
            "additional_keywords": []
            }
        """
        
        messages = [
            {"role": "user", "content": f"Label the following job posting in pure JSON format based on this example schema. If no information for a field, leave the field blank.\n\nExample schema:\n{schema}\n\nJob posting:\n{test_source}"}
        ]
        
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors=None
        )
        
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = inputs.to(model.device)
        
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=384,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated}")
        print(f"Expected: {test_target}")
        
    # Finish wandb if it was initialized
    if wandb.run is not None:
        wandb.finish() 