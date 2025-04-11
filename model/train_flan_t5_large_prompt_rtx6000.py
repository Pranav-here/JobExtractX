from data_preprocessing import prepare_data
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import Dataset
import wandb


data_pairs = prepare_data()
print("data set size: ", len(data_pairs))

# Print the structure of the first data item to understand its format
if len(data_pairs) > 0:
    print("Sample data structure:", type(data_pairs[0]))
    print("First data item:", data_pairs[0])

# Initialize tokenizer and model
model_name = "google/flan-t5-large"  # You can also use flan-t5-small, flan-t5-large, etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Dataset class
class FlanT5Dataset(Dataset):
    def __init__(self, data_pairs, tokenizer, max_source_length=1024, max_target_length=128):
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, index):
        # Assuming data_pairs contains tuples where the first element is the source
        # and the second element is the target
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
            
        
        # Flan-T5 expects inputs with a "prefix" that defines the task
        # Using a more instruction-based prompt format
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
        source_text = (
            f"Label the following job posting in pure JSON format based on this example schema. "
            f"If no information for a field, leave the field blank.\n\n"
            f"Example schema:\n{schema}\n\n"
            f"Job posting:\n{source_text}"
        )

        
        # Tokenize inputs and targets
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Important: don't use return_tensors="pt" here, we'll handle the conversion differently
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
        )
        
        input_ids = source_encoding.input_ids.squeeze()
        attention_mask = source_encoding.attention_mask.squeeze()
        
        # Convert target_encoding to tensor and handle -100 masking properly
        target_ids = torch.tensor(target_encoding.input_ids)
        labels = target_ids.clone().squeeze()
        
        # Replace padding token id with -100 so it's ignored in loss computation
        # But ensure we're not masking ALL tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Verify we have some valid labels
        if (labels != -100).sum() == 0:
            # If all are masked, unmask at least the first token to prevent zero loss
            # This is a failsafe - the actual issue should be fixed in data preparation
            print(f"WARNING: All labels were masked at index {index}. Unmasking first token.")
            if len(labels) > 0:
                labels[0] = target_ids[0]
        
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

# For long sequences (1000+ tokens), configure the model and dataset accordingly
max_source_length = 1536  # Increased for longer sequences
max_target_length = 384   # Adjust based on your target length requirements

print(f"Using max_source_length={max_source_length} and max_target_length={max_target_length}")

# Create datasets with longer sequence lengths
train_dataset = FlanT5Dataset(train_data, tokenizer, max_source_length=max_source_length, max_target_length=max_target_length)
val_dataset = FlanT5Dataset(val_data, tokenizer, max_source_length=max_source_length, max_target_length=max_target_length)

# Set up training arguments - RTX 6000 has 24GB VRAM
# Using gradient accumulation for handling long sequences
training_args = TrainingArguments(
    output_dir="./flan_t5_model_output_rtx6000_prompt",
    run_name="flan-t5-job-extraction-rtx6000-prompt",  # Adding explicit run_name
    num_train_epochs=3,
    per_device_train_batch_size=1,  
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16, 
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=False,  
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # Memory optimization settings
    gradient_checkpointing=False,  # Disabled since we have enough GPU memory
    optim="adafactor",  # Memory-efficient optimizer
    # Weights & Biases integration
    report_to="wandb",
)

# Add debug code to check for label issues
# Get a sample from the dataset to check label processing
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

# Initialize wandb before training
wandb.init(project="flan-t5-job-extraction", name="flan-t5-job-training-rtx6000-prompt-ordered-fields")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to GPU
model = model.to(device)

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
                if 'wandb' in args.report_to:
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
print("Starting training...")
trainer.train()

# Save the final model
model_path = "./flan_t5_model_final_rtx6000_prompt_ordered_fields"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model saved to {model_path}")


# Add debug information to log model inputs and outputs
def debug_model_io(model, tokenizer, input_text, n=3):
    """Print input token IDs and actual tokens to help debug model issues"""
    print("\n=== MODEL INPUT/OUTPUT DEBUGGING ===")
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
    prompt_text = (
        f"Label the following job posting in pure JSON format based on this example schema. "
        f"If no information for a field, leave the field blank.\n\n"
        f"Example schema:\n{schema}\n\n"
        f"Job posting:\n{input_text}"
    )
    inputs = tokenizer(prompt_text, 
                   return_tensors="pt", truncation=True, max_length=64)
    
    # Print first n tokens of input
    input_ids = inputs.input_ids[0]
    print(f"First {n} input token IDs: {input_ids[:n].tolist()}")
    print(f"First {n} input tokens: {[tokenizer.decode([id]) for id in input_ids[:n].tolist()]}")
    
    # Print special token IDs for reference
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print("===================================\n")

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
    
    # Run the input/output debug function
    debug_model_io(model, tokenizer, test_source[:100])
    
    print(f"\nTesting generation with sample from validation data:")
    print(f"Input: {test_source[:100]}...")
    
    # Generate text directly
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
    prompt_text = (
        f"Label the following job posting in pure JSON format based on this example schema. "
        f"If no information for a field, leave the field blank.\n\n"
        f"Example schema:\n{schema}\n\n"
        f"Job posting:\n{test_source}"
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_source_length)
    inputs = inputs.to(model.device)
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_target_length,
        num_beams=4,
        early_stopping=True,
        use_cache=True
    )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated}")
    print(f"Expected: {test_target}") 