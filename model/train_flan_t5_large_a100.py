from data_preprocessing import prepare_data
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
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
        
        # Flan-T5 expects inputs with a "prefix" that defines the task
        # Using a more instruction-based prompt format
        source_text = f"Extract job information from the following text: {source_text}"
        
        # Tokenize inputs and targets
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = source_encoding.input_ids.squeeze()
        attention_mask = source_encoding.attention_mask.squeeze()
        labels = target_encoding.input_ids.squeeze()
        
        # Replace padding token id with -100 so it's ignored in loss computation
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

# For long sequences (1000+ tokens), configure the model and dataset accordingly
max_source_length = 1536  # Increased for longer sequences
max_target_length = 512   # Adjust based on your target length requirements

print(f"Using max_source_length={max_source_length} and max_target_length={max_target_length}")

# Create datasets with longer sequence lengths
train_dataset = FlanT5Dataset(train_data, tokenizer, max_source_length=max_source_length, max_target_length=max_target_length)
val_dataset = FlanT5Dataset(val_data, tokenizer, max_source_length=max_source_length, max_target_length=max_target_length)

# Set up training arguments - RTX 6000 has 24GB VRAM
# Using gradient accumulation for handling long sequences
training_args = TrainingArguments(
    output_dir="./flan_t5_model_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2, 
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=True,  # Use mixed precision for faster training and memory savings
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # Memory optimization settings
    gradient_checkpointing=False,  # Disabled since we have enough GPU memory
    optim="adafactor",  # Memory-efficient optimizer
    # Weights & Biases integration
    report_to="wandb",
)

# Initialize wandb before training
wandb.init(project="flan-t5-job-extraction", name="flan-t5-job-training")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to GPU
model = model.to(device)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start training
print("Starting training...")
trainer.train()

# Save the final model
model_path = "./flan_t5_model_final"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model saved to {model_path}")

# Example of how to use the trained model
def generate_text(input_text, max_length=100):
    inputs = tokenizer(f"Extract job information from the following text: {input_text}", 
                      return_tensors="pt", truncation=True, max_length=max_source_length)
    inputs = inputs.to(model.device)
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    generated = generate_text(test_source)
    print(f"Generated: {generated}")
    print(f"Expected: {test_target}") 