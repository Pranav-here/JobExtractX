from data_preprocessing import prepare_data
from transformers import T5Tokenizer, T5ForConditionalGeneration, BitsAndBytesConfig
from torch.utils.data import DataLoader, random_split
import torch

data_pairs = prepare_data()
print("data set size: ", len(data_pairs))

# Load the T5 tokenizer and model with quantization
model_name = 'google/flan-t5-xl'
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True  # Enable 8-bit quantization
)

model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config
)

# Split the data into training and testing sets
train_size = int(0.8 * len(data_pairs))
test_size = len(data_pairs) - train_size
train_dataset, test_dataset = random_split(data_pairs, [train_size, test_size])

# Create data loaders with reduced batch size
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

scaler = torch.amp.GradScaler()

def check_for_nan_inf(tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print("Data contains NaN or Inf values")
        return True
    return False

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"Gradient for {name} contains NaN or Inf values")
                return True
    return False

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        inputs, targets = batch
        inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(device)
        targets = tokenizer(targets, return_tensors='pt', padding=True, truncation=True).to(device)

        # Check for NaNs/Infs in inputs
        if check_for_nan_inf(inputs['input_ids']) or check_for_nan_inf(targets['input_ids']):
            print(f"Skipping batch {i} due to NaN/Inf values in inputs")
            continue

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=inputs['input_ids'], labels=targets['input_ids'])
            loss = outputs.loss

        # Log loss value
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Loss is NaN or Inf at batch {i}, skipping update")
            continue

        # Scale the loss
        scaler.scale(loss).backward()

        # Unscale the gradients before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check for NaNs/Infs in gradients
        if check_gradients(model):
            print("NaN or Inf found in gradients, skipping update")
            continue

        # Step the optimizer and update the scaler
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        
        # Print loss every 10 batches for more frequent logging
        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss.item()}", flush=True)

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}", flush=True)

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        total_eval_loss = 0
        for batch in test_loader:
            inputs, targets = batch
            inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(device)
            targets = tokenizer(targets, return_tensors='pt', padding=True, truncation=True).to(device)

            outputs = model(input_ids=inputs['input_ids'], labels=targets['input_ids'])
            loss = outputs.loss
            total_eval_loss += loss.item()
        print(f"Validation Loss: {total_eval_loss / len(test_loader)}")

torch.autograd.set_detect_anomaly(True)