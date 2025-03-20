from data_preprocessing import prepare_data
from transformers import T5Tokenizer, T5ForConditionalGeneration, BitsAndBytesConfig
from torch.utils.data import DataLoader, random_split
import torch

# -------------------------------------------------------------------------
# 1) Data
# -------------------------------------------------------------------------
data_pairs = prepare_data()
print("data set size:", len(data_pairs))

# -------------------------------------------------------------------------
# 2) Tokenizer & Model in 8-bit
# -------------------------------------------------------------------------
model_name = 't5-large'
tokenizer = T5Tokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config
)

# -------------------------------------------------------------------------
# 3) Split data
# -------------------------------------------------------------------------
train_size = int(0.8 * len(data_pairs))
test_size = len(data_pairs) - train_size
train_dataset, test_dataset = random_split(data_pairs, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# -------------------------------------------------------------------------
# 4) Device setup
# -------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# DO NOT call model.to(device) since load_in_8bit=True

# -------------------------------------------------------------------------
# 5) Optimizer with smaller LR
# -------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)

# -------------------------------------------------------------------------
# 6) Helper functions
# -------------------------------------------------------------------------
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

torch.autograd.set_detect_anomaly(True)

# -------------------------------------------------------------------------
# 7) Training & Evaluation
# -------------------------------------------------------------------------
epochs = 3
max_length = 512  # define a max sequence length to avoid very long texts

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i, batch in enumerate(train_loader):
        # batch: (list_of_input_strings, list_of_target_strings)
        inputs_text, targets_text = batch

        # Tokenize inputs
        inputs = tokenizer(
            inputs_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        ).to(device)

        # Tokenize targets
        targets_enc = tokenizer(
            targets_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        ).to(device)

        # Replace pad tokens in labels
        labels = targets_enc['input_ids'].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        # Optional: check for NaNs in input
        if check_for_nan_inf(inputs['input_ids']) or check_for_nan_inf(labels):
            print(f"Skipping batch {i} due to NaN/Inf values in inputs or labels")
            continue

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels
        )
        loss = outputs.loss

        # Backprop
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check for NaNs in gradients
        if check_gradients(model):
            print("NaN or Inf found in gradients, skipping update")
            continue

        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss.item()}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}")

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------
    model.eval()
    total_eval_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs_text, targets_text = batch
            inputs = tokenizer(
                inputs_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=max_length
            ).to(device)

            targets_enc = tokenizer(
                targets_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=max_length
            ).to(device)

            labels = targets_enc['input_ids'].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=labels
            )
            total_eval_loss += outputs.loss.item()

    avg_val_loss = total_eval_loss / len(test_loader)
    print(f"Validation Loss: {avg_val_loss}")
