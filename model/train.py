from data_preprocessing import prepare_data
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, random_split
import torch

data_pairs = prepare_data()
print("data set size: ", len(data_pairs))

# Load the T5 tokenizer and model
model_name = 'google/flan-t5-xl'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Split the data into training and testing sets
train_size = int(0.8 * len(data_pairs))
test_size = len(data_pairs) - train_size
train_dataset, test_dataset = random_split(data_pairs, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)