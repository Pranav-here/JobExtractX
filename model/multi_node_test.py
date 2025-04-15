from accelerate import Accelerator
import torch
import torch.nn as nn
import os
import socket
from torch.utils.data import Dataset, DataLoader

# Define a simple dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size=100, features=10):
        self.data = torch.randn(size, features)
        self.targets = torch.randn(size, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def main():
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Get distributed information
    is_main_process = accelerator.is_main_process
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    device = accelerator.device
    
    # Get node information
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    
    # Print distributed training information
    if is_main_process:
        print("\n=== Distributed Training Configuration ===")
        print(f"Total number of processes: {num_processes}")
        print(f"Using device: {device}")
        
    # Each process prints its own information
    print(f"Process {process_index}: Running on {hostname} ({ip_address}) using {device}")

    # Create a dummy dataset and dataloader
    batch_size = 16
    train_dataset = DummyDataset(size=100, features=10)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True
    )

    # Initialize the model
    model = SimpleModel()

    # Prepare model, optimizer, and the dataloader for distributed training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Print model parameters to verify they're the same across processes
    params_sum = sum(p.sum().item() for p in model.parameters())
    print(f"Process {process_index}: Model parameters sum = {params_sum}")
    
    # Create a simple dummy "training loop"
    for step, (inputs, targets) in enumerate(train_dataloader):
        if step >= 3:  # Only run a few steps as a test
            break
            
        # Forward pass
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        
        # Backward pass
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        # Print loss on each process
        print(f"Process {process_index}: Step {step}, Loss = {loss.item()}")
    
    # Check model parameters again after training
    params_sum_after = sum(p.sum().item() for p in model.parameters())
    print(f"Process {process_index}: Model parameters sum after training = {params_sum_after}")
    
    # Synchronize to make output readable
    accelerator.wait_for_everyone()
    
    if is_main_process:
        print("\n=== Test Completed Successfully ===")
        print("Model parameter values before:", params_sum)
        print("Model parameter values after:", params_sum_after)
        if params_sum != params_sum_after:
            print("Parameters changed during training - good!")
        print("The model is working correctly across all processes!")

if __name__ == "__main__":
    main()