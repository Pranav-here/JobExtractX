from accelerate import Accelerator
import torch
import torch.nn as nn
import os
import socket

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

    # Initialize the model
    model = SimpleModel()

    # Prepare model and optimizer for distributed training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model, optimizer = accelerator.prepare(model, optimizer)

    # Create dummy input data
    batch_size = 16
    inputs = torch.randn(batch_size, 10).to(device)
    
    # Forward pass
    outputs = model(inputs)
    
    # Print model parameters to verify they're the same across processes
    params_sum = sum(p.sum().item() for p in model.parameters())
    print(f"Process {process_index}: Model parameters sum = {params_sum}")
    
    # Create a simple dummy "training loop"
    for step in range(3):
        # Forward pass with different inputs on each process to test parameter sync
        inputs = torch.randn(batch_size, 10).to(device) + process_index
        outputs = model(inputs)
        loss = outputs.mean()
        
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