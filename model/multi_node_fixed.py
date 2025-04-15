from accelerate import Accelerator
import torch
import torch.nn as nn
import os
import socket
import time
import sys
import signal
from torch.utils.data import Dataset, DataLoader

# Configure timeouts for NCCL
os.environ["NCCL_TIMEOUT"] = "30"  # 30 seconds timeout
os.environ["NCCL_SOCKET_NTHREADS"] = "4"
os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TORCH_DISTRIBUTED_DETAIL"] = "DEBUG"  # More detailed distributed logs

# Set up signal handler for clean shutdown
def timeout_handler(signum, frame):
    print(f"Timeout occurred! Process might be hanging.")
    sys.stdout.flush()
    sys.exit(1)

# Register timeout handler
signal.signal(signal.SIGALRM, timeout_handler)

# Define a simple dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size=100, features=10):
        # Use bfloat16 for the data to match DeepSpeed config
        self.data = torch.randn(size, features).to(torch.bfloat16)
        self.targets = torch.randn(size, 1).to(torch.bfloat16)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x)

def main():
    try:
        # Set a 60-second alarm for the entire script
        signal.alarm(60)
        
        print("Starting multi-node DeepSpeed test...")
        sys.stdout.flush()
        
        # Initialize Accelerator with minimal DeepSpeed config
        # Use ZeRO-1 instead of ZeRO-2 and disable optimizer offloading
        accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=1,  # Reduced from 8
            deepspeed_plugin={
                "zero_stage": 1,  # Changed from stage 2 to stage 1
                "offload_optimizer": False,  # Changed from CPU offload
                "offload_param": False
            }
        )
        
        # Get distributed information
        world_size = accelerator.num_processes
        local_rank = accelerator.local_process_index
        global_rank = accelerator.process_index
        device = accelerator.device
        
        # Get node information
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        print(f"Process {global_rank}/{world_size}: Running on {hostname} ({ip_address}) using {device}")
        sys.stdout.flush()
        
        # Create dataset and dataloader (smaller size for quicker test)
        print(f"Process {global_rank}: Creating dataset and dataloader")
        sys.stdout.flush()
        
        batch_size = 4
        train_dataset = DummyDataset(size=8, features=10)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False  # No shuffle for consistent results
        )
        
        print(f"Process {global_rank}: Creating model and optimizer")
        sys.stdout.flush()
        
        # Initialize model and optimizer
        model = SimpleModel().to(torch.bfloat16).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Prepare with DeepSpeed
        print(f"Process {global_rank}: Preparing model with DeepSpeed...")
        sys.stdout.flush()
        
        # Set a 20-second timeout for DeepSpeed preparation
        signal.alarm(20)
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        # Reset the alarm
        signal.alarm(60)
        
        print(f"Process {global_rank}: Model preparation complete")
        sys.stdout.flush()
        
        # Print initial parameter values sum
        params_sum = sum(p.sum().item() for p in model.parameters())
        print(f"Process {global_rank}: Initial model parameters sum = {params_sum}")
        sys.stdout.flush()
        
        # Run training loop (one batch only)
        print(f"Process {global_rank}: Starting training loop")
        sys.stdout.flush()
        
        inputs, targets = next(iter(train_dataloader))
        
        print(f"Process {global_rank}: Forward pass")
        sys.stdout.flush()
        outputs = model(inputs)
        
        print(f"Process {global_rank}: Computing loss")
        sys.stdout.flush()
        loss = torch.nn.functional.mse_loss(outputs, targets)
        print(f"Process {global_rank}: Loss = {loss.item()}")
        sys.stdout.flush()
        
        # Set a 15-second timeout for backward pass
        print(f"Process {global_rank}: Backward pass")
        sys.stdout.flush()
        signal.alarm(15)
        accelerator.backward(loss)
        # Reset the alarm
        signal.alarm(60)
        
        print(f"Process {global_rank}: Optimizer step")
        sys.stdout.flush()
        optimizer.step()
        optimizer.zero_grad()
        
        # Check parameter values after training
        params_sum_after = sum(p.sum().item() for p in model.parameters())
        print(f"Process {global_rank}: Final model parameters sum = {params_sum_after}")
        sys.stdout.flush()
        
        # Final synchronization
        print(f"Process {global_rank}: Training completed successfully!")
        sys.stdout.flush()
        
        # Turn off the alarm
        signal.alarm(0)
        
        # Clean up before exit
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            print(f"Process {global_rank}: Destroyed process group")
            sys.stdout.flush()
        
    except Exception as e:
        print(f"Process {global_rank if 'global_rank' in locals() else 'unknown'}: Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        
        # Clean up on error
        if 'torch' in globals() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
                print("Destroyed process group after error")
            except:
                pass
        
        # Turn off the alarm
        signal.alarm(0)

if __name__ == "__main__":
    main()