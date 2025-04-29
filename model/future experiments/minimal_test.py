from accelerate import Accelerator
import torch
import os
import socket
import sys
import time
import signal

# Configure timeouts for NCCL
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_TIMEOUT"] = "20"  # 20 seconds timeout
os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker"
os.environ["NCCL_DEBUG"] = "INFO"

# Set up signal handler for clean shutdown
def timeout_handler(signum, frame):
    print(f"Timeout occurred! Process might be hanging.")
    sys.stdout.flush()
    sys.exit(1)

# Register timeout handler
signal.signal(signal.SIGALRM, timeout_handler)

def main():
    try:
        # Set a 60-second alarm for the entire script
        signal.alarm(60)
        
        print("Starting minimal test...")
        sys.stdout.flush()
        
        # Initialize Accelerator with minimal settings (no DeepSpeed)
        accelerator = Accelerator()
        
        # Get process info
        global_rank = accelerator.process_index
        world_size = accelerator.num_processes
        device = accelerator.device
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        print(f"Process {global_rank}/{world_size} running on {hostname} ({ip_address}) using {device}")
        sys.stdout.flush()
        
        # Create a simple tensor on each process
        x = torch.tensor([float(global_rank + 1)], device=device)
        print(f"Process {global_rank}: Created tensor with value {x.item()}")
        sys.stdout.flush()
        
        # Test 1: All-reduce
        print(f"Process {global_rank}: Starting all-reduce test")
        sys.stdout.flush()
        
        # Set a 10-second timeout for this operation
        signal.alarm(10)
        torch.distributed.all_reduce(x)
        # Reset the alarm
        signal.alarm(60)
        
        expected_sum = sum(range(1, world_size + 1))
        print(f"Process {global_rank}: After all-reduce, x = {x.item()} (expected: {expected_sum})")
        sys.stdout.flush()
        
        # Test 2: Broadcast
        print(f"Process {global_rank}: Starting broadcast test")
        sys.stdout.flush()
        
        y = torch.tensor([42.0 * (global_rank + 1)], device=device)
        print(f"Process {global_rank}: Before broadcast, y = {y.item()}")
        sys.stdout.flush()
        
        # Set a 10-second timeout for this operation
        signal.alarm(10)
        torch.distributed.broadcast(y, src=0)
        # Reset the alarm
        signal.alarm(60)
        
        print(f"Process {global_rank}: After broadcast, y = {y.item()} (expected: 42.0)")
        sys.stdout.flush()
        
        # Test 3: All-gather
        print(f"Process {global_rank}: Starting all-gather test")
        sys.stdout.flush()
        
        z = torch.tensor([100.0 * (global_rank + 1)], device=device)
        z_list = [torch.zeros_like(z) for _ in range(world_size)]
        
        # Set a 10-second timeout for this operation
        signal.alarm(10)
        torch.distributed.all_gather(z_list, z)
        # Reset the alarm
        signal.alarm(60)
        
        print(f"Process {global_rank}: After all-gather, z_list = {[t.item() for t in z_list]}")
        sys.stdout.flush()
        
        # Synchronize and finish
        print(f"Process {global_rank}: All tests completed successfully!")
        sys.stdout.flush()
        
        # Turn off the alarm
        signal.alarm(0)
        
        # Ensure clean process group shutdown
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    except Exception as e:
        print(f"Process {global_rank if 'global_rank' in locals() else 'unknown'}: Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        
        # Ensure clean process group shutdown even on error
        if 'torch' in globals() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except:
                pass
        
        # Turn off the alarm
        signal.alarm(0)
        sys.exit(1)

if __name__ == "__main__":
    main()