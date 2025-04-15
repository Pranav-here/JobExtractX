from accelerate import Accelerator
import torch
import os
import socket
import time

def main():
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Get distributed information
    world_size = accelerator.num_processes
    local_rank = accelerator.local_process_index
    global_rank = accelerator.process_index
    device = accelerator.device
    
    # Get node information
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    
    # Print basic information
    print(f"Process {global_rank}/{world_size}: Running on {hostname} ({ip_address}) using {device}")
    
    # Create a simple tensor
    tensor = torch.ones(1).to(device) * global_rank
    
    # Synchronize to make sure all processes have created their tensor
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("\n=== Starting communication test ===")
        
    # Test 1: Simple all_gather operation
    try:
        if accelerator.is_main_process:
            print("Test 1: Running all_gather...")
            
        gathered_tensors = accelerator.gather(tensor)
        
        if accelerator.is_main_process:
            values = [t.item() for t in gathered_tensors]
            expected = list(range(world_size))
            print(f"Gathered values: {values} (expected: {expected})")
            print("Test 1: all_gather completed successfully")
    except Exception as e:
        print(f"Process {global_rank}: Test 1 failed with error: {str(e)}")
    
    accelerator.wait_for_everyone()
    time.sleep(1)  # Add a brief delay between tests
    
    # Test 2: Broadcast from rank 0
    try:
        if accelerator.is_main_process:
            print("\nTest 2: Running broadcast...")
            
        if accelerator.is_main_process:
            broadcast_data = torch.tensor([42.0], device=device)
        else:
            broadcast_data = torch.tensor([0.0], device=device)
            
        accelerator.wait_for_everyone()
        torch.distributed.broadcast(broadcast_data, src=0)
        
        print(f"Process {global_rank}: Received broadcast value: {broadcast_data.item()}")
        
        if accelerator.is_main_process:
            print("Test 2: broadcast completed successfully")
    except Exception as e:
        print(f"Process {global_rank}: Test 2 failed with error: {str(e)}")
    
    accelerator.wait_for_everyone()
    time.sleep(1)  # Add a brief delay between tests
    
    # Test 3: Simple all_reduce operation
    try:
        if accelerator.is_main_process:
            print("\nTest 3: Running all_reduce...")
            
        test_tensor = torch.tensor([1.0], device=device)
        torch.distributed.all_reduce(test_tensor)
        
        print(f"Process {global_rank}: all_reduce result: {test_tensor.item()} (expected: {world_size}.0)")
        
        if accelerator.is_main_process:
            print("Test 3: all_reduce completed successfully")
    except Exception as e:
        print(f"Process {global_rank}: Test 3 failed with error: {str(e)}")
    
    # Final synchronization
    try:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("\n=== All communication tests completed! ===")
    except Exception as e:
        print(f"Process {global_rank}: Final sync failed with error: {str(e)}")

if __name__ == "__main__":
    main()