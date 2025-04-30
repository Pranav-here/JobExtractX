# JobExtractX

A comprehensive system for job data extraction, labeling, and analysis using advanced language models.

## Install Requirements
```bash
cd JobExtractX
pip install -r requirements.txt
```

# Data Pipeline

## 1. Data Fetching

Configure your LinkedIn credentials in the `.env` file in the `data` directory:
```
LINKEDIN_EMAIL=your_linkedin_email@example.com
LINKEDIN_PASSWORD=your_linkedin_password
```

Run the data fetching script:
```bash
cd data
python fetch_raw_data.py
```

Optional flags:
- `--all`: Process all job IDs, including existing ones

## 2. Data Labeling

Configure your DeepSeek API key in the `.env` file:
```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

Run the labeling script:
```bash
python data_labeling.py
```

# Model Training and Evaluation
```bash
cd JobExtractX
```

## 1. Training Models

### Flan-T5 Models
- **Flan-T5 Large** (Full model training)
  ```bash
  python model/train_flan_t5_large_prompt_rtx6000.py
  ```
- **Flan-T5 XL** (LoRA training)
  ```bash
  # Batch size 2
  python model/train_flan_t5_xl_lora_prompt_bf16_batch_2.py
  
  # Batch size 4
  python model/train_flan_t5_xl_lora_prompt_bf16_batch_4.py
  ```

### Mistral-7B Models
- **Single Server** (Batch size 1)
  ```bash
  python model/train_mistral-7b_lora_prompt_bf16_batch_1_single_server.py
  ```
- **Multiple Servers** (Batch size 2)
    1. Config deepspeed on main server, our server config:
  
    ```bash
    (venv) cc@rtx-6000-17:~/JobExtractX$ accelerate config
    -----------------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
    This machine                                                                                                                                               
    -----------------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?                                                                                                                       
    multi-GPU                                                                                                                                                  
    How many different machines will you use (use more than 1 for multi-node training)? [1]: 2                                                                 
    -----------------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?                                                                                                                          
    0                                                                                                                                                          
    What is the IP address of the machine that will host the main process? 10.140.82.244                                                                         
    What is the port you will use to communicate with the main process? 29500                                                                                  
    Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: YES                      
    Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: YES                         
    Do you wish to optimize your script with torch dynamo?[yes/NO]:NO                                                                                          
    Do you want to use DeepSpeed? [yes/NO]: yes
    Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
    -----------------------------------------------------------------------------------------------------------------------------------------------------------What should be your DeepSpeed's ZeRO optimization stage?
    2                                                                                                                                                          
    -----------------------------------------------------------------------------------------------------------------------------------------------------------Where to offload optimizer states?                                                                                                                         
    cpu                                                                                                                                                        
    -----------------------------------------------------------------------------------------------------------------------------------------------------------Where to offload parameters?                                                                                                                               
    none                                                                                                                                                       
    How many gradient accumulation steps you're passing in your script? [1]: 8                                                                                 
    Do you want to use gradient clipping? [yes/NO]: NO                                                                                                         
    Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: NO                                          
    Do you want to enable Mixture-of-Experts training (MoE)? [yes/NO]: NO
    -----------------------------------------------------------------------------------------------------------------------------------------------------------Which Type of launcher do you want to use?
    pdsh                                                                                                                                                       
    DeepSpeed configures multi-node compute resources with hostfile. Each row is of the format `hostname slots=[num_gpus]`, e.g., `localhost slots=2`; for more information please refer official [documentation](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node). Please specify the location of hostfile: hostfile                                                                                                                                    
    Do you want to specify exclusion filter string? [yes/NO]: NO                                                                                               
    Do you want to specify inclusion filter string? [yes/NO]: NO                                                                                               
    How many GPU(s) should be used for distributed training? [1]:2                                                                                             
    -----------------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use mixed precision?
    bf16                                                                                                                                                       
    accelerate configuration saved at /home/cc/.cache/huggingface/accelerate/default_config.yaml  
    ```

    2. config the hostfile refered above
    ```bash
    10.140.82.244 slots=1
    10.140.83.105 slots=1
    ``` 
    3. To avoid the port restriction problem, we disabled the firewall
    
    4. run this script:
  ```bash
  python model/train_mistral-7b_lora_prompt_bf16_batch_2_multiple_servers.py
  ```


Note: All trained models are uploaded to Hugging Face for easy access.

## 2. Model Evaluation

### Flan-T5 Evaluation
```bash
# Evaluate Flan-T5 Large
python model/evaluate_flan_t5_large.py

# Evaluate Flan-T5 XL
python model/evaluate_flan_t5_xl_batch_2.py
python model/evaluate_flan_t5_xl_batch_4.py
```

### Mistral-7B Evaluation
```bash
python model/evaluate_mistral_lora_file.py
```

Evaluation results are stored in `evaluation_results/` within each model folder. To visualize results:
```bash
python evaluation_results/[model_folder]/visualize.py
```

## 3. Inference
```bash
python model/infer_job_extract_models.py
```

# Reserved GPU

```mermaid
gantt
    title GPU and CPU Reservations Timeline (Chicago Time)
    dateFormat  YYYY-MM-DD HH:mm
    section GPUs
    RTX_6000_11 :active, 2025-04-12 09:05, 2025-04-19 09:05
    RTX_6000_10 :active, 2025-04-11 19:05, 2025-04-16 07:55
    RTX_6000_7 :active, 2025-04-10 11:05, 2025-04-17 11:05
    V100_4 :active, 2025-03-29 15:15, 2025-04-05 15:15
    RTX_6000_12 :active, 2025-04-13 20:10, 2025-04-20 20:10
    A100_pcie_2 :active, 2025-04-01 10:00, 2025-04-03 21:00
    RTX_6000_13 :active, 2025-04-13 19:24, 2025-04-20 19:24
    A100_pcie :active, 2025-03-21 22:00, 2025-03-25 10:00
    RTX_6000 :active, 2025-03-14 12:05, 2025-03-21 12:05
    V100_3 :active, 2025-03-22 14:05, 2025-03-29 14:05
    RTX_6000_6 :active, 2025-04-08 11:36, 2025-04-09 10:00
    RTX_6000_8 :active, 2025-04-10 11:05, 2025-04-17 11:05
    V100_2 :active, 2025-03-15 14:00, 2025-03-22 14:00
    RTX_6000_9 :active, 2025-04-10 11:05, 2025-04-17 11:05
    RTX_6000_14 :active, 2025-04-14 13:16, 2025-04-21 13:16
    RTX_6000_16 :active, 2025-04-16 23:30, 2025-04-19 17:30
    RTX_6000_15 :active, 2025-04-14 13:35, 2025-04-15 13:35
    RTX_6000_17 :active, 2025-04-26 12:05, 2025-05-03 12:05

    section CPUs
    cpu_server :active, 2025-03-13 20:30, 2025-03-20 20:30