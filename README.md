# Fetch Raw Data

This step allows you to fetch job listings and store them in a MongoDB database using LinkedIn's API.

## Getting Started

### Clone the Repository

First, clone the repository to your local machine using the following command:

```bash
git https://github.com/alexlanxy/JobExtractX.git
cd JobExtractX
```


## Setup Instructions

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Configure LinkedIn Account

Before running the script, you need to configure your LinkedIn account credentials. This is done using environment variables, which can be set in a `.env` file in the `data` directory. The `.env` file should contain your LinkedIn email and password as follows:

```
LINKEDIN_EMAIL=your_linkedin_email@example.com
LINKEDIN_PASSWORD=your_linkedin_password
```

- If you do not have a `.env` file, you can replace the default values in `config_setup.py` with your LinkedIn email and password directly. However, using a `.env` file is recommended for security and flexibility.

### 3. Run the Script
Navigate to the `data` directory.
To fetch job listings and store them in MongoDB, run the following command:

```bash
cd data
python fetch_raw_data.py
```

- By default, the script skips job IDs that already exist in the `job_raw` collection.
- Use the `--all` flag to process all job IDs, including those that already exist:

```bash
python fetch_raw_data.py --all
```


## Data Labeling

### 1. Setup Requirements

Ensure you have all necessary Python packages installed. If you haven't already,install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Before running the data labeling script, ensure your environment is properly configured:

- **DeepSeek API Key**: Add your DeepSeek API key to the `.env` file in the `data` directory. The `.env` file should include:

```
DEEPSEEK_API_KEY=your_deepseek_api_key
```

This key is necessary for accessing the DeepSeek service for labeling.

### 3. Run the Data Labeling Script

To label job data and store the results in MongoDB, run the following command:

```bash
python data_labeling.py
```

- This script will process job data, label it, and update the MongoDB collection with labeled data.

## GPU Reservations Timeline
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



    section CPUs
    cpu_server :active, 2025-03-13 20:30, 2025-03-20 20:30




