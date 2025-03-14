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

Navigate to the `data` directory and install the required Python packages:

```bash
cd data
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

To fetch job listings and store them in MongoDB, run the following command:

```bash
python fetch_raw_data.py -s
```

- The `-s` flag is optional and is used to skip job IDs that already exist in the `job_raw` collection.


## Data Labeling

### 1. Setup Requirements

Ensure you have all necessary Python packages installed. If you haven't already, navigate to the `data` directory and install the required packages:

```bash
cd data
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
    title GPU and CPU Reservations Timeline
    dateFormat  YYYY-MM-DD HH:mm
    section GPUs
    V100_2 :pending, 2025-03-15 19:00, 2025-03-22 19:00
    V100_3 :pending, 2025-03-22 19:05, 2025-03-29 19:05
    V100_4 :pending, 2025-03-29 20:15, 2025-04-05 20:15
    A100_pcie :pending, 2025-03-22 03:00, 2025-03-25 15:00
    A100_pcie_2 :pending, 2025-04-01 15:00, 2025-04-04 02:00
    V100_4_TACC :pending, 2025-04-05 20:20, 2025-04-12 20:20

    section CPUs
    cpu_server :active, 2025-03-14 01:30, 2025-03-21 01:30



