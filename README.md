# Fetch Raw Data


## Setup Instructions

### 1. Install Requirements

Navigate to the `src` directory and install the required Python packages:

```bash
cd src
pip install -r requirements.txt
```

### 2. Configure LinkedIn Account

Before running the script, you need to configure your LinkedIn account credentials. This typically involves setting up a configuration file or environment variables with your LinkedIn username and password. Ensure that your `config_setup.py` is correctly set up to handle these credentials.

### 3. Run the Script

To fetch job listings and store them in MongoDB, run the following command:

```bash
python fetch_raw_data.py -s
```

- The `-s` flag is optional and is used to skip job IDs that already exist in the `job_raw` collection.

