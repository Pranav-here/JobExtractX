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

Navigate to the `src` directory and install the required Python packages:

```bash
cd src
pip install -r requirements.txt
```

### 2. Configure LinkedIn Account

Before running the script, you need to configure your LinkedIn account credentials. This is done using environment variables, which can be set in a `.env` file in the `src` directory. The `.env` file should contain your LinkedIn email and password as follows:

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


