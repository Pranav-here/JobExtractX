from linkedin_api import Linkedin
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_config():
    return {
        'linkedin_email': os.getenv('LINKEDIN_EMAIL', 'default_email@example.com'), # if you don't have a .env, replac with your linkedin email
        'linkedin_password': os.getenv('LINKEDIN_PASSWORD', 'default_password'), # if you don't have a .env, replace with your linkedin password
        'mongodb_uri': "mongodb+srv://team_member:CS584GROUP9@cluster0.b498i.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    }

def setup_config():
    # Load configuration from config file
    config = get_config()

    # Initialize LinkedIn API
    api = Linkedin(config["linkedin_email"], config["linkedin_password"])

    # Initialize MongoDB client
    client = MongoClient(config["mongodb_uri"], server_api=ServerApi('1'))
    db = client["myDatabase"]

    return config, api, client, db

