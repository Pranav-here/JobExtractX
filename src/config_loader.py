import json
import os


def load_config(file_path=os.path.join(os.path.dirname(__file__), 'config.json')):
    with open(file_path, 'r') as config_file:
        return json.load(config_file) 