'''
Utility stuff for the project
'''
import os

def setup_project_directories():
    # Create necessary project directories if they don't exist
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'results'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)