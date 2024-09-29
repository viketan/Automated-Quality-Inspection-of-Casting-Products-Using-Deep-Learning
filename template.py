import os
from pathlib import Path
import logging

# Configure logging to display messages in a specific format
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# List of files and directories to be created
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/components/__init__.py",
    f"src/utils/__init__.py",
    f"src/config/__init__.py",
    f"src/config/configuration.py",
    f"src/pipeline/__init__.py",
    f"src/entity/__init__.py",
    f"src/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
]

def create_directory(filedir):
    """
    Create a directory if it does not already exist.

    Args:
        filedir (str): The directory path to create.
    """
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

def create_file(filepath):
    """
    Create an empty file if it does not exist or if it is empty.

    Args:
        filepath (Path): The file path to create.
    """
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            # Create an empty file
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filepath.name} already exists")

def setup_project_structure(file_list):
    """
    Set up the project structure by creating directories and files.

    Args:
        file_list (list): A list of file paths to create.
    """
    for filepath in file_list:
        filepath = Path(filepath)
        filedir, _ = os.path.split(filepath)

        # Create the necessary directory
        create_directory(filedir)
        
        # Create the file
        create_file(filepath)

if __name__ == "__main__":
    # Execute the project setup
    setup_project_structure(list_of_files)
