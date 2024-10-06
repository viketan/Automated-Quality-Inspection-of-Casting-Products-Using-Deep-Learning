from setuptools import setup, find_packages
from typing import List

# Declaring variables for setup functions
PROJECT_NAME = "Automated-Quality-Inspection-of-Casting-Product"  # Adjusted to a more conventional package name (no spaces)
VERSION = "0.0.1"
AUTHOR = "Viketan"
DESCRIPTION = ("""This project focuses on automating the quality inspection process of casting products using deep learning techniques. The dataset consists of grayscale images of submersible pump impellers, categorized into defective and non-defective (ok) classes. Casting defects, such as blowholes, pinholes, and shrinkage defects, are a significant challenge in the casting industry, leading to high rejection rates and financial loss due to manual, time-consuming inspections.

The project aims to develop a classification model that automates the defect detection process, improving accuracy and reducing inspection time. The dataset is already pre-processed with image augmentation applied and includes both 300x300 and 512x512 grayscale images for training and testing. The goal is to create a robust system capable of identifying casting defects, contributing to more efficient quality control in manufacturing.""")
REQUIREMENT_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."

def get_requirements_list() -> List[str]:
    """
    Reads the requirements from the requirements.txt file.

    Returns:
        List[str]: A list of package names specified in requirements.txt.
    """
    try:
        with open(REQUIREMENT_FILE_NAME) as requirement_file:
            # Read lines and clean up whitespace
            requirement_list = [line.strip() for line in requirement_file.readlines()]
            # Remove editable installation if present
            if HYPHEN_E_DOT in requirement_list:
                requirement_list.remove(HYPHEN_E_DOT)
            return requirement_list
    except FileNotFoundError:
        print(f"Warning: {REQUIREMENT_FILE_NAME} not found. Please ensure it exists.")
        return []  # Return an empty list if requirements.txt is not found

setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages=find_packages(),  # Automatically find and include all packages in the project
    install_requires=get_requirements_list(),  # Install required packages
    python_requires='>=3.8',  # Specify the minimum Python version required
    classifiers=[  # Classifiers to help others find your package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
