import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

LOG_DIR = "logs"  # Directory to store log files

def get_log_file_name() -> str:
    """
    Generates a log file name based on the current timestamp.

    Returns:
        str: Log file name in the format 'log_YYYY-MM-DD_HH-MM-SS.log'.
    """
    return f"log_{TIMESTAMP}.log"

LOG_FILE_NAME = get_log_file_name()

# Remove the existing log directory if it exists
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)

# Create a new log directory
os.makedirs(LOG_DIR, exist_ok=True)

# Define the full path for the log file
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# Configure the logging settings
logging.basicConfig(
    filename=Path(LOG_FILE_PATH),
    filemode="w",  # Use "a" to append logs instead of overwriting
    format='[%(asctime)s] \t%(levelname)s \t%(lineno)d \t%(filename)s \t%(funcName)s() \t%(message)s',
    level=logging.INFO  # Log level can be set to DEBUG for more verbose output
)

# Create a logger instance
logger = logging.getLogger("Manufacturing-Classification Logs")

# Optional: Log a message to indicate that logging has started
logger.info("Logging started for Chest-Cancer-Classification.")
