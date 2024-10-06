import os
import zipfile
import gdown
from src import logger
from src.utils.common import get_size
from src.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        '''
        Fetch data from the gdrive
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            data_ingestion_dir = self.config.root_dir
            os.makedirs(data_ingestion_dir, exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            result = gdown.download(prefix + file_id, zip_download_dir, quiet=False)

            if result is None:
                raise Exception(f"Failed to download file from {dataset_url}")

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
            return zip_download_dir

        except Exception as e:
            logger.error(f"Error in downloading file: {e}")
            raise e

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory
        Function returns None
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            logger.info(f"Extracting zip file {self.config.local_data_file} to {unzip_path}")
            
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            
            logger.info(f"Extraction completed for {self.config.local_data_file}")

        except Exception as e:
            logger.error(f"Error in extracting file: {e}")
            raise e
