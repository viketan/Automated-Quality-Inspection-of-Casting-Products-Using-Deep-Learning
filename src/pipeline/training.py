from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.training import Training
from src.components.evaluation import Evaluation




STAGE_NAME = "Data Ingestion stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

STAGE_NAME = "Training stage"

class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training_param = config.get_training_param()
        training = Training(training_config,training_param)
        training.initiate_training()

STAGE_NAME = "Evaluation stage"

class Evalutionipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        training_param = config.get_training_param()
        evaluate = Evaluation(evaluation_config,training_param)
        evaluate.evaluate_model()





