import os
from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity import (DataIngestionConfig,
                        TrainingConfig,
                        TrainingParam,
                        EvaluationConfig,
                        PredictionConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def get_training_config(self) -> TrainingConfig:
        config = self.config.training

        create_directories([config.root_dir])

        training_config = TrainingConfig(
            root_dir=config.root_dir,
            data_dir=config.data_dir,
            model_filename=config.model_filename
        )

        return training_config
    
    def get_training_param(self)->TrainingParam:
        param = self.params.training
        training_param = TrainingParam(
            img_size = param.img_size,
            batch_size= param.batch_size,
            channel= param.channel,
            epochs=  param.epochs,
            activation_function_hidden= param.activation_function_hidden,
            activation_function_output = param.activation_function_output,
            optimizer= param.optimizer,
            loss= param.loss,
            metrics= param.metrics
        )
        return training_param
    
    def get_evaluation_config(self)-> EvaluationConfig:
        config = self.config.evaluation
        evaluation_config = EvaluationConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            mlflow_uri= config.mlflow_uri,
            data_dir=config.data_dir
        )
        return evaluation_config
    
    def get_prediction_config(self)-> PredictionConfig:
        config = self.config.prediction
        evaluation_config = PredictionConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            img_size= config.img_size,
            class_names=config.class_names
        )
        return evaluation_config