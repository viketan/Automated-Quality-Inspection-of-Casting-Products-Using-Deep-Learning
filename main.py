from src import logger
from src.pipeline.training import DataIngestionTrainingPipeline,TrainingPipeline,Evalutionipeline
from src.pipeline.prediction import Predictor
from src.config.configuration import ConfigurationManager



STAGE_NAME = "Training stage"


try:
    #logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    #obj = DataIngestionTrainingPipeline()
    #obj.main()
    #logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    #logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    #obj = TrainingPipeline()
    #obj.main()
    #logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    #logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    #obj = Evalutionipeline()
    #obj.main()
    #logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    config = ConfigurationManager()
    predcition_config = config.get_prediction_config()
    predict = Predictor(predcition_config)
    print(predict.make_prediction("test.jpeg"))
except Exception as e:
    logger.exception(e)
    raise e