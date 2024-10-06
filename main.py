from src import logger
from src.pipeline.training import DataIngestionPipeline, TrainingPipeline, Evalutionipeline
from src.pipeline.prediction import Predictor
from src.config.configuration import ConfigurationManager

STAGE_NAMES = {
    "data_ingestion": "Data Ingestion Stage",
    "training": "Training Stage",
    "evaluation": "Evaluation Stage"
}

def run_pipeline(stage_name, pipeline_obj):
    """
    Run the specified pipeline stage with logging.

    Args:
        stage_name (str): The name of the stage being executed.
        pipeline_obj: The pipeline object to execute.
    """
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        pipeline_obj.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"Error in {stage_name}: {str(e)}")
        raise e

def main():
    try:
        # Data Ingestion Stage
        data_ingestion_pipeline = DataIngestionPipeline()
        run_pipeline(STAGE_NAMES["data_ingestion"], data_ingestion_pipeline)

        # Training Stage
        training_pipeline = TrainingPipeline()
        run_pipeline(STAGE_NAMES["training"], training_pipeline)

        # Evaluation Stage
        evaluation_pipeline = Evalutionipeline()
        run_pipeline(STAGE_NAMES["evaluation"], evaluation_pipeline)

        # Prediction
        config = ConfigurationManager()
        prediction_config = config.get_prediction_config()
        predictor = Predictor(prediction_config)

        # Assuming the image path is correct and the model has been trained
        prediction_result = predictor.make_prediction("test.jpeg")
        logger.info(f"Prediction Result: {prediction_result}")

    except Exception as e:
        logger.exception("An error occurred in the pipeline execution.")
        raise e

if __name__ == "__main__":
    main()
