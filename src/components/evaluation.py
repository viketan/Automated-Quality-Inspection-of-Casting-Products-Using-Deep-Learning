import os
import mlflow
import tensorflow as tf
from src import logger
from src.entity import EvaluationConfig, TrainingParam
import mlflow.tensorflow
from dotenv import load_dotenv
import dagshub
dagshub.init(repo_owner='viketanrevankar108', repo_name='Automated-Quality-Inspection-of-Casting-Products-Using-Deep-Learning', mlflow=True)
load_dotenv()


class Evaluation():
    """
    A class to handle the evaluation of the trained model on the test dataset.
    """

    def __init__(self, config: EvaluationConfig, param: TrainingParam):
        """
        Initializes the Evaluation class with configurations.
        
        Args:
            config (EvaluationConfig): Configuration object containing paths and filenames.
        """
        self.config = config
        self.param = param

    def load_model(self):
        """
        Loads the saved model from the specified path.
        
        Returns:
            model: The loaded Keras model.
        """
        try:
            model_path = self.config.model_path
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error in loading model: {e}")
            raise e

    def evaluate_model(self):
        """
        Evaluates the model on the test dataset and logs the metrics to MLflow.
        
        Raises:
            Exception: If there is an error during the evaluation process.
        """
        try:
            # Set up MLflow with DagsHub tracking URI
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            #mlflow.set_experiment(self.config.experiment_name)

            # Start MLflow run for evaluation
            with mlflow.start_run():
                # Load the test dataset
                test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                    self.config.data_dir,
                    shuffle=True,
                    image_size=(self.param.img_size, self.param.img_size),
                    batch_size=self.param.batch_size
                )

                # Load the saved model
                model = self.load_model()

                # Evaluate the model
                loss, accuracy = model.evaluate(test_dataset)

                # Log metrics to MLflow
                mlflow.log_metric("loss", loss)
                mlflow.log_metric("accuracy", accuracy)
                # Log model parameters (you can log other parameters as well)
                mlflow.log_param("optimizer", self.param.optimizer)
                mlflow.log_param("epochs", self.param.epochs)
                mlflow.log_param("activation function hidden", self.param.activation_function_hidden)
                mlflow.log_param("activation function output", self.param.activation_function_output)
                mlflow.log_param("loss", self.param.loss)
                mlflow.log_param("bact size", self.param.batch_size)

                # Save the model in MLflow's format
                mlflow.tensorflow.log_model(model, "model")

                logger.info(f"Model evaluated with loss: {loss}, accuracy: {accuracy}")

        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise e
