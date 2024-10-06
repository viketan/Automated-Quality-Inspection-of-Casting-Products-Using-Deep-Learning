import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from src import logger
from src.entity import PredictionConfig

class Predictor:
    def __init__(self, config: PredictionConfig):
        self.config = config

    def load_model(self):
        """
        Load the trained model from the saved model file.
        """
        try:
            model_path = self.config.model_path
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def preprocess_image(self, img_path):
        """
        Preprocess the image for prediction. Resizes and scales the image.
        """
        try:
            img = image.load_img(img_path, target_size=(self.config.img_size, self.config.img_size))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension
            img_array = img_array / 255.0  # Normalizing to [0,1] scale
            logger.info(f"Image preprocessed successfully from {img_path}")
            return img_array
        except Exception as e:
            logger.error(f"Error in preprocessing image: {e}")
            raise e

    def make_prediction(self, img_path):
        """
        Load model, preprocess image, and make a prediction.
        """
        try:
            model = self.load_model()
            preprocessed_img = self.preprocess_image(img_path)

            # Perform prediction
            predictions = model.predict(preprocessed_img)
            predicted_class = np.argmax(predictions, axis=1)
            
            class_names = self.config.class_names
            predicted_label = class_names[predicted_class[0]]

            logger.info(f"Prediction made successfully. Predicted class: {predicted_label}")
            return predicted_label
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise e
