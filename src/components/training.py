import os
from src import logger
from src.utils.common import get_size
from src.entity import TrainingConfig, TrainingParam
from tensorflow.keras import models, layers
import tensorflow as tf
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class Training():
    """
    A class to handle the entire training pipeline, including data preparation, 
    model building, and training initiation for a Convolutional Neural Network (CNN).
    """
    
    def __init__(self, config: TrainingConfig, param: TrainingParam):
        """
        Initializes the Training class with configurations and parameters.
        
        Args:
            config (TrainingConfig): Configuration object containing paths and filenames.
            param (TrainingParam): Parameters object for the model such as image size, batch size, etc.
        """
        self.config = config
        self.param = param
    
    def prepare_data(self):
        """
        Prepares the training and validation datasets by loading the data from the given directory,
        applying image resizing, shuffling, and prefetching for better performance.
        
        Returns:
            train_dataset: Training dataset (80% of the total data).
            validate_dataset: Validation dataset (20% of the total data).
            class_name: List of class names in the dataset.
            
        Raises:
            Exception: If there is an error during dataset extraction or processing.
        """
        try:
            data_dir = self.config.data_dir  # Directory where data is stored
            
            # Load the dataset from the directory with image resizing and shuffling
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                data_dir,
                shuffle=True,
                image_size=(self.param.img_size, self.param.img_size),
                batch_size=self.param.batch_size
            )
            
            # Extract class names
            class_name = dataset.class_names
            
            # Split the dataset into 80% training and 20% validation
            train_dataset = dataset.take(int(len(dataset) * 0.8))
            validate_dataset = dataset.skip(int(len(dataset) * 0.8))
            
            # Prefetch and shuffle for performance optimization
            train_dataset = train_dataset.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            validate_dataset = validate_dataset.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            
            return train_dataset, validate_dataset, class_name
        
        except Exception as e:
            logger.error(f"Error in extracting file: {e}")
            raise e
    
    def base_model(self):
        """
        Builds and returns the base Convolutional Neural Network (CNN) model.
        
        The model consists of multiple convolutional layers with pooling, data augmentation layers,
        and fully connected layers for classification.
        
        Returns:
            model (tf.keras.Model): The constructed CNN model.
            
        Raises:
            Exception: If there is an error during model creation.
        """
        try:
            # Prepare data to get class names
            _, _, class_name = self.prepare_data()
            
            # Resizing and rescaling layers
            resize_rescale = tf.keras.Sequential([
                layers.Resizing(self.param.img_size, self.param.img_size),
                layers.Rescaling(1.0 / 255)
            ]) 
            
            # Data augmentation layers
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
                layers.RandomZoom(0.2),
                layers.RandomContrast(0.2),
            ])
            
            # Input shape and number of classes
            input_shape = (self.param.img_size, self.param.img_size, self.param.channel)
            n_classes = len(class_name)
            
            # Define the CNN model architecture
            model = models.Sequential([
                layers.Input(shape=input_shape),  # Input layer
                resize_rescale,  # Resizing and rescaling
                data_augmentation,  # Data augmentation

                # Convolutional and pooling layers
                layers.Conv2D(32, (3, 3), activation=self.param.activation_function_hidden),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(64, (3, 3), activation=self.param.activation_function_hidden),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(64, (3, 3), activation=self.param.activation_function_hidden),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(128, (3, 3), activation=self.param.activation_function_hidden),
                layers.MaxPooling2D((2, 2)),
                
                layers.Flatten(),  # Flatten layer
                
                # Dropout to avoid overfitting
                layers.Dropout(0.2),
                
                # Fully connected layers
                layers.Dense(64, activation=self.param.activation_function_hidden),
                layers.Dense(n_classes, activation=self.param.activation_function_output)
            ])
            
            return model
        
        except Exception as e:
            logger.error(f"Error in creating model: {e}")
            raise e
        
    def initiate_training(self):
        """
        Initiates the training process by preparing data, compiling the model, 
        and starting the model training process. The trained model is saved after training.
        
        Raises:
            Exception: If there is an error during the training process.
        """
        try:
            # Prepare the training and validation datasets
            train_dataset, validate_dataset, _ = self.prepare_data()
            
            # Get the model
            model = self.base_model()
            
            # Compile the model with optimizer, loss, and metrics
            model.compile(
                optimizer=self.param.optimizer,
                loss=self.param.loss,
                metrics=[self.param.metrics]
            )
            
            # Train the model
            history = model.fit(
                train_dataset,
                validation_data=validate_dataset,
                epochs=self.param.epochs  # Use the number of epochs from param
            )
            
            # Save the model to the specified path
            model_filename = self.config.model_filename
            model_path = os.path.join(self.config.root_dir, model_filename)
            model.save(model_path)
            
            logger.info(f"Model saved at {model_path}")
        
        except Exception as e:
            logger.error(f"Error in training: {e}")
            raise e
