from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from src.logger import logging
from src.exception import CustomException
import sys


if __name__ == "__main__":
    try:
        logging.info("Starting the training pipeline")
        obj = DataIngestion()
        train_data,test_data = obj.initiate_data_ingestion()

        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

        modeltrainer=ModelTrainer()
        print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
        logging.info("Training pipeline completed successfully")
    except Exception as e:
        logging.error(f"Error occurred in the training pipeline: {e}")
        raise CustomException(e,sys)