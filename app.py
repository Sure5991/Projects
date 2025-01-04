from src.mlproject.logger import logging
from src.mlproject.exception import customexception
from src.mlproject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.mlproject.components.data_transform import DataTransformation,DataTransformConfig
from src.mlproject.components.model_trainer import ModelTrainer,ModelTrainerConfig
import sys




if __name__ == '__main__':
    logging.info(' execution has started')

    try:
        # Data Ingestion
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        #data_transform_config = data_transform_config()
        data_tranformation = DataTransformation()
        train_arr,test_arr,_ = data_tranformation.initiate_data_transform(train_path, test_path)

        # Model training
        model_trainer = ModelTrainer()
        _,_,score = model_trainer.initiate_model_trainer(train_arr,test_arr)
        print(score)
        
    except Exception as e:
        raise customexception(e,sys)
