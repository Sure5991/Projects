from src.mlproject.logger import logging
from src.mlproject.exception import customexception
from src.mlproject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.mlproject.components.data_transform import DataTransformation,DataTransformConfig
import sys




if __name__ == '__main__':
    logging.info(' execution has started')

    try:
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        #data_transform_config = data_transform_config()
        data_tranformation = DataTransformation()
        data_tranformation.initiate_data_transform(train_path, test_path)

    except Exception as e:
        raise customexception(e,sys)
