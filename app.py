from src.mlproject.logger import logging
from src.mlproject.exception import customexception
from src.mlproject.components.data_ingestion import DataIngestion,DataIngestionConfig
import sys




if __name__ == '__main__':
    logging.info(' execution has started')

    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        raise customexception(e,sys)
