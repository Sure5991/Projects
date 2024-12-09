import os
import sys
from src.mlproject.exception import customexception
from src.mlproject.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.mlproject.utils import read_sql_data
from sklearn.model_selection import train_test_split



@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifact','train.csv')
    test_data_path:str = os.path.join('artifact','test.csv')
    raw_path:str = os.path.join('artifact','raw.csv')

class DataIngestion:
    def __init__ (self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            df = read_sql_data()            
            logging.info('reading from mysql database')
            os.makedirs(os.path.dirname(DataIngestionConfig.train_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_path,index= False, header= True)
            trainset,testset = train_test_split(df,test_size=0.2,random_state=42, stratify= df.loan_status) # loan status is class label
            trainset.to_csv(self.ingestion_config.train_data_path,index= False, header= True)
            testset.to_csv(self.ingestion_config.test_data_path,index= False, header= True)
            logging.info('data ingestion is completed')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise customexception(e,sys)
