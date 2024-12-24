import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.mlproject.exception import customexception
from src.mlproject.logger import logging
from src.mlproject.utils import save_obj
import os

@dataclass
class DataTransformConfig:
    preprocess_obj_file_path = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformConfig()

    def get_data_transformer_object(self,x):

        try:
            
            numeric_columns = x.select_dtypes(exclude='object').columns
            cat_columns = x.select_dtypes(include='object').columns
            num_pipline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encoder',OneHotEncoder())
            ])

            logging.info(f'Numerical Columns : {numeric_columns}')
            logging.info(f'Numerical Columns : {cat_columns}')

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipline,numeric_columns),
                ('cat_pipeline',cat_pipeline,cat_columns)
            ])
            return preprocessor

        except Exception as e:
            raise customexception(e,sys)
    
    def initiate_data_transform(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('reading train and test data')

            input_train_features = train_df.drop(['loan_status'],axis = 1).set_index('id')
            target_train_feature = train_df.set_index('id')['loan_status']
            
            input_test_features = test_df.drop(['loan_status'],axis = 1).set_index('id')
            target_test_feature = test_df.set_index('id')['loan_status']

            logging.info('Applying transformation to input train and test data')

            preprocessor_obj = self.get_data_transformer_object(input_train_features)
            input_feature_train_arr = preprocessor_obj.fit_transform(input_train_features)
            input_feature_test_arr = preprocessor_obj.transform(input_test_features)
            train_arr = np.c_[input_feature_train_arr,np.array(target_train_feature)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_test_feature)]
            logging.info('Saved preprocessing object')
            save_obj(filepath = self.transformation_config.preprocess_obj_file_path , obj = preprocessor_obj)
            return (train_arr,test_arr,self.transformation_config.preprocess_obj_file_path)

        except Exception as e:
            customexception(e,sys)