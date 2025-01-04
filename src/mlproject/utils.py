import os
import sys
from src.mlproject.exception import customexception
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score
from datetime import datetime

load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
database = os.getenv('database')

def read_sql_data():
   logging.info('Reading SQL has been started')
   try:
      mydb = pymysql.connect(
         host = host,
         user= user,
         password= password,
         db = database
      )
      logging.info('Connection Established',mydb)
      df = pd.read_sql_query('select * from loan_train',mydb)
      print(df.head())
      return df
   except Exception as e:
      raise customexception(e,sys)

def save_obj(filepath,obj):
   try:
      dir_path = os.path.dirname(filepath)
      os.makedirs(dir_path,exist_ok= True)
      with open(filepath,'wb') as file_obj:
         pickle.dump(obj,file_obj)

   except Exception as e:
      raise customexception(e,sys)

def evaluate_model(xtrain,ytrain,xtest,ytest,models):
   try:
      model_list = {'Models':[]}
      scores = {
         'f1score':[],
         'roc_auc':[]
         }
      for name,model in models.items():
         model.fit(xtrain , ytrain)
         yhat = model.predict(xtest)
         model_list['Models'].append(name)
         metrics = f1_score(ytest,yhat) , roc_auc_score(ytest,yhat)
         for key,value in zip(scores.keys(),metrics):
            scores[key].append(value)
      model_report = pd.concat([pd.DataFrame(model_list),pd.DataFrame(scores)],axis = 1)
      return model_report
   except Exception as e:
      raise customexception(e,sys)
   
def load_obj(filepath):
   try:
      with open(filepath,'rb') as file_obj:
         obj = pickle.load(file_obj)
      return obj
   except Exception as e:
      raise customexception(e,sys)
   
def report(df, report_folder='report'):
    
    if not os.path.exists(report_folder):
        os.makedirs(report_folder)
    
    
    date_str = datetime.now().strftime('%d-%m-%Y')
    filename = f'eval_report_{date_str}.csv'
    filepath = os.path.join(report_folder, filename)
    
    df.to_csv(filepath, index=False)
    print(f'DataFrame saved to {filepath}')   