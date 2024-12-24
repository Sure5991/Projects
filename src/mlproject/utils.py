import os
import sys
from src.mlproject.exception import customexception
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
import numpy as np



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
