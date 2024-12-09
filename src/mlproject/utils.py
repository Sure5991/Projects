import os
import sys
from src.mlproject.exception import customexception
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql


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
