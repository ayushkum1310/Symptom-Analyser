import os
import sys
from src.Disease_classification.exception import CustomException
from src.Disease_classification.logger import logging 
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pymysql

import pickle
import numpy as np


load_dotenv()
db=os.getenv('db')
host=os.getenv("host")
user=os.getenv("user")
pas=os.getenv("pas")


def read_sql_data():
    try :
        logging.info("Reading from sql")
        mydb=pymysql.connect(
            database=db,
            host=host,
            user=user,
            password=pas
        )
        logging.info("Fetched")
        df=pd.read_sql_query('SELECT * FROM disease.disease_symptom_and_patient_profile_dataset',mydb)
        logging.info("Loaded")
        return df
    except Exception as e:
        raise CustomException(e,sys)
    
