import os
import sys
from src.Disease_classification.exception import CustomException
from src.Disease_classification.logger import logging 
from src.Disease_classification.utils import read_sql_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pymysql
from sklearn.preprocessing import LabelEncoder 
from src.Disease_classification.utils import save_object
import pickle
import numpy as np
import joblib

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')
    label_preproxcessor:str=os.path.join('artifacts','Dise.jobllib')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            ##reading the data from mysql
            df=read_sql_data()
            logging.info("Reading completed mysql database")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.label_preproxcessor),exist_ok=True)
            lr=LabelEncoder()
            df['Disease']=lr.fit_transform(df['Disease'])
            joblib.dump(lr,self.ingestion_config.label_preproxcessor)
            logging.info("Dumped")
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42,shuffle=True)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            # save_object(self.ingestion_config.label_preproxcessor,lr)
            logging.info("Data Ingestion is completed")
           
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path


            )


        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    a,b=DataIngestion().initiate_data_ingestion()