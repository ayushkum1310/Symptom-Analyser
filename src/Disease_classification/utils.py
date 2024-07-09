import os
import sys
import pandas as pd
import pymysql
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from src.Disease_classification.exception import CustomException
from src.Disease_classification.logger import logging
import pickle

load_dotenv()
db = os.getenv('db')
host = os.getenv('host')
user = os.getenv('user')
pas = os.getenv('pas')


def read_sql_data():
    try:
        logging.info("Reading from SQL")
        mydb = pymysql.connect(
            database=db,
            host=host,
            user=user,
            password=pas
        )
        logging.info("Fetched")
        df = pd.read_sql_query('SELECT * FROM disease.disease_symptom_and_patient_profile_dataset', mydb)
        logging.info("Loaded")
        return df
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            # para=param[list(models.keys())[i]]

            # gs = GridSearchCV(model,para,cv=3)
            # gs.fit(X_train,y_train)

            # model.set_params(**gs.best_params_)
            # model.fit(X_train,y_train)

            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)