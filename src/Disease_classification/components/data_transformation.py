import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from src.Disease_classification.logger import logging
from src.Disease_classification.exception import CustomException
from src.Disease_classification.utils import save_object


from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts', 'preprocessor.pkl')



class DataTransformation:
    def __init__(self):
        self.data_trans_config = DataTransformationConfig()

    def data_transformation_obj(self):
        # Responsible for data transformation
        try:
            num = ['Age']
            one_hot_cols = [
                'Fever',
                'Cough',
                'Fatigue',
                'Difficulty Breathing',
                'Gender',
                'Blood Pressure',
                'Cholesterol Level',
                'Outcome Variable'
            ]
            

            pipe1 = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", MinMaxScaler())
            ])

            pipe2 = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("encoder", OneHotEncoder(handle_unknown='ignore'))
            ])

           

            final_preprocessor = ColumnTransformer(
                transformers=[
                    ('num', pipe1, num),
                    ('cat', pipe2, one_hot_cols),
                    
                ],
                remainder='passthrough'
            )

            return final_preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Reading data frames")

            preprocessor = self.data_transformation_obj()

            train_ = preprocessor.fit_transform(train_data)
            test_ = preprocessor.transform(test_data)

            save_object(self.data_trans_config.preprocessor_obj_file, preprocessor)

            return (train_, test_, self.data_trans_config.preprocessor_obj_file)

        except Exception as e:
            raise CustomException(e, sys)
