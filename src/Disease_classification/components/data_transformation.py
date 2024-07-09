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
                'Difficulty_Breathing',
                'Gender',
                'Blood_Pressure',
                'Cholesterol_Level',
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
                remainder='passthrough'  # This will keep the 'Disease' column after LabelEncoder
            )

            return final_preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Reading data frames")

            # Initialize the preprocessor
            preprocessor = self.data_transformation_obj()
            
            # Apply LabelEncoder to 'Disease' and 'Outcome Variable'
            

            input_train = train_data.drop(columns=['Outcome Variable'], axis=1)
            target_train = train_data['Outcome Variable'].map({'Positive':1,'Negative':0})
            input_test = test_data.drop(columns=['Outcome Variable'], axis=1)
            target_test = test_data['Outcome Variable'].map({'Positive':1,'Negative':0})

            # Fit and transform the training data
            train_ = preprocessor.fit_transform(input_train)
            # Transform the test data
            test_ = preprocessor.transform(input_test)

            # Concatenate the 'Disease' column back to the transformed data
            train_arr = np.c_[train_,  np.array(target_train)]
            test_arr = np.c_[test_,  np.array(target_test)]
            # pd.DataFrame(train_arr).to_csv("nhipta.csv")
            # pd.DataFrame(train_arr).to_csv('Hale1.csv')
            save_object(self.data_trans_config.preprocessor_obj_file, preprocessor)
            
            return train_arr, test_arr, self.data_trans_config.preprocessor_obj_file

        except Exception as e:
            raise CustomException(e, sys)

# Example usage
if __name__ == "__main__":
    train_path = "train.csv"
    test_path = "test.csv"
    data_transformer = DataTransformation()
    train_arr, test_arr, preprocessor_file = data_transformer.initiate_data_transformation(train_path, test_path)
    print("Train array:")
    print(train_arr)
    print("\nTest array:")
    print(test_arr)
