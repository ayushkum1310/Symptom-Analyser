from src.Disease_classification.logger import logging

from src.Disease_classification.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.Disease_classification.components.data_transformation import DataTransformation
from src.Disease_classification.components.model_tranier import ModelTrainerConfig,ModelTrainer
if __name__=="__main__":
    a,b=DataIngestion().initiate_data_ingestion()
    a,b,_=DataTransformation().initiate_data_transformation(a,b)
    c=ModelTrainer().initiate_model_trainer(a,b)
    
    
    print(a.shape)
    print(b.shape)
    print(c)