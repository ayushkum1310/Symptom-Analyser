from src.Disease_classification.logger import logging

from src.Disease_classification.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.Disease_classification.components.data_transformation import DataTransformation

if __name__=="__main__":
    a,b=DataIngestion().initiate_data_ingestion()
    a,b,c=DataTransformation().initiate_data_transformation(a,b)
    print(a.shape)
    print(b.shape)
    print(c)