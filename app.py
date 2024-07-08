from src.Disease_classification.logger import logging

from src.Disease_classification.components.data_ingestion import DataIngestion,DataIngestionConfig


if __name__=="__main__":
    a,b=DataIngestion().initiate_data_ingestion()
    print(a,b)