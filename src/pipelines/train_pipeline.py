import sys
import os 
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_array, test_array,_=data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer=ModelTrainer()

    model_report, best_model=modeltrainer.initiate_model_selection(train_array, test_array)
    logging.info(model_report, best_model)
    logging.info('hyper paremeter tuning')
    modeltrainer.initiate_model_hyperparameter_tuning(train_array, test_array,best_model)
    logging.info('model file is saved')
