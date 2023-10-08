import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

models = {
    "Random Forest": RandomForestRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBRegressor": XGBRegressor(),
    "AdaBoost Regressor": AdaBoostRegressor()}
params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }}


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_selection(self,train_array,test_array):
        try:
            logging.info("Spliting train and test")
            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],
                                          test_array[:,:-1],test_array[:,-1])
            
             
            model_file,model_report=evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test
                                             ,models=models)
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            logging.info("model report is",model_report)
            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            

            ## To get best model name from dict

            if best_model_score<0.6:
                raise CustomException("No best model found")
            else:
                logging.info(f"Best found model on both training and testing dataset")

            return model_report, best_model_name
        except Exception as e:
            raise CustomException(e,sys)     
    def initiate_model_hyperparameter_tuning(self,train_array,test_array, best_model_name):
        try:
            logging.info("Spliting train and test")
            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],
                                          test_array[:,:-1],test_array[:,-1])
            
            final_model={best_model_name:models[best_model_name]}
            
            model_file,model_report_final=evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test
                                             ,models=final_model, param=params)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model_file
            )

            predicted=model_file.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return print("R2score is",r2_square)
     

            
        except Exception as e:
            raise CustomException(e,sys)

            

            