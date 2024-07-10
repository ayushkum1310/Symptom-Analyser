import os
import sys 
from urllib.parse import urlparse
import mlflow
from sklearn.ensemble import RandomForestClassifier
# from catboost import CatBoostClassifier
import numpy as np
from src.Disease_classification.utils import save_object,evaluate_models
from dataclasses import dataclass
from src.Disease_classification.logger import logging
from src.Disease_classification.exception import CustomException
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import dagshub
dagshub.init(repo_owner='ayushkum1310', repo_name='Disease_classification', mlflow=True)

@dataclass
class ModelTrainerConfig:
    train_model_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self) :
        self.model_trainer_config_obj=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Model_training has started")
            # X_train, y_train, X_test, y_test=(train_data[:,:-1],train_data[:,-1],test_data[:,:-1],test_data[:,-1])
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={"random_forest":RandomForestClassifier()
                    }
            # model_params = {
            #         'random_forest': {
            #         'n_estimators': 100,            # Number of trees in the forest
            #         'criterion': 'gini',            # Function to measure the quality of a split
            #         'max_depth': None,              # Maximum depth of the tree
            #         'min_samples_split': 2,         # Minimum number of samples required to split an internal node
            #         'min_samples_leaf': 1,          # Minimum number of samples required to be at a leaf node
            #         'max_features': 'auto',         # Number of features to consider when looking for the best split
            #         'bootstrap': True,              # Whether bootstrap samples are used when building trees
            #         'oob_score': False,             # Whether to use out-of-bag samples to estimate the generalization accuracy
            #         'n_jobs': -1,                   # Number of jobs to run in parallel
            #         'random_state': 42,             # Seed used by the random number generator
            #         'verbose': 0                    # Controls the verbosity when fitting and predicting
            #                          }
                # 'catboost': {
                #     'iterations': 1000,             # Number of boosting iterations
                #     'learning_rate': 0.03,          # Step size shrinkage used in update to prevent overfitting
                    # 'depth': 6,                     # Depth of the tree
                    # 'l2_leaf_reg': 3,               # L2 regularization term on weights
                    # 'bootstrap_type': 'Bernoulli',  # Type of bootstrap
                    # 'subsample': 0.66,              # Subsample ratio of the training instance
                    # 'scale_pos_weight': 1,          # Balancing of positive and negative weights
                    # 'eval_metric': 'Accuracy',      # Metric used for evaluation
                    # 'random_seed': 42,              # Seed used by the random number generator
                    # 'use_best_model': True,         # Whether to use the best model
                    # 'verbose': 100                  # Controls the verbosity when fitting
                # }
            
            model_report=evaluate_models(X_train,y_train,X_test,y_test,models)
            best_model_score = max(sorted(model_report.values()))

             ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            save_object(self.model_trainer_config_obj.train_model_path,best_model)
            import dagshub
            dagshub.init(repo_owner='ayushkum1310', repo_name='Disease_classification', mlflow=True)


            model_names = list(models.keys())

            actual_model=""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            # best_params = params[actual_model]
            mlflow.set_registry_uri("https://dagshub.com/ayushkum1310/Disease_classification.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # mlflow

            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                a = accuracy_score(y_test, predicted_qualities)
                b= f1_score(y_test, predicted_qualities)
                c = recall_score(y_test, predicted_qualities)
                d = precision_score(y_test, predicted_qualities)
                

                # mlflow.log_params(best_params)

                mlflow.log_metric("Accuracy", a)
                mlflow.log_metric("f1", b)
                mlflow.log_metric("recal", c)
                mlflow.log_metric("precision",d)
                # mlflow.log_metric("r2", r2)
                # mlflow.log_metric("mae", mae)


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")
            p=best_model.predict(X_test)
            return accuracy_score(p,y_test)

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    a=np.load('artifacts/train_arr.npy')
    b=np.load('artifacts/test_arr.npy')
    a=ModelTrainer().initiate_model_trainer(a,b)
    
    # # print(a)