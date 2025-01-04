import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
import xgboost as xgb
from src.mlproject.exception import customexception
from src.mlproject.logger import logging
from src.mlproject.utils import save_obj,evaluate_model,report



@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifact','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spilting train and test input data")
            xtrain,ytrain,xtest,ytest = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models = {
                'SVC':SVC(class_weight = 'balanced'),
                'DecisionTreeClassifier':DecisionTreeClassifier(class_weight = 'balanced'),
                'RandomForestClassifier':RandomForestClassifier(class_weight = 'balanced'),
                'GradientBoostingClassifier':GradientBoostingClassifier(),
                'KNeighborsClassifier':KNeighborsClassifier(),
                'LogisticRegression':LogisticRegression(class_weight = 'balanced'),
                'GaussianNB':GaussianNB(),
                'CatBoostClassifier':CatBoostClassifier(verbose=False),
                'XGBClassifier':xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                'AdaBoostClassifierda':AdaBoostClassifier()
                }
            
            model_report = evaluate_model(xtrain,ytrain,xtest,ytest,models)
            best_auroc_score = model_report.roc_auc.max()
            best_model_name = model_report.iloc[model_report.roc_auc.idxmax(),0]
            best_model = models[best_model_name]
            report(model_report)
            save_obj(filepath= self.model_trainer_config.trained_model_path,obj=best_model)
            if (best_auroc_score < 0.75):
                raise customexception('No Best model found')
            logging.info('Best found model for both training and testing')
            return best_model,best_auroc_score,best_model_name
        
        except Exception as e:
            raise customexception(e,sys)

