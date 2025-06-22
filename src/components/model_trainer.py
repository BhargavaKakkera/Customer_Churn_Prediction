import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.combine import SMOTEENN

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("saved_model", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Apply SMOTEENN to handle class imbalance
            logging.info("Applying SMOTEENN to training data")
            smote_enn = SMOTEENN()
            X_train, y_train = smote_enn.fit_resample(X_train, y_train)

            # Base models
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(probability=True),
                "Neural Network": MLPClassifier(),
                "XGBoost": XGBClassifier(),
                "Naive Bayes": GaussianNB()
            }

            # Add Voting and Stacking classifiers
            base_estimators = [
                ('lr', LogisticRegression()),
                ('rf', RandomForestClassifier()),
                ('nb', GaussianNB()),

            ]

            models["Voting Classifier"] = VotingClassifier(estimators=base_estimators, voting='soft')
            models["Stacking Classifier"] = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression())

            params = {
                "Logistic Regression": {"C": [0.1, 1.0, 10]},
                "Random Forest": {"n_estimators": [50, 100], "max_depth": [None, 10, 20], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]},
                "Decision Tree": {},
                "KNN": {},
                "SVM": {},
                "Neural Network": {"max_iter": [500,1000]},
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "subsample": [0.8, 1],
                    "colsample_bytree": [0.8, 1],
                },
                "Naive Bayes": {},
                "Voting Classifier": {},
                "Stacking Classifier": {}
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)


            eligible_models = [model for model in model_report
                                if model_report[model]["F1 Score"] > 0.5
                                and model_report[model]["Recall"] > 0.7
                                and model_report[model]["Precision"] > 0.5]

            # Select best model based on F1 Score
            best_model_name = max(eligible_models, key=lambda x: model_report[x]["F1 Score"])
            best_model = model_report[best_model_name]["Best Estimator"]
            best_f1 = model_report[best_model_name]["F1 Score"]
            best_recall = model_report[best_model_name]["Recall"]
            best_precision = model_report[best_model_name]["Precision"]


            for model_name, result in model_report.items():

                 
                logging.info(f"{model_name} - F1: {result['F1 Score']}, Recall: {result['Recall']}, Precision: {result['Precision']}")



            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            return best_f1

        except Exception as e:
            raise CustomException(e, sys)
