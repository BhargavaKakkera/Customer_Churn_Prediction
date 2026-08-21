import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

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

            # Calculate scale_pos_weight for XGBoost and LightGBM imbalance handling
            num_neg = np.sum(y_train == 0)
            num_pos = np.sum(y_train == 1)
            scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
            logging.info(f"Imbalance ratio calculated on training set: {scale_pos_weight:.4f}")

            # Base models with cost-sensitive learning to handle class imbalance
            models = {
                "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=10000),
                "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
                "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(probability=True, class_weight='balanced', random_state=42),
                "Neural Network": MLPClassifier(random_state=42),
                "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42),
                "LightGBM": LGBMClassifier(scale_pos_weight=scale_pos_weight, verbose=-1, random_state=42),
                "Naive Bayes": GaussianNB()
            }

            # Hyperparameter tuning grids
            params = {
                "Logistic Regression": {
                    "C": [0.1, 1.0, 10],
                    "solver": ['liblinear', 'saga']
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "max_features": ["sqrt", "log2"]
                },
                "Decision Tree": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_leaf": [1, 5, 10]
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                },
                "SVM": {
                    "C": [0.1, 1.0, 10],
                    "kernel": ["linear", "rbf"]
                },
                "Neural Network": {
                    "max_iter": [500, 1000],
                    "alpha": [0.0001, 0.001, 0.01]
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                },
                "LightGBM": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "num_leaves": [7, 15, 31],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                },
                "Naive Bayes": {},
                "Voting Classifier": {},
                "Stacking Classifier": {}
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params, target_recall=0.80)

            # Log all evaluated model metrics
            for model_name, result in model_report.items():
                logging.info(f"{model_name} - F1: {result['F1 Score']:.4f}, Recall: {result['Recall']:.4f}, Precision: {result['Precision']:.4f}, Optimal Threshold: {result['Threshold']:.4f}")

            # Define eligibility filters
            eligible_models = [model for model in model_report
                                if model_report[model]["F1 Score"] > 0.5
                                and model_report[model]["Recall"] >= 0.75
                                and model_report[model]["Precision"] > 0.45]

            # Select best model based on F1 Score (with fallback if strict thresholds are not met)
            if not eligible_models:
                logging.warning("No models met the strict eligibility criteria. Falling back to the model with the highest F1 Score.")
                best_model_name = max(model_report, key=lambda x: model_report[x]["F1 Score"])
            else:
                best_model_name = max(eligible_models, key=lambda x: model_report[x]["F1 Score"])

            best_model = model_report[best_model_name]["Best Estimator"]
            best_f1 = model_report[best_model_name]["F1 Score"]
            best_recall = model_report[best_model_name]["Recall"]
            best_precision = model_report[best_model_name]["Precision"]
            best_threshold = model_report[best_model_name]["Threshold"]

            logging.info(f"Selected Best Model: {best_model_name} with F1: {best_f1:.4f}, Threshold: {best_threshold:.4f}")

            # Save best model along with its optimal classification threshold
            model_data = {
                "model": best_model,
                "threshold": best_threshold
            }
            save_object(self.model_trainer_config.trained_model_file_path, model_data)

            return best_f1

        except Exception as e:
            raise CustomException(e, sys)
