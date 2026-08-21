import os
import sys
import numpy as np
import dill as pickle
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging  # Added logger import

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param, target_recall=0.80):
    try:
        report = {}

        for model_name in models:
            model = models[model_name]
            hyperparams = param.get(model_name, {})

            # Informative log to track execution progress
            logging.info(f" >>> [TRAINING START] GridSearch tuning for: {model_name}...")
            
            gs = GridSearchCV(model, hyperparams, cv=5, scoring='f1', n_jobs=-1)
            gs.fit(X_train, y_train)
            
            logging.info(f" <<< [TRAINING END] GridSearch tuning completed for: {model_name}")

            best_model = gs.best_estimator_

            # Optimal threshold search on training predictions targeting high recall (>= target_recall)
            best_threshold = 0.5
            if hasattr(best_model, "predict_proba"):
                y_train_proba = best_model.predict_proba(X_train)[:, 1]
                precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_proba)
                if len(thresholds) > 0:
                    # Filter thresholds where training recall satisfies target_recall
                    valid_mask = recalls[:-1] >= target_recall
                    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
                    if np.any(valid_mask):
                        valid_f1 = np.where(valid_mask, f1_scores, -1)
                        best_idx = np.argmax(valid_f1)
                    else:
                        best_idx = np.argmax(f1_scores)
                    best_threshold = thresholds[best_idx]

            # Generate predictions on test data using optimal threshold
            if hasattr(best_model, "predict_proba"):
                y_test_proba = best_model.predict_proba(X_test)[:, 1]
                y_test_pred = (y_test_proba >= best_threshold).astype(int)
            else:
                y_test_pred = best_model.predict(X_test)
                best_threshold = 0.5

            f1 = f1_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)

            report[model_name] = {
                "F1 Score": f1,
                "Recall": recall,
                "Accuracy": accuracy,
                "Best Estimator": best_model,
                "Precision": precision,
                "Threshold": best_threshold
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
