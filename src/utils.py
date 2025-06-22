import os
import sys
import dill as pickle
from sklearn.metrics import f1_score, recall_score, accuracy_score,precision_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

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

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name in models:
            model = models[model_name]
            hyperparams = param.get(model_name, {})

            gs = GridSearchCV(model, hyperparams, cv=3, scoring='f1', n_jobs=-1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            y_test_pred = best_model.predict(X_test)

            f1 = f1_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)

            report[model_name] = {
                "F1 Score": f1,
                "Recall": recall,
                "Accuracy": accuracy,
                "Best Estimator": best_model,
                "Precision": precision
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
