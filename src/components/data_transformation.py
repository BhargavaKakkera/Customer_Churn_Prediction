import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns=None):
        self.categorical_columns = categorical_columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            X = X.copy()
            # Ensure categorical columns are cast to string to prevent type issues in OHE
            for col in self.categorical_columns:
                if col in X.columns:
                    X[col] = X[col].astype(str)

            # 1. NumServices: count of active service subscriptions
            service_cols = [
                'PhoneService', 'MultipleLines', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies'
            ]
            existing_service_cols = [col for col in service_cols if col in X.columns]
            if existing_service_cols:
                X['NumServices'] = (X[existing_service_cols] == 'Yes').sum(axis=1)
            else:
                X['NumServices'] = 0
            
            # 2. IsNewCustomer: tenure <= 6
            if 'tenure' in X.columns:
                X['IsNewCustomer'] = (X['tenure'] <= 6).astype(int)
            else:
                X['IsNewCustomer'] = 0
            
            # 3. AverageMonthlyCharge: TotalCharges / (tenure + 1)
            if 'TotalCharges' in X.columns and 'tenure' in X.columns:
                total_charges_numeric = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0)
                X['AverageMonthlyCharge'] = total_charges_numeric / (X['tenure'] + 1)
            else:
                X['AverageMonthlyCharge'] = 0.0
                
            return X
        except Exception as e:
            raise CustomException(e, sys)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('saved_model', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Inputs list (including engineered features which will be appended by FeatureEngineer)
            numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices", "IsNewCustomer", "AverageMonthlyCharge"]
            categorical_columns = [
                "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
                "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "PaymentMethod"
            ]
            ordinal_columns = ["Contract"]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            ord_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal_encoder", OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]])),
                ("scaler", StandardScaler())
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Ordinal columns: {ordinal_columns}")

            column_transformer = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns),
                ("ord_pipeline", ord_pipeline, ordinal_columns)
            ])

            # Preprocessor wraps feature engineering and column transformations
            preprocessor = Pipeline(steps=[
                ("feature_engineering", FeatureEngineer(categorical_columns=categorical_columns)),
                ("column_transformer", column_transformer)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Handle blank strings and convert numeric columns
            train_df.replace(" ", np.nan, inplace=True)
            test_df.replace(" ", np.nan, inplace=True)

            numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
            for col in numerical_columns:
                train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
                test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

            logging.info("Cleaned numerical columns and replaced blank spaces")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Churn"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name].map({"No": 0, "Yes": 1})

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name].map({"No": 0, "Yes": 1})

            logging.info("Applying preprocessing object on train and test data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object to disk")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
