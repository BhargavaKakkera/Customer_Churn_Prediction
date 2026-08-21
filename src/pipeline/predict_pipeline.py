import sys
import os
import warnings
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

warnings.filterwarnings("ignore")

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "saved_model", "model.pkl")
            preprocessor_path = os.path.join(base_dir, "saved_model", "preprocessor.pkl")

            model_data = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            model = model_data["model"]
            #threshold = model_data.get("threshold", 0.5)

            data_scaled = preprocessor.transform(features)
            
            # Predict churn probability using model
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(data_scaled)[:, 1][0]
            else:
                proba = float(model.predict(data_scaled)[0])
                
            return float(proba)

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 SeniorCitizen: str,
                 Partner: str,
                 Dependents: str,
                 PhoneService: str,
                 MultipleLines: str,
                 InternetService: str,
                 OnlineSecurity: str,
                 OnlineBackup: str,
                 DeviceProtection: str,
                 TechSupport: str,
                 StreamingTV: str,
                 StreamingMovies: str,
                 Contract: str,
                 PaperlessBilling: str,
                 PaymentMethod: str,
                 tenure: float,
                 MonthlyCharges: float,
                 TotalCharges: float):

        self.gender = gender
        # Map "Yes"/"No" back to numeric 1/0 as in original dataset
        self.SeniorCitizen = 1 if SeniorCitizen == "Yes" else 0
        self.Partner = Partner
        self.Dependents = Dependents
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.tenure = tenure
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "tenure": [self.tenure],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
