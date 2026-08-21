from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        form_data = request.form
        try:
            tenure_val = request.form.get('tenure')
            monthly_val = request.form.get('MonthlyCharges')
            total_val = request.form.get('TotalCharges')

            if tenure_val is None or str(tenure_val).strip() == '':
                raise ValueError("Tenure (in months) is required and must be a number.")
            if monthly_val is None or str(monthly_val).strip() == '':
                raise ValueError("Monthly Charges is required and must be a number.")
            if total_val is None or str(total_val).strip() == '':
                raise ValueError("Total Charges is required and must be a number.")

            try:
                tenure = float(tenure_val)
                MonthlyCharges = float(monthly_val)
                TotalCharges = float(total_val)
            except ValueError:
                raise ValueError("Tenure, Monthly Charges, and Total Charges must be valid numbers.")

            data = CustomData(
                gender=request.form.get('gender', 'Female'),
                SeniorCitizen=request.form.get('SeniorCitizen', 'No'),
                Partner=request.form.get('Partner', 'No'),
                Dependents=request.form.get('Dependents', 'No'),
                PhoneService=request.form.get('PhoneService', 'Yes'),
                MultipleLines=request.form.get('MultipleLines', 'No'),
                InternetService=request.form.get('InternetService', 'Fiber optic'),
                OnlineSecurity=request.form.get('OnlineSecurity', 'No'),
                OnlineBackup=request.form.get('OnlineBackup', 'No'),
                DeviceProtection=request.form.get('DeviceProtection', 'No'),
                TechSupport=request.form.get('TechSupport', 'No'),
                StreamingTV=request.form.get('StreamingTV', 'No'),
                StreamingMovies=request.form.get('StreamingMovies', 'No'),
                Contract=request.form.get('Contract', 'Month-to-month'),
                PaperlessBilling=request.form.get('PaperlessBilling', 'Yes'),
                PaymentMethod=request.form.get('PaymentMethod', 'Electronic check'),
                tenure=tenure,
                MonthlyCharges=MonthlyCharges,
                TotalCharges=TotalCharges
            )

            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            proba = predict_pipeline.predict(pred_df)
            proba_pct = round(proba * 100, 1)

            # Determine 3-tier risk classification
            if proba >= 0.60:
                risk_level = "High Risk"
                risk_class = "high"
            elif proba >= 0.35:
                risk_level = "Medium Risk"
                risk_class = "medium"
            else:
                risk_level = "Low Risk"
                risk_class = "low"

            results = {
                "level": risk_level,
                "class": risk_class,
                "probability": proba_pct
            }
            return render_template('home.html', results=results, form_data=form_data)

        except Exception as e:
            return render_template('home.html', error=str(e), form_data=form_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
