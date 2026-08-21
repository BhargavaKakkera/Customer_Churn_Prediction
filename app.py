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
        data = CustomData(
            gender=request.form.get('gender'),
            SeniorCitizen=request.form.get('SeniorCitizen'),
            Partner=request.form.get('Partner'),
            Dependents=request.form.get('Dependents'),
            PhoneService=request.form.get('PhoneService'),
            MultipleLines=request.form.get('MultipleLines'),
            InternetService=request.form.get('InternetService'),
            OnlineSecurity=request.form.get('OnlineSecurity'),
            OnlineBackup=request.form.get('OnlineBackup'),
            DeviceProtection=request.form.get('DeviceProtection'),
            TechSupport=request.form.get('TechSupport'),
            StreamingTV=request.form.get('StreamingTV'),
            StreamingMovies=request.form.get('StreamingMovies'),
            Contract=request.form.get('Contract'),
            PaperlessBilling=request.form.get('PaperlessBilling'),
            PaymentMethod=request.form.get('PaymentMethod'),
            tenure=float(request.form.get('tenure')),
            MonthlyCharges=float(request.form.get('MonthlyCharges')),
            TotalCharges=float(request.form.get('TotalCharges'))
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
