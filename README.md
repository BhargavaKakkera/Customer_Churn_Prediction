## 📉 Telecom Churn Prediction – ML Web App

This project predicts whether a customer is likely to **churn (leave)** the telecom service based on their demographics and usage behavior. It is built using a full **ML pipeline**, deployed via **Flask** and **Docker**, and includes a web-based prediction interface.

---

### 🚀 Live Demo & Deployment

#### 🌐 Render Deployment (Recommended for Hosting)

🔗 **Live App on Render**: https://customer-churn-prediction-zohx.onrender.com

👉 Try it live with real input features!

**Deployment Steps:**

1. Create an account on [https://render.com](https://render.com)
2. Connect your GitHub repo
3. Set:

   * **Build Command**: pip install -r requirements.txt
   * **Start Command**: python app.py
4. Optional `.render.yaml` config:

```yaml
services:
  - type: web
    name: telecom-churn
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python app.py"
    plan: free
```

---

#### 🐻 Docker Deployment

🔗 **DockerHub Image**: (https://hub.docker.com/r/bhargava1420/churn-prediction)

```bash
docker pull bhargava1420/churn-prediction
docker run -p 5000:5000 bhargava1420/churn-prediction
```

Access it at: [http://localhost:5000](http://localhost:5000)

---

### 🧠 ML Model Info

* **Model Used**: Random Forest Classifier
* **Target**: `Churn` (binary classification)

**Key Features**:

* SeniorCitizen
* Dependents
* Internet & Streaming services
* Contract type
* Paperless billing
* Payment method
* Tenure, Monthly Charges, Total Charges

---

### 📁 Folder Structure

```
customer churn prediction/
├── app.py
├── Dockerfile
├── .render.yaml
|__.dockerignore
├── requirements.txt
├── templates/
│   ├── index.html
│   └── home.html
├── src/
│   ├── notebook/
│   │   ├── eda.ipynb
│   │__ data.csv
│   ├── pipeline/
│   │   |__train_pipeline.py
|   |   |__predict_pipeline.py
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
├── saved_model/   ← Contains saved model.pkl, transformer.pkl

### 🚡 Run Locally (Without Docker)

```bash
# Step 1: Create a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux

# Step 2: Install packages
pip install -r requirements.txt

# Step 3: Run the app
python app.py
```

---

### ✅ Features

* Clean architecture with modular code
* OOP-based predict pipeline
* Custom exception and logging system
* Dockerized for portability
* Live hosted on Render
* Only `predict_pipeline` included in Docker for efficiency

---

