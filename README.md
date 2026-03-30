# ML Model Readiness and Leakage Detection

**Evaluating Machine Learning Models for Deployment Safety and Data Leakage Risks**

## Project Overview

The **ML Model Readiness and Leakage Detection** system is designed to evaluate whether a machine learning model is safe for real-world deployment.
It detects potential **data leakage** and assesses model reliability using both **random split** and **time-based split** validation strategies.

This project demonstrates a production-style machine learning workflow including:

* Data validation and model evaluation
* Leakage risk detection
* API-based prediction system
* Docker containerization
* Cloud deployment readiness

The goal is to help data scientists and organizations make informed deployment decisions before releasing models into production.

---

## Problem Statement

Machine learning models sometimes show very high performance during testing but fail in real-world scenarios.
One major reason for this is **data leakage**, where information from the future or target variable unintentionally enters the training data.

This project automatically detects such risks and determines whether a model is ready for deployment.

---

## Key Features

* Upload dataset (CSV file)
* Evaluate model readiness
* Compare Random Split vs Time-Based Split performance
* Calculate Leakage Score
* Provide Deployment Decision
* REST API using FastAPI
* Docker-ready application

---

## Tech Stack

* Python
* FastAPI
* Scikit-learn
* Pandas
* NumPy
* Joblib
* Docker
* Uvicorn

---

## Project Structure

```
ML model readiness and leakage detection/
│
├── app/
│   ├── main.py          # FastAPI entry point
│   └── predict.py       # Prediction and evaluation logic
│
├── pipeline.py          # Model evaluation pipeline
├── pipeline.pkl         # Saved trained model
├── requirements.txt     # Dependencies
├── Dockerfile           # Docker configuration
├── .dockerignore
├── .gitignore
└── README.md
```

---

## How the System Works

1. User uploads a dataset
2. API reads the CSV file
3. Model is evaluated using:

   * Random split validation
   * Time-based split validation
4. Performance difference is calculated
5. Leakage score is generated
6. Deployment readiness decision is returned

---

## API Endpoint

### POST /predict

This endpoint evaluates model readiness and detects potential data leakage.

### Parameters

```
target_col : Target column name
time_col   : Time column name
file       : CSV dataset
```

---

## Example Request

```
POST /predict
```

Query Parameters:

```
target_col = Class
time_col = Time
```

File:

```
creditcard.csv
```

---

## Example Response

```
{
  "Random Split AUC": 0.97,
  "Time-Based AUC": 0.89,
  "Leakage Score": 0.08,
  "Deployment Decision": "SAFE",
  "Leakage Verdict": "Model ready for deployment"
}
```

---

## Installation — Local Setup

### Step 1 — Clone the repository

```
git clone https://github.com/your-username/ml-model-readiness-leakage-detection.git
cd ml-model-readiness-leakage-detection
```

### Step 2 — Install dependencies

```
pip install -r requirements.txt
```

### Step 3 — Run the API

```
uvicorn app.main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

---

## Running with Docker

### Build Docker image

```
docker build -t model-readiness-api .
```

### Run container

```
docker run -p 8000:8000 model-readiness-api
```

Open:

```
http://localhost:8000/docs
```

---

## Use Cases

* Detect data leakage before model deployment
* Evaluate machine learning model readiness
* Validate time-series and production models
* Improve model reliability
* Prevent deployment risks

---

## Future Improvements

* Streamlit web interface
* Automated feature leakage detection
* Model monitoring dashboard
* Cloud deployment automation
* CI/CD pipeline integration

---

## Author

**Ishwari Punyarthi**
Field: Data Science / Machine Learning

---

## Project Type

Machine Learning | Model Validation | Data Leakage Detection | API Development | MLOps
