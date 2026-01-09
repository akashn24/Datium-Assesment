# Vehicle Sold Amount Prediction – Take-Home Assignment

This repository contains an end-to-end machine learning workflow for predicting vehicle sold prices using historical auction data. The project covers exploratory data analysis, feature engineering, model training, evaluation, and experiment tracking using MLflow.

---

## 📁 Repository Structure

```text
.
├── data
│   ├── DatiumTest.rpt
│   └── DatiumTrain.rpt
├── notebooks
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   ├── final_model.py
│   └── modelling.ipynb
├── README.md
└── requirements.txt
```

---

## 📊 Notebooks Overview

### 1️⃣ Exploratory Data Analysis (`eda.ipynb`)

This notebook focuses on understanding the dataset and identifying potential data quality issues early. Key steps include:

- Schema consistency checks and data validation
- Distribution analysis of `Sold_Amount` (including log transformation)
- Missing value analysis and visualisation
- Outlier detection and sanity checks
- Initial relationships between key numerical and categorical features

Findings from this notebook directly inform feature engineering decisions.

---

### 2️⃣ Feature Engineering (`feature_engineering.ipynb`)

This notebook prepares the dataset for modelling. Key steps include:

- Cleaning and filtering invalid records
- Handling missing values for numerical and categorical features
- Encoding categorical variables (One-Hot / Ordinal where appropriate)
- Creating derived features (e.g. vehicle age)
- Final feature selection based on EDA insights

---

### 3️⃣ Modelling & Evaluation (`modelling.ipynb`)

This notebook contains the core modelling workflow:

- Training gradient boosting models (LightGBM and CatBoost)
- Logging experiments using MLflow (parameters, metrics, artifacts)
- Model comparison and selection
- Residual analysis and segment-level performance evaluation
- SHAP-based feature importance and interpretability
- Final evaluation on the provided test dataset

The best-performing model is logged as the final production candidate.

---

## 🧠 Final Model (`final_model.py`)

The `final_model.py` module contains a reusable class that encapsulates:

- Model training
- Model saving and loading
- Inference on new data

This structure reflects how the model could be operationalised outside a notebook environment.

---

## ⚙️ Environment Setup

### Create and activate a virtual environment (optional)

This project was run and created with Python 3.12.12

```bash
python -m venv .venv
```
To activate for macOS/Linux:
```bash
source .venv/bin/activate
```
To activate for Windows:
```bash
.venv\Scripts\activate 
```
Install Packages
```bash
pip install -r requirements.txt
```

## 🏃🏽 Run ML Flow

```bash
mlflow server --host 127.0.0.1 --port 5000
```

