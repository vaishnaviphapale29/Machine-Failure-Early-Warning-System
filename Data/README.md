# Machine Failure Early-Warning System

## Overview
A Machine Learning project that predicts potential machine failures using machine operating conditions and sensor-related data. The system helps identify possible failures early and supports predictive maintenance.

## Dataset
- **Dataset:** AI4I 2020 Predictive Maintenance Dataset
- **Target:** `Machine failure`
- **Input features:** Machine type, air temperature, process temperature, rotational speed, torque, and tool wear.

## Machine Learning Algorithms
- Logistic Regression
- Random Forest Classifier (Bagging-based ensemble)

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib

## Project Structure
```text
Machine_Failure_Early_Warning_System/
├── Data/
│   └── ai4i2020.csv
├── images/
├── outputs/
├── Machine_Failure_Early_Warning_System.py
├── Test_New_Data.py
├── requirements.txt
└── README.md
```

## Features
- Dataset loading and validation
- Data preprocessing and exploratory data analysis
- Machine failure prediction
- Model comparison using classification metrics
- Confusion matrix and data visualization
- Saving the trained model
- Testing new machine data through user input

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

First, train and evaluate the models:

```bash
python Machine_Failure_Early_Warning_System.py
```

Then test new machine data:

```bash
python Test_New_Data.py
```

Run the main script first to generate the trained model file.

## Disclaimer
This project provides machine failure predictions based on a trained model. Predictions are estimates and should not be treated as guaranteed failure warnings.