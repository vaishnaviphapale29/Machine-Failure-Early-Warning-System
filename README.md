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

## Model Performance

Two Machine Learning models were trained and compared using 5-fold cross-validation with F1-score.

| Model | Cross-Validation F1-Score |
|---|---:|
| Random Forest | **0.6602** |
| Logistic Regression | 0.2346 |

**Best Model:** Random Forest Classifier

### Evaluation Results

The selected Random Forest model was evaluated on the test dataset.

| Metric | Score |
|---|---:|
| Accuracy | 97.95% |
| Precision | 72.13% |
| Recall | 64.71% |
| F1-Score | 68.22% |
| ROC-AUC | 96.12% |

### Dataset Summary

- **Total Records:** 10,000
- **Total Features:** 14
- **Training Records:** 8,000
- **Testing Records:** 2,000
- **Machine Failure Cases:** 339 (3.39%)
- **Non-Failure Cases:** 9,661 (96.61%)

### Key Findings

- Random Forest outperformed Logistic Regression based on cross-validation F1-score.
- The model achieved 97.95% test accuracy.
- The model identified machine failures with 64.71% recall.
- The ROC-AUC score of 0.9612 indicates strong ability to distinguish between failure and non-failure cases.

### Generated Outputs

- `outputs/Machine_Failure_Model.pkl` — Trained Random Forest model
- `images/` — Generated charts and visualizations
- `outputs/` — Test predictions and other saved results

### Future Improvements

- Improve failure detection recall to reduce missed failures.
- Experiment with class-balancing techniques and hyperparameter tuning.
- Add real-time machine monitoring and early-warning alerts.
- Develop an interactive dashboard for machine health monitoring.

## Disclaimer
This project provides machine failure predictions based on a trained model. Predictions are estimates and should not be treated as guaranteed failure warnings.