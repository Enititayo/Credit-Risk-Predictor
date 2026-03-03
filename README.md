**Credit Risk Prediction System**
*📌 Project Overview*

This project is a machine learning–based Credit Risk Prediction System that classifies whether a loan applicant is likely to default or not.

The model is designed to support financial risk assessment by predicting:

0 → No Default

1 → Default

The primary objective is to build a precision-focused classification model that reduces financial losses while maintaining reliable predictions.

*🎯 Problem Statement*

Financial institutions must evaluate loan applicants before approving credit. Incorrect predictions can lead to:

Financial losses (approving risky borrowers)

Lost customers (rejecting safe borrowers)

This project applies supervised machine learning to predict loan default risk using structured financial data.

*📊 Model Performance*

Precision Score: 0.88
Accuracy: 0.92

Confusion Matrix
[[4932  140]
 [ 390 1055]]
Interpretation

True Negatives (4932) → Correctly predicted safe borrowers

False Positives (140) → Safe borrowers incorrectly flagged as default

False Negatives (390) → Risky borrowers missed by the model

True Positives (1055) → Correctly identified defaulters

The model prioritizes precision, meaning when it predicts a borrower will default, it is correct 88% of the time.

*🏗 Project Structure*
├── data/
│   └── credit_data.csv
├── models/
│   ├── c_model.pkl
│   └── scaler.pkl
├── preprocess.py
├── train.py
├── evaluate.py
└── README.md
**🔄 Workflow**
*1️⃣ Data Preprocessing*

Missing values handled using .fillna(0)

Categorical variables encoded using pd.get_dummies()

Binary columns mapped to numeric values

Feature-target separation performed

*2️⃣ Model Training*

80/20 train-test split

Classification model trained

Model saved using joblib

Precision used as primary optimization metric

*3️⃣ Model Evaluation***

Precision Score

Confusion Matrix

Classification Report

**🛠 Technologies Used**

Python

Pandas

Scikit-learn

Joblib

**🧠 Key Machine Learning Concepts Demonstrated**

Feature engineering

One-hot encoding

Binary encoding

Model training and persistence

Confusion matrix interpretation

Precision-focused classification strategy

Modular ML project structure

*🚀 Future Improvements*

Hyperparameter tuning using RandomizedSearchCV

Implementation of XGBoost

Threshold optimization to improve recall

ROC-AUC evaluation

API deployment (Flask / FastAPI)

*▶️ How to Run the Project*

Install required libraries

Run:

python train.py
python evaluate.py
**👤 Author**
Oyewole Enititayo

Machine Learning Project – Credit Risk Prediction

This project demonstrates applied machine learning engineering, structured model development, and real-world financial risk modeling principles.