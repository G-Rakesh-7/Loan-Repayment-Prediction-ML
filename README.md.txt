🚀 Project Overview
This project aims to predict whether a customer will repay a mobile microloan within 5 days based on their behavior and historical data. The model is designed to help a telecom company working with a microfinance institution (MFI) minimize loan defaults and optimize the distribution of loans.

🧠 Problem Statement
A telecom company, in collaboration with an MFI, wants to predict if customers will repay small microloans (₹5 or ₹10) on time. By using customer behavior data, the model will help the company reduce defaults and make more informed loan decisions.

📊 Dataset Details
Size: 429 records and 36 features
Features Include: Recharge patterns, loan history, and repayment behavior

📈 Approach
Data Preprocessing & Feature Engineering:
Clean and prepare the data to ensure it’s suitable for machine learning.
Extract and create meaningful features that can improve the model’s performance.
Model Training with XGBoost:
Train a predictive model using XGBoost, a powerful algorithm for handling imbalanced data.
Tune the model’s hyperparameters to maximize its performance.
Model Evaluation:
Evaluate the model using metrics like Log Loss, AUC, Precision, and Recall to ensure accurate predictions.


💻 Libraries Used
pandas: For data manipulation and analysis.
numpy: For numerical computations.
scikit-learn: For machine learning algorithms and evaluation.
xgboost: For model training with gradient boosting.
joblib: For saving and loading models.

🧪 How to Run the Project
To set up and run the project, follow these steps:

1)Install the required dependencies:
   pip install -r requirements.txt

2)Train the model:
   python src/train_model.py