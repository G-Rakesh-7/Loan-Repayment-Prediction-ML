import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, log_loss, precision_score, recall_score
)
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("data/Micro-credit-Data-file.csv")

# Drop irrelevant columns
df.drop(columns=['Unnamed: 0', 'msisdn', 'pdate'], inplace=True)

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Feature Engineering
df['repayment_ratio'] = df['payback30'] / (df['amnt_loans30'] + 1e-6)
df['loan_frequency'] = df['cnt_loans30'] + df['cnt_loans90']
df['avg_loan_amt'] = (df['amnt_loans30'] + df['amnt_loans90']) / 2

# Encode categorical features
df = pd.get_dummies(df, columns=['pcircle'], drop_first=True)

# Define features and target
X = df.drop(columns='label')
y = df['label']

# Split the data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42)

# Train model
model = XGBClassifier(
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=4,
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("Evaluation:")
print(f"Log Loss: {log_loss(y_test, y_pred_proba):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "outputs/xgboost_loan_model.pkl")
