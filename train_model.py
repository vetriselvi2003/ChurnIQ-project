"""
ChurnIQ — Model Training Pipeline
Run this script to retrain the model: python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ChurnIQ — Customer Churn Prediction Model Training")
print("=" * 60)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("\n[1/5] Loading dataset...")
df = pd.read_csv('telco_churn.csv')
print(f"      Shape: {df.shape}")
print(f"      Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
print("\n[2/5] Preprocessing...")
df = df.drop('customerID', axis=1)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = list(le.classes_)
    print(f"      Encoded: {col} → {le_dict[col]}")

X = df.drop('Churn', axis=1)
y = df['Churn']
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n      Train: {X_train.shape}, Test: {X_test.shape}")

# ── 3. Model Comparison ───────────────────────────────────────────────────────
print("\n[3/5] Comparing models...")
models = {
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                        learning_rate=0.1, random_state=42),
}

results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    acc = clf.score(X_test, y_test)
    results[name] = {'auc': auc, 'accuracy': acc}
    print(f"      {name:25s} — AUC: {auc:.4f} | Accuracy: {acc:.4f}")

best_name = max(results, key=lambda k: results[k]['auc'])
print(f"\n      ✅ Best model: {best_name} (AUC: {results[best_name]['auc']:.4f})")

# ── 4. Final Model Evaluation ─────────────────────────────────────────────────
print("\n[4/5] Final evaluation...")
best_model = models[best_name]
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, output_dict=True)
auc    = roc_auc_score(y_test, y_prob)

print("\n      Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['No Churn', 'Churn']))

metrics = {
    'auc'      : round(auc, 4),
    'accuracy' : round(report['accuracy'] * 100, 2),
    'recall'   : round(report['1']['recall'] * 100, 2),
    'precision': round(report['1']['precision'] * 100, 2),
    'f1'       : round(report['1']['f1-score'] * 100, 2),
}
print(f"      ROC-AUC: {metrics['auc']}")

# Feature importance
fi = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\n      Top 5 Churn Drivers:")
for feat, score in fi.head(5).items():
    print(f"        {feat:25s}: {score:.4f}")

# ── 5. Save Artifacts ─────────────────────────────────────────────────────────
print("\n[5/5] Saving model artifacts...")
pickle.dump(best_model,    open('churn_model.pkl',    'wb'))
pickle.dump(le_dict,       open('label_encoders.pkl', 'wb'))
pickle.dump(feature_names, open('feature_names.pkl',  'wb'))
json.dump(metrics,         open('metrics.json', 'w'))

print("      ✅ churn_model.pkl saved")
print("      ✅ label_encoders.pkl saved")
print("      ✅ feature_names.pkl saved")
print("      ✅ metrics.json saved")
print("\n" + "=" * 60)
print("Training complete! Run: streamlit run app.py")
print("=" * 60)