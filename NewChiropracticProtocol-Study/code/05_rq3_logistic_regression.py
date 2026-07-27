"""
05_rq3_logistic_regression.py
RQ3: Multiple logistic regression predicting composite success.
Condition variable excluded due to insufficient variability in the sample.
Uses 4 predictors: age, vas_pre, maxlimit_num, neuro_def_pre.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from logFileHandler import Logger

log = Logger('../processLog.txt', '05_rq3')

df = pd.read_csv('../data/clinical_data.csv')

# Prepare data — condition excluded due to insufficient variability
df_model = df[['success', 'age', 'vas_pre', 'maxlimit_pre', 'neuro_def_pre']].dropna()

# Map ordinal maxlimit_pre to numeric
limit_map = {'WNL': 0, 'MILD': 1, 'MOD': 2, 'SEVERE': 3, 'FUL': 3}
df_model['maxlimit_num'] = df_model['maxlimit_pre'].map(limit_map)

# Feature matrix and target — 4 predictors only
feature_cols = ['age', 'vas_pre', 'maxlimit_num', 'neuro_def_pre']
X = df_model[feature_cols].values
y = df_model['success'].values

# Standardize for better convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit logistic regression
model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
model.fit(X_scaled, y)

coefs = model.coef_[0]
intercept = model.intercept_[0]
y_pred = model.predict(X_scaled)

log.log("=== RQ3: LOGISTIC REGRESSION ===")
log.log(f"Features (condition excluded due to low variability): {feature_cols}")
log.log(f"Intercept: {intercept:.4f}")
for name, coef in zip(feature_cols, coefs):
    or_val = np.exp(coef)
    log.log(f"  {name}: coef={coef:.4f}, OR={or_val:.4f}")

log.log(f"Training accuracy: {(y_pred == y).mean():.3f}")
log.log(f"Note: Full inferential statistics (p-values, CI) require larger sample size.")
log.log(f"The condition variable was excluded due to insufficient variability")
log.log(f"(96.7% of patients presented with disc herniation-related complaints).")
log.log(f"This model structure will scale to N=160.")

log.close()