"""
05_rq3_logistic_regression.py
RQ3: Multiple logistic regression predicting composite success.
Condition variable excluded due to insufficient variability in the sample.
Uses 4 predictors: age, vas_pre, maxlimit_num, neuro_def_pre.
Computes coefficients, odds ratios, Wald p-values, and confidence intervals.
"""
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from logFileHandler import Logger

log = Logger('../processLog.txt', '05_rq3')

# ------------------------------------------------------------
# 1. Load and prepare data
# ------------------------------------------------------------
df = pd.read_csv('../data/clinical_data.csv')

# Condition excluded due to insufficient variability
df_model = df[['success', 'age', 'vas_pre', 'maxlimit_pre', 'neuro_def_pre']].dropna()

# Map ordinal maxlimit_pre to numeric
limit_map = {'WNL': 0, 'MILD': 1, 'MOD': 2, 'SEVERE': 3, 'FUL': 3}
df_model['maxlimit_num'] = df_model['maxlimit_pre'].map(limit_map)

# Feature matrix and target — 4 predictors only
feature_cols = ['age', 'vas_pre', 'maxlimit_num', 'neuro_def_pre']
X = df_model[feature_cols].values
y = df_model['success'].values

# Standardize for stable convergence (coefficients reported on standardized scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# 2. Fit logistic regression
# ------------------------------------------------------------
model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
model.fit(X_scaled, y)

coefs = model.coef_[0]
intercept = model.intercept_[0]
y_pred = model.predict(X_scaled)
accuracy = (y_pred == y).mean()

# ------------------------------------------------------------
# 3. Compute Wald p-values and confidence intervals
# ------------------------------------------------------------
# Design matrix with intercept
X_design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
n, p = X_design.shape

# Predicted probabilities
p_hat = model.predict_proba(X_scaled)[:, 1]

# Weight matrix W = diag(p*(1-p))
W = np.diag(p_hat * (1 - p_hat))

# Covariance matrix (X' W X)^{-1}
try:
    cov = np.linalg.inv(X_design.T @ W @ X_design)
except np.linalg.LinAlgError:
    # Add small ridge if singular
    cov = np.linalg.inv(X_design.T @ W @ X_design + np.eye(p) * 1e-6)

se = np.sqrt(np.diag(cov))

# Combined coefficients (intercept + predictors)
beta_all = np.concatenate([[intercept], coefs])

# Wald z and p-values
z_stats = beta_all / se
p_values = 2 * (1 - scipy_stats.norm.cdf(np.abs(z_stats)))

# 95% confidence intervals for coefficients
ci_lower = beta_all - 1.96 * se
ci_upper = beta_all + 1.96 * se

# Odds ratios and CIs
or_vals = np.exp(beta_all)
or_lower = np.exp(ci_lower)
or_upper = np.exp(ci_upper)

# ------------------------------------------------------------
# 4. Log results
# ------------------------------------------------------------
log.log("=== RQ3: LOGISTIC REGRESSION WITH INFERENTIAL STATISTICS ===")
log.log(f"Features (condition excluded due to low variability): {feature_cols}")
log.log(f"Training accuracy: {accuracy:.4f}")
log.log(f"")
log.log(f"{'Predictor':<22} {'β':>10} {'SE':>10} {'z':>10} {'p':>10} {'OR':>10} {'95% CI':>20}")
log.log(f"{'-'*90}")

names = ['Intercept'] + feature_cols
for i, name in enumerate(names):
    log.log(f"{name:<22} {beta_all[i]:>10.4f} {se[i]:>10.4f} {z_stats[i]:>10.3f} {p_values[i]:>10.4f} {or_vals[i]:>10.4f} [{or_lower[i]:.3f}, {or_upper[i]:.3f}]")

log.log(f"")
log.log(f"Significance at α = .05:")

for i, name in enumerate(names[1:], start=1):
    if p_values[i] < 0.05:
        log.log(f"  {name}: significant (p = {p_values[i]:.4f})")
    else:
        log.log(f"  {name}: not significant (p = {p_values[i]:.4f})")

log.log(f"")
log.log(f"Note: Coefficients are on the standardized scale for continuous variables.")
log.log(f"Odds ratios represent the change in odds per 1 SD increase for standardized variables.")
log.log(f"The condition variable was excluded due to insufficient variability")
log.log(f"(96.7% of patients presented with disc herniation-related complaints).")

log.close()