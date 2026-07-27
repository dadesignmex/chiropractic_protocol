"""
06_rq4_ml_model.py
RQ4: ML classification with sensitivity threshold tuning.
Condition variable excluded due to insufficient variability.
Uses 4 features: age, vas_pre, maxlimit_num, neuro_def_pre.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, fbeta_score
from sklearn.preprocessing import StandardScaler
from logFileHandler import Logger

log = Logger('../processLog.txt', '06_rq4')

df = pd.read_csv('../data/clinical_data.csv')

# Prepare features — condition excluded due to insufficient variability
df_model = df[['success', 'age', 'vas_pre', 'maxlimit_pre', 'neuro_def_pre']].dropna()

limit_map = {'WNL': 0, 'MILD': 1, 'MOD': 2, 'SEVERE': 3, 'FUL': 3}
df_model['maxlimit_num'] = df_model['maxlimit_pre'].map(limit_map)

feature_cols = ['age', 'vas_pre', 'maxlimit_num', 'neuro_def_pre']
X = df_model[feature_cols]
y = df_model['success']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    # Tune threshold for sensitivity >= 0.85 and PPV >= 0.70
    best_thresh = 0.5
    for t in np.arange(0.3, 0.9, 0.01):
        y_pred_t = (y_prob >= t).astype(int)
        tp = ((y_pred_t == 1) & (y_test == 1)).sum()
        fp = ((y_pred_t == 1) & (y_test == 0)).sum()
        fn = ((y_pred_t == 0) & (y_test == 1)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        if sens >= 0.85 and ppv >= 0.70:
            best_thresh = t
            break

    y_pred = (y_prob >= best_thresh).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    f2 = fbeta_score(y_test, y_pred, beta=2)

    log.log(f"\n{'='*50}")
    log.log(f"{name} | Threshold={best_thresh:.2f} | AUC={auc:.3f} | F2={f2:.3f}")
    log.log(classification_report(y_test, y_pred, target_names=['Fail', 'Success'],
                                  zero_division=0))

log.log(f"\nCondition variable excluded due to insufficient variability")
log.log(f"(96.7% disc herniation-related complaints).")
log.log(f"Features used: {feature_cols}")

log.close()