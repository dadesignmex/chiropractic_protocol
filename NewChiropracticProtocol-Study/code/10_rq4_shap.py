"""
10_rq4_shap.py
Generate SHAP summary plot for the best model (LightGBM).
Uses saved model, scaler, and PCA from 08_rq4_advanced.py.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from logFileHandler import Logger

log = Logger('../processLog.txt', '10_shap')

log.log("=== SHAP ANALYSIS ===")

# Load saved artifacts
model = joblib.load('../output/models/lightgbm_best.pkl')
scaler = joblib.load('../output/models/scaler.pkl')
pca = joblib.load('../output/models/pca.pkl')

# Load data
df = pd.read_csv('../data/clinical_data.csv')

# Recreate the same feature engineering as in 08_rq4_advanced.py
limit_map = {'WNL': 0, 'MILD': 1, 'MOD': 2, 'SEVERE': 3, 'FUL': 3}
pain_map = {'NONE': 0, 'MILD': 1, 'MOD': 2, 'SEVERE': 3}

feature_dict = {}
feature_dict['age'] = df['age']
feature_dict['vas_pre'] = df['vas_pre']
feature_dict['maxlimit_num'] = df['maxlimit_pre'].map(limit_map)
feature_dict['neuro_def_pre'] = df['neuro_def_pre']
feature_dict['maxpain_num'] = df['maxpain_pre'].map(pain_map)

# Count of abnormal neurological tests
def count_abnormal_neuro(row):
    count = 0
    for side in ['LEFT', 'RIGHT']:
        for level in ['L4', 'L5', 'S1']:
            col = f'SENSORY {side} {level} PRE'
            if col in row.index:
                val = str(row[col]).strip().upper()
                if pd.notna(val) and val not in ['WNL', 'NORMAL', '', 'NAN']:
                    count += 1
    for side in ['LEFT', 'RIGHT']:
        for level in ['L1-3', 'L4', 'L5', 'S1']:
            col = f'MOTOR {side} {level} PRE'
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    try:
                        if int(float(val)) < 5: count += 1
                    except:
                        if str(val).strip() == '-': count += 1
    for side in ['LEFT', 'RIGHT']:
        for level in ['L4', 'S1']:
            col = f'DTR {side} {level} PRE'
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    try:
                        r = int(float(val))
                        if r < 2 or r > 2: count += 1
                    except: pass
    return count

feature_dict['neuro_abnormal_count'] = df.apply(count_abnormal_neuro, axis=1)

# SLR features
slr_left = pd.to_numeric(df['SLR DEGREES LEFT_PRE'], errors='coerce')
slr_right = pd.to_numeric(df['SLR DEGREES RIGHT_PRE'], errors='coerce')
feature_dict['slr_mean'] = (slr_left + slr_right) / 2
feature_dict['slr_min'] = np.minimum(slr_left, slr_right)
feature_dict['slr_asymmetry'] = np.abs(slr_left - slr_right)

# Special tests count
special_cols = [
    'LEFT KEMPS_PRE', 'RIGHT KEMPS_PRE',
    "FABER Patrick's TEST LEFT_PRE", "FABER Patrick's TEST RIGHT_PRE",
    'Passive SLR LEFT_PRE', 'Passive SLR RIGHT_PRE'
]
def count_positive_tests(row):
    count = 0
    for col in special_cols:
        if col in row.index:
            val = str(row[col]).strip().upper()
            if val in ['POS', 'POSITIVE', 'YES', '1']: count += 1
    return count
feature_dict['positive_tests_count'] = df.apply(count_positive_tests, axis=1)

feature_df = pd.DataFrame(feature_dict)
feature_df = feature_df.fillna(feature_df.median())

X_raw = feature_df.values

# Apply saved scaler and PCA
X_scaled = scaler.transform(X_raw)
X_pca = pca.transform(X_scaled)

# SHAP explainer for tree model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_pca)

# Summary plot
plt.figure(figsize=(12, 6))
shap.summary_plot(
    shap_values,
    X_pca,
    feature_names=[f'PC{i+1}' for i in range(X_pca.shape[1])],
    show=False
)
plt.tight_layout()
plt.savefig('../output/figures/shap_summary.png', dpi=300, bbox_inches='tight')
log.log("SHAP summary plot saved to output/figures/shap_summary.png")

log.close()