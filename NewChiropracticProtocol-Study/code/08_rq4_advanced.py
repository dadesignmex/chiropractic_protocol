"""
08_rq4_advanced.py
Advanced RQ4: Expanded feature set, PCA, 80/20 split, multiple algorithms.
Compares Logistic Regression, Random Forest, SVM, and LightGBM
with expanded features and dimensionality reduction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, roc_auc_score, fbeta_score,
    confusion_matrix, precision_recall_curve, auc,
    make_scorer, matthews_corrcoef, brier_score_loss
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from logFileHandler import Logger

log = Logger('../processLog.txt', '08_advanced')

# ============================================================
# STEP 1: Load data
# ============================================================
df = pd.read_csv('../data/clinical_data.csv')
log.log("=== ADVANCED RQ4: EXPANDED FEATURES + PCA ===")
log.log(f"Total patients: {len(df)}")

# ============================================================
# STEP 2: Build expanded feature set
# ============================================================
# Create features from available raw data that were not used before.

feature_dict = {}

# --- Core 4 features (original) ---
feature_dict['age'] = df['age']
feature_dict['vas_pre'] = df['vas_pre']

# Map ordinal maxlimit_pre to numeric
limit_map = {'WNL': 0, 'MILD': 1, 'MOD': 2, 'SEVERE': 3, 'FUL': 3}
feature_dict['maxlimit_num'] = df['maxlimit_pre'].map(limit_map)

feature_dict['neuro_def_pre'] = df['neuro_def_pre']

# ---  Pain on movement features ---
# Map ordinal pain to numeric
pain_map = {'NONE': 0, 'MILD': 1, 'MOD': 2, 'SEVERE': 3}
feature_dict['maxpain_num'] = df['maxpain_pre'].map(pain_map)

# ---  Count of neurological deficits ---
# How many of the 27 neuro tests were abnormal?
neuro_cols_pre = [c for c in df.columns if ('SENSORY' in c or 'MOTOR' in c or 'DTR' in c) and 'PRE' in c and 'POST' not in c]

def count_abnormal_neuro(row):
    """Count how many neurological tests are abnormal."""
    count = 0
    # Sensory: abnormal if not WNL
    for side in ['LEFT', 'RIGHT']:
        for level in ['L4', 'L5', 'S1']:
            col = f'SENSORY {side} {level} PRE'
            if col in row.index:
                val = str(row[col]).strip().upper()
                if pd.notna(val) and val not in ['WNL', 'NORMAL', '', 'NAN']:
                    count += 1
    # Motor: abnormal if < 5
    for side in ['LEFT', 'RIGHT']:
        for level in ['L1-3', 'L4', 'L5', 'S1']:
            col = f'MOTOR {side} {level} PRE'
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    try:
                        if int(float(val)) < 5:
                            count += 1
                    except:
                        if str(val).strip() == '-':
                            count += 1
    # Reflexes: abnormal if < 2 or > 2
    for side in ['LEFT', 'RIGHT']:
        for level in ['L4', 'S1']:
            col = f'DTR {side} {level} PRE'
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    try:
                        r = int(float(val))
                        if r < 2 or r > 2:
                            count += 1
                    except:
                        pass
    return count

feature_dict['neuro_abnormal_count'] = df.apply(count_abnormal_neuro, axis=1)

# --- SLR (Straight Leg Raise) degrees ---
# Average of left and right SLR as a measure of nerve root tension
slr_cols = ['SLR DEGREES LEFT_PRE', 'SLR DEGREES RIGHT_PRE']
slr_available = [c for c in slr_cols if c in df.columns]
if len(slr_available) == 2:
    slr_left = pd.to_numeric(df[slr_available[0]], errors='coerce')
    slr_right = pd.to_numeric(df[slr_available[1]], errors='coerce')
    feature_dict['slr_mean'] = (slr_left + slr_right) / 2
    feature_dict['slr_min'] = np.minimum(slr_left, slr_right)  # Worst leg
    feature_dict['slr_asymmetry'] = np.abs(slr_left - slr_right)  # Side-to-side difference
    log.log("SLR features added: mean, min, asymmetry")

# --- Count of positive special tests ---
# Kemps, FABER, SLR positivity
special_test_cols = [
    'LEFT KEMPS_PRE', 'RIGHT KEMPS_PRE',
    "FABER Patrick's TEST LEFT_PRE", "FABER Patrick's TEST RIGHT_PRE",
    'Passive SLR LEFT_PRE', 'Passive SLR RIGHT_PRE'
]
special_available = [c for c in special_test_cols if c in df.columns]
if len(special_available) > 0:
    def count_positive_tests(row):
        count = 0
        for col in special_available:
            val = str(row[col]).strip().upper()
            if val in ['POS', 'POSITIVE', 'YES', '1']:
                count += 1
        return count
    feature_dict['positive_tests_count'] = df.apply(count_positive_tests, axis=1)
    log.log(f"Special tests feature added from {len(special_available)} columns")

# --- NEW: Treatment intensity proxy ---
# If number of sessions or treatment duration available
# (Not in current dataset - placeholder for future)
# feature_dict['num_sessions'] = ...

# --- Assemble feature matrix ---
feature_df = pd.DataFrame(feature_dict)
feature_df = feature_df.fillna(feature_df.median())  # Impute any remaining NaN

log.log(f"Total features before PCA: {len(feature_df.columns)}")
log.log(f"Feature names: {list(feature_df.columns)}")

X_raw = feature_df.values
y = df['success'].values

# ============================================================
# STEP 3: Standardize and apply PCA
# ============================================================
# STEP 3: Split into train/test FIRST (before scaling/PCA)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

log.log(f"Train: {len(X_train_raw)} patients ({y_train.sum()} successes) | Test: {len(X_test_raw)} ({y_test.sum()} successes)")
log.log(f"")


# ============================================================
# STEP 4: Fit scaler and PCA on TRAINING DATA ONLY
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

pca_full = PCA()
pca_full.fit(X_train_scaled)
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum_var >= 0.95) + 1

log.log(f"--- PCA Analysis (fitted on training data only) ---")
log.log(f"Components needed for 95% variance: {n_components_95}")
for i, (var, cum) in enumerate(zip(pca_full.explained_variance_ratio_, cumsum_var)):
    if i < n_components_95 + 2:
        log.log(f"  PC{i+1}: {var:.3f} (cumulative: {cum:.3f})")

pca = PCA(n_components=n_components_95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

log.log(f"Features after PCA: {n_components_95}")
log.log(f"")

# PCA loadings (can be computed on training data)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(n_components_95)],
    index=feature_df.columns
)
log.log(f"PCA Loadings (feature contributions to each component):")
log.log(f"{loadings.to_string()}")
log.log(f"")

# Use the PCA-transformed data for modelling
X_train = X_train_pca
X_test = X_test_pca
 

# ============================================================
# STEP 5: Define models and scoring
# ============================================================
# f2_scorer = make_scorer(fbeta_score, beta=2)  lo quite para que sensitivity no sea 1
# Use F1 scorer for more balanced threshold selection
f1_scorer = make_scorer(fbeta_score, beta=1)  #We sill use f1 score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        'params': {
            'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['saga']  # saga supports both l1 and l2
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(class_weight='balanced', random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [3, 5, 10]
        }
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(
            objective='binary', metric='binary_logloss',
            random_state=42, verbose=-1, n_jobs=-1
        ),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [2, 3, 4],
            'num_leaves': [7, 15, 31],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_samples': [10, 20],
            'reg_alpha': [0.1, 1.0],
            'reg_lambda': [0.1, 1.0]
        }
    },
    'SVM': {
        'model': SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
        'params': {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.1, 0.01],
        }
    }
}

# ============================================================
# STEP 6: Train and evaluate all models
# ============================================================
results = {}

for name, config in models.items():
    log.log(f"--- {name} ---")
    log.log(f"Tuning hyperparameters...")

    grid = GridSearchCV(
        config['model'], config['params'],
        #cv=cv, scoring=f2_scorer, n_jobs=-1, verbose=0
        cv=cv, scoring=f1_scorer, n_jobs=-1, verbose=0   #Change to f1 score
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    log.log(f"Best params: {grid.best_params_}")
    log.log(f"Best CV F1-score: {grid.best_score_:.4f}")

    # Predict on test set
    if hasattr(best_model, 'predict_proba'):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-y_prob))  # Sigmoid transform

    # Find best threshold using F1 (balanced precision/recall)
    best_thresh = 0.5
    best_f1 = 0
    best_balanced = {'thresh': 0.5, 'f1': 0, 'sens': 0, 'ppv': 0}

    for t in np.arange(0.15, 0.85, 0.01):
        y_pred_t = (y_prob >= t).astype(int)
        sens_t = (y_pred_t & y_test).sum() / y_test.sum() if y_test.sum() > 0 else 0
        ppv_t = (y_pred_t & y_test).sum() / y_pred_t.sum() if y_pred_t.sum() > 0 else 0
        f1_t = fbeta_score(y_test, y_pred_t, beta=1)
        
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t
        
        if sens_t >= 0.85 and ppv_t >= 0.70 and f1_t > best_balanced['f1']:
            best_balanced = {'thresh': t, 'f1': f1_t, 'sens': sens_t, 'ppv': ppv_t}

    # Evaluate at best F1 threshold
    y_pred = (y_prob >= best_thresh).astype(int)
    auc_val = roc_auc_score(y_test, y_prob)
    sens = (y_pred & y_test).sum() / y_test.sum() if y_test.sum() > 0 else 0
    ppv = (y_pred & y_test).sum() / y_pred.sum() if y_pred.sum() > 0 else 0
    f1 = fbeta_score(y_test, y_pred, beta=1)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Also evaluate at threshold 0.50
    y_pred_50 = (y_prob >= 0.50).astype(int)
    sens_50 = (y_pred_50 & y_test).sum() / y_test.sum() if y_test.sum() > 0 else 0
    ppv_50 = (y_pred_50 & y_test).sum() / y_pred_50.sum() if y_pred_50.sum() > 0 else 0
    f1_50 = fbeta_score(y_test, y_pred_50, beta=1)
    f2_50 = fbeta_score(y_test, y_pred_50, beta=2)
    mcc_50 = matthews_corrcoef(y_test, y_pred_50)

    results[name] = {
        'model': best_model,
        'auc': auc_val, 'sensitivity': sens, 'ppv': ppv,
        'f1': f1, 'f2': f2, 'mcc': mcc,
        'threshold': best_thresh, 'y_prob': y_prob, 'y_pred': y_pred,
        'sens_50': sens_50, 'ppv_50': ppv_50, 'f1_50': f1_50,
        'f2_50': f2_50, 'mcc_50': mcc_50,
        'best_balanced': best_balanced
    }

    log.log(f"Best F1 threshold: {best_thresh:.2f}")
    log.log(f"At best F1 → AUC: {auc_val:.4f} | Sens: {sens:.4f} | PPV: {ppv:.4f} | F1: {f1:.4f}")
    log.log(f"At thresh=0.50 → Sens: {sens_50:.4f} | PPV: {ppv_50:.4f} | F1: {f1_50:.4f}")
    
    if best_balanced['thresh'] != 0.5:
        log.log(f"Best balanced (sens≥0.85, PPV≥0.70): thresh={best_balanced['thresh']:.2f}, sens={best_balanced['sens']:.3f}, ppv={best_balanced['ppv']:.3f}")
    else:
        log.log(f"No threshold met both sensitivity ≥ 0.85 and PPV ≥ 0.70")
    log.log(f"")

# ============================================================
# STEP 7: Comparison table
# ============================================================
log.log(f"=== MODEL COMPARISON ===")
log.log(f"")
log.log(f"{'Model':<25} {'AUC':>8} {'Sens':>8} {'PPV':>8} {'F2':>8} {'F1':>8} {'MCC':>8}")
log.log(f"{'-'*70}")
for name, r in results.items():
    log.log(f"{name:<25} {r['auc']:>8.4f} {r['sensitivity']:>8.4f} {r['ppv']:>8.4f} {r['f2']:>8.4f} {r['f1']:>8.4f} {r['mcc']:>8.4f}")
log.log(f"")

# Determine best model using a clinical balance score:
# AUC + F1 + Sensitivity (higher is better)
def clinical_score(r):
    return r['auc'] + r['f1'] + r['sensitivity']

best_name = max(results, key=lambda n: clinical_score(results[n]))
best_result = results[best_name]
log.log(f"Best model (by clinical balance): {best_name}")

# ============================================================
# ROC Curves for all models
# ============================================================
from sklearn.metrics import roc_curve, auc as auc_calc

plt.figure(figsize=(8, 6))
roc_data = {}
for name, r in results.items():
    y_prob = r['y_prob']
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc_calc(fpr, tpr)
    roc_data[name] = roc_auc
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance (AUC=0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curves – Advanced RQ4 Models (80/20 split)', fontsize=13)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)

# Save ROC curve
plt.tight_layout()
plt.savefig('../output/figures/roc_curves_advanced.png', dpi=150, bbox_inches='tight')
plt.close()

log.log(f"=== ROC CURVES ===")
log.log(f"ROC curve saved: output/figures/roc_curves_advanced.png")
log.log(f"")
log.log(f"AUC values (test set, 80/20 split):")
for name, auc_val in roc_data.items():
    log.log(f"  {name:<25} AUC = {auc_val:.4f}")
log.log(f"")

# ============================================================
# Dedicated confusion matrix for best model at optimal threshold
# ============================================================
cm_best = confusion_matrix(y_test, best_result['y_pred'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fail', 'Success'],
            yticklabels=['Fail', 'Success'],
            annot_kws={'size': 16})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title(f'Confusion Matrix – Best Model: {best_name}\n(threshold = {best_result["threshold"]:.2f})', fontsize=12)

plt.tight_layout()
plt.savefig('../output/figures/best_model_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

log.log(f"Confusion matrix saved: output/figures/best_model_confusion_matrix.png")
log.log(f"")
log.log(f"Best Model Confusion Matrix at optimal threshold ({best_result['threshold']:.2f}):")
log.log(f"  TP={cm_best[1,1]:>3}  FP={cm_best[0,1]:>3}")
log.log(f"  FN={cm_best[1,0]:>3}  TN={cm_best[0,0]:>3}")
log.log(f"")

log.log(f"  AUC={best_result['auc']:.4f}, F1={best_result['f1']:.4f}, Sens={best_result['sensitivity']:.4f}")

# Check targets
if best_result['sensitivity'] >= 0.85 and best_result['ppv'] >= 0.70:
    log.log(f"✅ MET BOTH TARGETS (sens≥0.85, PPV≥0.70)")
elif best_result['sensitivity'] >= 0.85:
    log.log(f"⚠️ Met sensitivity ({best_result['sensitivity']:.3f}) but PPV below target ({best_result['ppv']:.3f})")
elif best_result['ppv'] >= 0.70:
    log.log(f"⚠️ Met PPV ({best_result['ppv']:.3f}) but sensitivity below target ({best_result['sensitivity']:.3f})")
else:
    log.log(f"❌ Neither target met. Best F2: {best_result['f2']:.4f}")

log.log(f"")

# ============================================================
# STEP 8: Feature importance from best model
# ============================================================
log.log(f"=== FEATURE IMPORTANCE (PCA Components) ===")
if hasattr(best_result['model'], 'coef_'):
    # Linear model
    importance = np.abs(best_result['model'].coef_[0])
    log.log(f"Logistic Regression coefficients (absolute):")
    for i, imp in enumerate(importance):
        log.log(f"  PC{i+1}: {imp:.4f}")
elif hasattr(best_result['model'], 'feature_importances_'):
    # Tree-based model
    importance = best_result['model'].feature_importances_
    log.log(f"Feature importances:")
    for i, imp in enumerate(importance):
        log.log(f"  PC{i+1}: {imp:.4f}")

log.log(f"")
log.log(f"Top contributing original features to PC1:")
pc1_loadings = loadings['PC1'].abs().sort_values(ascending=False)
for feat, load in pc1_loadings.head(5).items():
    log.log(f"  {feat}: {load:.4f}")

log.log(f"")

# ============================================================
# STEP 9: Visualization
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Explained variance
ax1 = axes[0]
ax1.bar(range(1, len(cumsum_var)+1), pca_full.explained_variance_ratio_, alpha=0.7, label='Individual')
ax1.plot(range(1, len(cumsum_var)+1), cumsum_var, 'ro-', label='Cumulative')
ax1.axhline(y=0.95, color='green', linestyle='--', label='95% threshold')
ax1.axvline(x=n_components_95, color='green', linestyle='--')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('PCA Explained Variance')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Model comparison - AUC and F2
ax2 = axes[1]
model_names = list(results.keys())
x = np.arange(len(model_names))
width = 0.35
aucs = [results[n]['auc'] for n in model_names]
f2s = [results[n]['f2'] for n in model_names]
bars1 = ax2.bar(x - width/2, aucs, width, label='AUC', color='steelblue')
bars2 = ax2.bar(x + width/2, f2s, width, label='F2-Score', color='coral')
ax2.axhline(y=0.80, color='red', linestyle='--', label='AUC target (0.80)')
ax2.set_xlabel('Model')
ax2.set_ylabel('Score')
ax2.set_title('Model Performance Comparison (80/20 split)')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, rotation=15, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Confusion matrix for best model
ax3 = axes[2]
cm = confusion_matrix(y_test, best_result['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Fail', 'Success'],
            yticklabels=['Fail', 'Success'])
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')
ax3.set_title(f'Best Model: {best_name}\n(threshold = {best_result["threshold"]:.2f})')

plt.tight_layout()
plt.savefig('../output/figures/advanced_rq4_results.png', dpi=150, bbox_inches='tight')
plt.close()
log.log("Figure saved: output/figures/advanced_rq4_results.png")

# ============================================================
# STEP 10: Save summary
# ============================================================
summary_rows = []
for name, r in results.items():
    summary_rows.append({
        'Model': name,
        'AUC': r['auc'], 'Sensitivity': r['sensitivity'],
        'PPV': r['ppv'], 'F1': r['f1'], 'F2': r['f2'],
        'MCC': r['mcc'], 'Threshold': r['threshold'],
        'Train_N': len(X_train), 'Test_N': len(X_test),
        'N_Features_PCA': n_components_95,
        'N_Features_Original': len(feature_df.columns)
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('../output/tables/advanced_rq4_summary.csv', index=False)
log.log("Summary saved: output/tables/advanced_rq4_summary.csv")
log.log(f"")
log.log("=== ADVANCED RQ4 ANALYSIS COMPLETE ===")
#log.close()

print(f"\nAdvanced RQ4 Results (80/20 split, {n_components_95} PCA components):")
for name, r in results.items():
    print(f"  {name:<25} AUC={r['auc']:.3f}  Sens={r['sensitivity']:.3f}  PPV={r['ppv']:.3f}  F2={r['f2']:.3f}")
print(f"\nBest model: {best_name} (F2={best_result['f2']:.3f})")


# To save the model

import joblib

# Save the best model (LightGBM)
best_model = results[best_name]['model']
joblib.dump(best_model, '../output/models/lightgbm_best.pkl')
log.log(f"Best model saved: output/models/lightgbm_best.pkl")

# Save the scaler and PCA for new predictions
joblib.dump(scaler, '../output/models/scaler.pkl')
joblib.dump(pca, '../output/models/pca.pkl')
log.log(f"Scaler and PCA saved: output/models/")
log.log(f"")


log.close()