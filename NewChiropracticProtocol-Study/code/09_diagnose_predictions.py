"""
09_diagnose_predictions.py
Diagnostic: Confusion matrices for all models at threshold = 0.50.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('../data/clinical_data.csv')

# Build features (same as 08)
limit_map = {'WNL': 0, 'MILD': 1, 'MOD': 2, 'SEVERE': 3, 'FUL': 3}
pain_map = {'NONE': 0, 'MILD': 1, 'MOD': 2, 'SEVERE': 3}

feature_dict = {}
feature_dict['age'] = df['age']
feature_dict['vas_pre'] = df['vas_pre']
feature_dict['maxlimit_num'] = df['maxlimit_pre'].map(limit_map)
feature_dict['neuro_def_pre'] = df['neuro_def_pre']
feature_dict['maxpain_num'] = df['maxpain_pre'].map(pain_map)

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

slr_left = pd.to_numeric(df['SLR DEGREES LEFT_PRE'], errors='coerce')
slr_right = pd.to_numeric(df['SLR DEGREES RIGHT_PRE'], errors='coerce')
feature_dict['slr_mean'] = (slr_left + slr_right) / 2
feature_dict['slr_min'] = np.minimum(slr_left, slr_right)
feature_dict['slr_asymmetry'] = np.abs(slr_left - slr_right)

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
y = df['success'].values

# Split first
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

# Fit scaler and PCA on training only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train_scaled)
X_test = pca.transform(X_test_scaled)

print(f"\n{'='*70}")
print(f"CONFUSION MATRICES AT THRESHOLD = 0.50")
print(f"Test set: {len(X_test)} patients ({y_test.sum()} successes, base rate = {y_test.sum()/len(y_test):.1%})")
print(f"{'='*70}\n")

# Best models
"""
PASADOS
models = {
    'Logistic Regression': LogisticRegression(C=0.5, penalty='l2', solver='saga', max_iter=2000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(max_depth=3, min_samples_leaf=3, min_samples_split=5, n_estimators=300, class_weight='balanced', random_state=42),
    'LightGBM': lgb.LGBMClassifier(learning_rate=0.05, max_depth=2, min_child_samples=20, n_estimators=100, num_leaves=7, reg_alpha=1.0, reg_lambda=0.1, objective='binary', random_state=42, verbose=-1),
    'SVM': SVC(C=10.0, gamma=0.01, kernel='rbf', class_weight='balanced', probability=True, random_state=42)
}
"""
# Best models from v1 re-run
models = {
    'Logistic Regression': LogisticRegression(
        C=0.5, penalty='l2', solver='saga', max_iter=2000,
        class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        max_depth=3, min_samples_leaf=3, min_samples_split=5,
        n_estimators=300, class_weight='balanced', random_state=42
    ),
    'LightGBM': lgb.LGBMClassifier(
        learning_rate=0.05, max_depth=2, min_child_samples=20,
        n_estimators=100, num_leaves=7, reg_alpha=1.0, reg_lambda=0.1,
        objective='binary', random_state=42, verbose=-1
    ),
    'SVM': SVC(
        C=1.0, gamma='scale', kernel='rbf',
        class_weight='balanced', probability=True, random_state=42
    )
}
# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    # Print to console
    print(f"--- {name} ---")
    print(f"              Predicted Fail  Predicted Success")
    print(f"Actual Fail        {tn:>5}            {fp:>5}")
    print(f"Actual Success     {fn:>5}            {tp:>5}")
    print(f"")
    print(f"Sensitivity: {sens:.3f} | PPV: {ppv:.3f} | Specificity: {spec:.3f} | Accuracy: {acc:.3f}")
    print(f"{classification_report(y_test, y_pred, target_names=['Fail', 'Success'])}")
    print(f"{'-'*50}\n")
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Fail', 'Success'],
                yticklabels=['Fail', 'Success'],
                annot_kws={'size': 14})
    axes[idx].set_xlabel('Predicted', fontsize=11)
    axes[idx].set_ylabel('Actual', fontsize=11)
    axes[idx].set_title(f'{name}\nSens={sens:.2f} | PPV={ppv:.2f} | Acc={acc:.2f}', fontsize=11)

plt.tight_layout()
plt.savefig('../output/figures/confusion_matrices_threshold_050.png', dpi=150, bbox_inches='tight')
plt.close()

print("Figure saved: output/figures/confusion_matrices_threshold_050.png")
print("Done.")