"""
01_preprocess.py
Loads raw_merged.csv, derives all composite variables, 
and saves clinical_data.csv.
Note: The condition variable (free-text complaint) is retained for descriptive 
purposes only. It is excluded from RQ3 and RQ4 models due to insufficient 
variability (96.7% disc herniation-related complaints).
"""
import pandas as pd
import numpy as np
from utils import derive_max_pain, derive_max_limit, derive_neuro_deficit
from logFileHandler import Logger

log = Logger('../processLog.txt', '01_preprocess')

df = pd.read_csv('../data/raw_merged.csv')

# Strip whitespace from column names again (safety)
df.columns = df.columns.str.strip()

# ---- Age calculation ----
dob_col = [c for c in df.columns if 'D.O.B' in c.upper()][0]
df['DOB'] = pd.to_datetime(df[dob_col], dayfirst=True, errors='coerce')
df['first_consult'] = pd.to_datetime('2024-01-01')  # placeholder midpoint
df['age'] = (df['first_consult'] - df['DOB']).dt.days // 365

# ---- Condition type (retained for descriptive reference only) ----
# Excluded from RQ3/RQ4 due to insufficient variability in the sample.
# 96.7% of patients presented with disc herniation-related complaints.
complaint_col = [c for c in df.columns if 'COMPLAINT' in c.upper()][0]
df['condition_raw'] = df[complaint_col].str.strip().str.upper()
df['condition'] = df['condition_raw']

log.log(f"Condition variable retained for descriptive reference only.")
log.log(f"Unique complaints: {df['condition_raw'].nunique()}")
log.log(f"")

# ---- VAS ----
df['vas_pre'] = pd.to_numeric(df['VAS PRE'], errors='coerce')
df['vas_post'] = pd.to_numeric(df['VAS POST'], errors='coerce')

# Check missing
missing_pre = df['vas_pre'].isna().sum()
missing_post = df['vas_post'].isna().sum()
log.log(f"VAS PRE missing before fill: {missing_pre}")
log.log(f"VAS POST missing before fill: {missing_post}")

# Fill missing with median
median_pre = df['vas_pre'].median()
median_post = df['vas_post'].median()
df['vas_pre'] = df['vas_pre'].fillna(median_pre)
df['vas_post'] = df['vas_post'].fillna(median_post)

log.log(f"VAS PRE missing after fill: {df['vas_pre'].isna().sum()}")
log.log(f"VAS POST missing after fill: {df['vas_post'].isna().sum()}")

# ---- Composite variables ----
# Find columns by content patterns
pre_pain_cols = [c for c in df.columns if 'PAIN LEVEL' in c and ('PRE' in c.upper()) and 'POST' not in c.upper()]
post_pain_cols = [c for c in df.columns if 'PAIN LEVEL' in c and 'POST' in c.upper()]

pre_limit_cols = [c for c in df.columns if 'MOTION LIMIT' in c and ('PRE' in c.upper()) and 'POST' not in c.upper()]
post_limit_cols = [c for c in df.columns if 'MOTION LIMIT' in c and 'POST' in c.upper()]

# Derive max pain
def max_ordinal_from_cols(row, cols, ordered_labels):
    values = []
    for c in cols:
        val = row.get(c)
        if pd.notna(val):
            val_clean = str(val).strip().upper()
            for lbl in ordered_labels:
                if lbl.upper() == val_clean:
                    values.append(lbl)
                    break
    if not values:
        return None
    return max(values, key=lambda v: ordered_labels.index(v))

PAIN_LABELS  = ['NONE', 'MILD', 'MOD', 'SEVERE']
LIMIT_LABELS = ['WNL', 'MILD', 'MOD', 'SEVERE', 'FUL']

df['maxpain_pre']  = df.apply(lambda r: max_ordinal_from_cols(r, pre_pain_cols, PAIN_LABELS), axis=1)
df['maxpain_post'] = df.apply(lambda r: max_ordinal_from_cols(r, post_pain_cols, PAIN_LABELS), axis=1)
df['maxlimit_pre']  = df.apply(lambda r: max_ordinal_from_cols(r, pre_limit_cols, LIMIT_LABELS), axis=1)
df['maxlimit_post'] = df.apply(lambda r: max_ordinal_from_cols(r, post_limit_cols, LIMIT_LABELS), axis=1)

# Neurological deficit
sensory_cols = [c for c in df.columns if 'SENSORY' in c.upper()]
motor_cols = [c for c in df.columns if 'MOTOR' in c.upper()]
dtr_cols = [c for c in df.columns if 'DTR' in c.upper()]

def neuro_deficit_from_row(row, sensory_cols, motor_cols, dtr_cols):
    # Sensory
    for c in sensory_cols:
        val = row.get(c)
        if pd.notna(val) and str(val).strip().upper() not in ['WNL', 'NORMAL', '', 'NAN']:
            return 1
    # Motor
    for c in motor_cols:
        val = row.get(c)
        if pd.notna(val):
            try:
                if int(float(val)) < 5:
                    return 1
            except (ValueError, TypeError):
                if str(val).strip() == '-':
                    return 1
    # DTR
    for c in dtr_cols:
        val = row.get(c)
        if pd.notna(val):
            try:
                r = int(float(val))
                if r < 2 or r > 2:
                    return 1
            except (ValueError, TypeError):
                pass
    return 0

# Separate pre and post neuro columns
sensory_pre = [c for c in sensory_cols if 'PRE' in c.upper() and 'POST' not in c.upper()]
sensory_post = [c for c in sensory_cols if 'POST' in c.upper()]
motor_pre = [c for c in motor_cols if 'PRE' in c.upper() and 'POST' not in c.upper()]
motor_post = [c for c in motor_cols if 'POST' in c.upper()]
dtr_pre = [c for c in dtr_cols if 'PRE' in c.upper() and 'POST' not in c.upper()]
dtr_post = [c for c in dtr_cols if 'POST' in c.upper()]

df['neuro_def_pre'] = df.apply(lambda r: neuro_deficit_from_row(r, sensory_pre, motor_pre, dtr_pre), axis=1)
df['neuro_def_post'] = df.apply(lambda r: neuro_deficit_from_row(r, sensory_post, motor_post, dtr_post), axis=1)

# ---- Anonymised patient ID ----
df['patient_id'] = [f'P{i:03d}' for i in range(1, len(df)+1)]

# ---- Proxy surgical candidate ----
df['surgical_candidate'] = ((df['vas_pre'] >= 7) | (df['neuro_def_pre'] == 1)).astype(int)

# ---- Composite successful outcome ----
df['pct_reduction'] = (df['vas_pre'] - df['vas_post']) / df['vas_pre']
df['success'] = ((df['pct_reduction'] >= 0.50) & (df['neuro_def_post'] == 0)).astype(int)

# ---- Select and save final dataset ----
analysis_cols = [
    'patient_id', 'age', 'condition', 'condition_raw',
    'vas_pre', 'vas_post',
    'maxpain_pre', 'maxpain_post',
    'maxlimit_pre', 'maxlimit_post',
    'neuro_def_pre', 'neuro_def_post',
    'surgical_candidate', 'success'
]

# Keep raw columns for transparency
raw_cols = [c for c in df.columns if any(x in c.upper() for x in ['SENSORY', 'MOTOR', 'DTR', 'KEMPS', 'FABER', 'SLR'])]

final_cols = analysis_cols + raw_cols
df[final_cols].to_csv('../data/clinical_data.csv', index=False)

log.log(f"clinical_data.csv saved with {len(df)} patients and {len(final_cols)} columns.")
log.log(f"")
log.log(f"Derived variables summary:")
log.log(f"  maxpain_pre:  {df['maxpain_pre'].value_counts().to_dict()}")
log.log(f"  maxpain_post: {df['maxpain_post'].value_counts().to_dict()}")
log.log(f"  neuro_def_pre:  {df['neuro_def_pre'].value_counts().to_dict()}")
log.log(f"  neuro_def_post: {df['neuro_def_post'].value_counts().to_dict()}")
log.log(f"  surgical_candidate: {df['surgical_candidate'].value_counts().to_dict()}")
log.log(f"  success: {df['success'].value_counts().to_dict()}")

log.close()