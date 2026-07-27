"""
synthetic_data_generator.py
============================================================
Generates 70 synthetic patient records from the statistical
distributions of 90 real patients using Gaussian Copula synthesis.

Method:
  1. Transform each real variable to uniform via empirical CDF.
  2. Transform uniform to standard normal (probit).
  3. Fit a multivariate normal distribution (mean + covariance)
     to capture the joint distribution of all variables.
  4. Draw 70 new samples from this multivariate normal.
  5. Reverse transformations (normal → uniform → original scale).
  6. Add controlled noise to continuous variables.
  7. Round discrete variables to valid clinical categories.
  8. Combine real + synthetic into a 160-patient dataset.

Output:
  - data/clinical_data_augmented.csv (90 real + 70 synthetic)
  - Validation report in processLog.txt
============================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from logFileHandler import Logger

log = Logger('../processLog.txt', 'synthetic_gen')

# ---- 1. Load real data ----
real = pd.read_csv('../data/clinical_data.csv')
log.log(f"Real patients loaded: {len(real)}")

# ---- 2. Variables to synthesise ----
synth_vars = [
    'age', 'vas_pre', 'vas_post',
    'maxpain_pre', 'maxpain_post',
    'maxlimit_pre', 'maxlimit_post',
    'neuro_def_pre', 'neuro_def_post',
    'surgical_candidate', 'success'
]

# ---- 3. Map ordinals to numeric ----
pain_map = {'NONE': 0, 'MILD': 1, 'MOD': 2, 'SEVERE': 3, None: np.nan}
limit_map = {'WNL': 0, 'MILD': 1, 'MOD': 2, 'SEVERE': 3, 'FUL': 4, None: np.nan}

real_synth = real[synth_vars].copy()
real_synth['maxpain_pre_num']  = real_synth['maxpain_pre'].map(pain_map)
real_synth['maxpain_post_num'] = real_synth['maxpain_post'].map(pain_map)
real_synth['maxlimit_pre_num'] = real_synth['maxlimit_pre'].map(limit_map)
real_synth['maxlimit_post_num']= real_synth['maxlimit_post'].map(limit_map)

# Numeric columns for copula
numeric_cols = [
    'age', 'vas_pre', 'vas_post',
    'maxpain_pre_num', 'maxpain_post_num',
    'maxlimit_pre_num', 'maxlimit_post_num',
    'neuro_def_pre', 'neuro_def_post',
    'surgical_candidate', 'success'
]

real_numeric = real_synth[numeric_cols].dropna()
log.log(f"Complete cases for synthesis: {len(real_numeric)}")

# ---- 4. Gaussian Copula fit ----
# Step 4a: Uniform via empirical CDF
real_uniform = pd.DataFrame()
for col in numeric_cols:
    noisy = real_numeric[col] + np.random.normal(0, 0.01, len(real_numeric))
    real_uniform[col] = stats.rankdata(noisy) / (len(noisy) + 1)

# Step 4b: Normal via probit
real_normal = pd.DataFrame()
for col in numeric_cols:
    real_normal[col] = stats.norm.ppf(np.clip(real_uniform[col], 0.001, 0.999))

# Step 4c: Mean and covariance
mean_vec = real_normal.mean().values
cov_mat  = real_normal.cov().values
log.log("Gaussian Copula fitted.")

# ---- 5. Generate synthetic samples ----
n_synthetic = 70
np.random.seed(42)

synth_normal = np.random.multivariate_normal(mean_vec, cov_mat, size=n_synthetic)
synth_normal_df = pd.DataFrame(synth_normal, columns=numeric_cols)

# Transform back: normal → uniform → original scale
synth_uniform = stats.norm.cdf(synth_normal_df)
synth_numeric = pd.DataFrame()
for col in numeric_cols:
    real_vals = real_numeric[col].values
    synth_numeric[col] = np.quantile(real_vals,
                                     np.clip(synth_uniform[col].values, 0.001, 0.999))

# ---- 6. Controlled noise ----
for col in ['age', 'vas_pre', 'vas_post']:
    noise_scale = real_numeric[col].std() * 0.05
    synth_numeric[col] += np.random.normal(0, noise_scale, n_synthetic)
    if col == 'age':
        synth_numeric[col] = np.clip(synth_numeric[col], 18, 90).round().astype(int)
    else:
        synth_numeric[col] = np.clip(synth_numeric[col], 0, 10).round().astype(int)

# Round discrete variables
for col in ['maxpain_pre_num', 'maxpain_post_num']:
    synth_numeric[col] = np.clip(np.round(synth_numeric[col]), 0, 3).astype(int)
for col in ['maxlimit_pre_num', 'maxlimit_post_num']:
    synth_numeric[col] = np.clip(np.round(synth_numeric[col]), 0, 4).astype(int)
for col in ['neuro_def_pre', 'neuro_def_post', 'surgical_candidate', 'success']:
    synth_numeric[col] = np.clip(np.round(synth_numeric[col]), 0, 1).astype(int)

# ---- 7. Map back to labels ----
pain_rev  = {0: 'NONE', 1: 'MILD', 2: 'MOD', 3: 'SEVERE'}
limit_rev = {0: 'WNL', 1: 'MILD', 2: 'MOD', 3: 'SEVERE', 4: 'FUL'}

synth_final = synth_numeric.copy()
synth_final['maxpain_pre']  = synth_numeric['maxpain_pre_num'].map(pain_rev)
synth_final['maxpain_post'] = synth_numeric['maxpain_post_num'].map(pain_rev)
synth_final['maxlimit_pre'] = synth_numeric['maxlimit_pre_num'].map(limit_rev)
synth_final['maxlimit_post']= synth_numeric['maxlimit_post_num'].map(limit_rev)

synth_final = synth_final[['age','vas_pre','vas_post',
                           'maxpain_pre','maxpain_post',
                           'maxlimit_pre','maxlimit_post',
                           'neuro_def_pre','neuro_def_post',
                           'surgical_candidate','success']]

# ---- 8. Add metadata ----
synth_final['patient_id'] = [f'SYNTH_{i:03d}' for i in range(1, n_synthetic+1)]
real_cond = real['condition_raw'].dropna()
cond_probs = real_cond.value_counts(normalize=True)
synth_final['condition_raw'] = np.random.choice(cond_probs.index, size=n_synthetic, p=cond_probs.values)
synth_final['condition'] = 'See condition_raw'
synth_final['source'] = 'synthetic'

# ---- 9. Combine and save ----
real_out = real.copy()
real_out['source'] = 'real'

combined = pd.concat([real_out, synth_final], ignore_index=True)
combined.to_csv('../data/clinical_data_augmented.csv', index=False)
log.log(f"Augmented dataset saved: {len(combined)} patients (90 real + 70 synthetic)")

# ---- 10. Validation report ----
log.log("\n=== SYNTHETIC DATA VALIDATION ===")
log.log(f"{'Variable':<20} {'Real Mean':>10} {'Synth Mean':>10}")
for col in ['age','vas_pre','vas_post']:
    log.log(f"{col:<20} {real_out[col].mean():>10.2f} {synth_final[col].mean():>10.2f}")
for col in ['neuro_def_pre','surgical_candidate','success']:
    log.log(f"{col:<20} {real_out[col].mean():>10.3f} {synth_final[col].mean():>10.3f}")
log.log("Validation complete.")
log.close()