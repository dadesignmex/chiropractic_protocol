"""
03_rq1_vas_change.py
RQ1: Mean change in VAS, MCID and SCB proportions.
"""
import pandas as pd
import numpy as np
from scipy import stats
from logFileHandler import Logger
log = Logger('../processLog.txt', '03_rq1_vas_change')  

df = pd.read_csv('../data/clinical_data.csv')

# Paired t-test
t_stat, p_val = stats.ttest_rel(df['vas_pre'], df['vas_post'])
log.log(f"Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")

# Wilcoxon as robustness check
w_stat, w_p = stats.wilcoxon(df['vas_pre'], df['vas_post'])
log.log(f"Wilcoxon: W={w_stat:.1f}, p={w_p:.4f}")

# Effect size (Cohen's d for paired)
d = (df['vas_pre'] - df['vas_post']).mean() / (df['vas_pre'] - df['vas_post']).std()
log.log(f"Cohen's d: {d:.2f}")

# MCID (>=30% reduction)
mcid = ((df['vas_pre'] - df['vas_post']) / df['vas_pre']) >= 0.30
log.log(f"MCID (>=30%): {mcid.sum()}/{len(df)} = {mcid.mean()*100:.1f}%")

# SCB (>=50% reduction)
scb = ((df['vas_pre'] - df['vas_post']) / df['vas_pre']) >= 0.50
log.log(f"SCB (>=50%): {scb.sum()}/{len(df)} = {scb.mean()*100:.1f}%")

log.close()