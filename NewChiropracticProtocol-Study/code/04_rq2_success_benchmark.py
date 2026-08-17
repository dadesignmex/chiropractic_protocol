"""
04_rq2_success_benchmark.py
RQ2: Composite success rate among surgical candidates vs. 75% benchmark.
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from logFileHandler import Logger
log = Logger('../processLog.txt', '04_rq2_success_benchmark')  

df = pd.read_csv('../data/clinical_data.csv')
candidates = df[df['surgical_candidate'] == 1]
n = len(candidates)
k = candidates['success'].sum()
p_obs = k / n

log.log(f"Surgical candidates: {n}")
log.log(f"Composite success: {k}/{n} = {p_obs*100:.1f}%")

# Exact binomial CI
ci_low, ci_upp = proportion_confint(k, n, method='beta')
log.log(f"95% CI (Clopper-Pearson): [{ci_low*100:.1f}%, {ci_upp*100:.1f}%]")

# One-sample z-test against 75% benchmark
p0 = 0.75
z_stat = (p_obs - p0) / np.sqrt(p0 * (1 - p0) / n)
p_val = 1 - stats.norm.cdf(z_stat)
log.log(f"One-sample z-test vs 75%: z={z_stat:.3f}, p={p_val:.4f} (one-tailed)")

log.close()