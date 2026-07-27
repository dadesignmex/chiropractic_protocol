"""
02_eda.py
Exploratory Data Analysis for Interim Report.
Generates figures and logs key insights.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logFileHandler import Logger

log = Logger('../processLog.txt', '02_eda')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# ---- Load data ----
df = pd.read_csv('../data/clinical_data.csv')
log.log(f"=== EXPLORATORY DATA ANALYSIS ===")
log.log(f"Total patients: {len(df)}")

# ---- Figure 1: Age Distribution ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['age'], bins=15, color='steelblue', edgecolor='white')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Frequency')
ax.set_title('Figure 1: Age Distribution of Patients')
fig.savefig('../output/figures/fig1_age_distribution.png', bbox_inches='tight')
plt.close()
log.log("Figure 1 saved: Age distribution histogram")
log.log(f"  Mean age: {df['age'].mean():.1f} (SD={df['age'].std():.1f})")
log.log(f"  Range: {df['age'].min():.0f}–{df['age'].max():.0f}")

# ---- Figure 2: Pre vs Post VAS ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df['vas_pre'], bins=10, color='coral', edgecolor='white')
axes[0].set_xlabel('VAS Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Pre-Treatment VAS')
axes[1].hist(df['vas_post'], bins=10, color='seagreen', edgecolor='white')
axes[1].set_xlabel('VAS Score')
axes[1].set_title('Post-Treatment VAS')
fig.suptitle('Figure 2: Pre- and Post-Treatment VAS Distributions')
fig.tight_layout()
fig.savefig('../output/figures/fig2_vas_distributions.png', bbox_inches='tight')
plt.close()
log.log("Figure 2 saved: Pre/Post VAS distributions")
log.log(f"  Pre-treatment VAS: mean={df['vas_pre'].mean():.1f}, SD={df['vas_pre'].std():.1f}")
log.log(f"  Post-treatment VAS: mean={df['vas_post'].mean():.1f}, SD={df['vas_post'].std():.1f}")

# ---- Figure 3: VAS Change Box Plot ----
df['vas_change'] = df['vas_pre'] - df['vas_post']
fig, ax = plt.subplots(figsize=(6, 6))
ax.boxplot(df['vas_change'].dropna(), vert=True, patch_artist=True,
           boxprops=dict(facecolor='lightblue'))
ax.set_ylabel('VAS Reduction (Pre − Post)')
ax.set_title('Figure 3: VAS Change After Protocol')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
fig.savefig('../output/figures/fig3_vas_change_boxplot.png', bbox_inches='tight')
plt.close()
log.log("Figure 3 saved: VAS change box plot")
log.log(f"  Mean VAS reduction: {df['vas_change'].mean():.1f} (SD={df['vas_change'].std():.1f})")
log.log(f"  Median reduction: {df['vas_change'].median():.1f}")

# ---- Figure 4: MCID and SCB Bar Chart ----
mcid = (df['vas_change'] / df['vas_pre'] >= 0.30).sum()
scb  = (df['vas_change'] / df['vas_pre'] >= 0.50).sum()
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(['MCID (≥30%)', 'SCB (≥50%)'], [mcid, scb], color=['steelblue', 'darkorange'])
ax.set_ylabel('Number of Patients')
ax.set_title('Figure 4: Patients Achieving MCID and SCB')
for bar, val in zip(bars, [mcid, scb]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val),
            ha='center', fontweight='bold')
fig.savefig('../output/figures/fig4_mcid_scb.png', bbox_inches='tight')
plt.close()
log.log(f"Figure 4 saved: MCID={mcid}/{len(df)} ({mcid/len(df)*100:.1f}%), SCB={scb}/{len(df)} ({scb/len(df)*100:.1f}%)")

# ---- Figure 5: Surgical Candidates and Success ----
candidates = df['surgical_candidate'].sum()
successes = df['success'].sum()
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(['Surgical\nCandidates', 'Composite\nSuccess'],
              [candidates, successes], color=['indianred', 'seagreen'])
ax.set_ylabel('Number of Patients')
ax.set_title('Figure 5: Surgical Candidates and Composite Success')
for bar, val in zip(bars, [candidates, successes]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val),
            ha='center', fontweight='bold')
fig.savefig('../output/figures/fig5_candidates_success.png', bbox_inches='tight')
plt.close()
log.log(f"Figure 5 saved: Surgical candidates={candidates}, Composite success={successes}")

# ---- Figure 6: Success Rate Among Surgical Candidates ----
surg_df = df[df['surgical_candidate'] == 1]
surg_success = surg_df['success'].sum()
surg_fail = len(surg_df) - surg_success
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie([surg_success, surg_fail], labels=['Success', 'Fail'],
       autopct='%1.1f%%', colors=['seagreen', 'lightgray'],
       explode=(0.05, 0))
ax.set_title('Figure 6: Composite Success Among Surgical Candidates')
fig.savefig('../output/figures/fig6_success_pie.png', bbox_inches='tight')
plt.close()
log.log(f"Figure 6 saved: Surgical candidate success rate = {surg_success}/{len(surg_df)} ({surg_success/len(surg_df)*100:.1f}%)")

# ---- Figure 7: Neurological Deficit Pre vs Post ----
neuro_pre = df['neuro_def_pre'].value_counts()
neuro_post = df['neuro_def_post'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].bar(['No Deficit', 'Deficit'], neuro_pre.values, color=['seagreen', 'indianred'])
axes[0].set_title('Pre-Treatment')
axes[0].set_ylabel('Patients')
axes[1].bar(['No Deficit', 'Deficit'], neuro_post.values, color=['seagreen', 'indianred'])
axes[1].set_title('Post-Treatment')
fig.suptitle('Figure 7: Neurological Deficit Status')
fig.tight_layout()
fig.savefig('../output/figures/fig7_neuro_deficit.png', bbox_inches='tight')
plt.close()
log.log(f"Figure 7 saved: Neuro deficit PRE={neuro_pre.to_dict()}, POST={neuro_post.to_dict()}")

# ---- Figure 8: Max Pain Pre vs Post ----
pain_order = ['NONE', 'MILD', 'MOD', 'SEVERE']
pain_pre = df['maxpain_pre'].value_counts().reindex(pain_order, fill_value=0)
pain_post = df['maxpain_post'].value_counts().reindex(pain_order, fill_value=0)
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(pain_order))
w = 0.35
ax.bar(x - w/2, pain_pre.values, w, label='Pre-Treatment', color='coral')
ax.bar(x + w/2, pain_post.values, w, label='Post-Treatment', color='seagreen')
ax.set_xticks(x)
ax.set_xticklabels(pain_order)
ax.set_ylabel('Number of Patients')
ax.set_title('Figure 8: Maximum Pain on Movement')
ax.legend()
fig.savefig('../output/figures/fig8_maxpain.png', bbox_inches='tight')
plt.close()
log.log(f"Figure 8 saved: Max pain shift from pre to post")

# ---- EDA Summary ----
log.log(f"")
log.log(f"=== EDA COMPLETE ===")
log.log(f"Figures saved in output/figures/")
log.log(f"")
log.log(f"Key findings:")
log.log(f"  1. Mean VAS reduction: {df['vas_change'].mean():.1f} points (Cohen's d ≈ {(df['vas_change'].mean()/df['vas_change'].std()):.2f})")
log.log(f"  2. {mcid/len(df)*100:.1f}% achieved MCID; {scb/len(df)*100:.1f}% achieved SCB")
log.log(f"  3. {candidates/len(df)*100:.1f}% are surgical candidates")
log.log(f"  4. Composite success rate: {successes/len(df)*100:.1f}% overall; {surg_success/len(surg_df)*100:.1f}% among surgical candidates")
log.log(f"  5. Neurological deficit: {neuro_pre.get(1,0)} pre → {neuro_post.get(1,0)} post")

log.close()