"""
02_descriptive_stats.py
Summary statistics, demographics, and baseline characteristics.
"""
import pandas as pd
from logFileHandler import Logger
log = Logger('../processLog.txt', '02_descriptive_stats') 

df = pd.read_csv('../data/clinical_data.csv')

log.log("=== DEMOGRAPHICS ===")
log.log(f"Total patients: {len(df)}")
log.log(f"Age: mean={df['age'].mean():.1f}, SD={df['age'].std():.1f}, range={df['age'].min()}-{df['age'].max()}")

log.log("\n=== CONDITION DISTRIBUTION ===")
log.log(df['condition_raw'].value_counts())

log.log("\n=== BASELINE VAS ===")
log.log(f"VAS PRE: mean={df['vas_pre'].mean():.1f}, SD={df['vas_pre'].std():.1f}")
log.log(f"VAS POST: mean={df['vas_post'].mean():.1f}, SD={df['vas_post'].std():.1f}")

log.log("\n=== SURGICAL CANDIDATES ===")
log.log(df['surgical_candidate'].value_counts())

log.log("\n=== COMPOSITE SUCCESS ===")
log.log(df['success'].value_counts())

log.log("\n=== NEUROLOGICAL DEFICIT ===")
log.log("PRE:", df['neuro_def_pre'].value_counts().to_dict())
log.log("POST:", df['neuro_def_post'].value_counts().to_dict())

log.close()
