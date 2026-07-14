"""
00_import_raw.py
Reads raw Excel file with PRE and POST sheets.
Locates the header row (containing 'Patient ID#').
Fixes duplicate and misnamed columns.
Merges sheets and saves a raw merged CSV.
"""
import pandas as pd

from logFileHandler import Logger
log = Logger('../processLog.txt', '00_import_raw')    

RAW_FILE = "D:/DBA/NewChiropracticProtocol-Study/data/RAW_DATA.xlsx"  
OUTPUT_FILE = "../data/raw_merged.csv"

def read_sheet_skip_to_header(file_path, sheet_name, header_keyword='Patient ID#'):
    """
    Read an Excel sheet, searching for the row that contains header_keyword.
    Returns a DataFrame with proper column names and data.
    """
    # First, read the entire sheet without headers
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    
    # Find the row index that contains the keyword in any cell
    header_row_idx = None
    for idx, row in raw.iterrows():
        if row.astype(str).str.contains(header_keyword, case=False).any():
            header_row_idx = idx
            break
    
    if header_row_idx is None:
        raise ValueError(f"Could not find header row containing '{header_keyword}' in sheet '{sheet_name}'")
    
    # Set that row as column names, and take data from the next row onward
    raw.columns = raw.iloc[header_row_idx].fillna('').astype(str).str.strip()
    df = raw.iloc[header_row_idx + 1:].reset_index(drop=True)
    
    # Drop any completely empty rows
    df.dropna(how='all', inplace=True)
    
    return df

# Read both sheets using the header-finding function
pre = read_sheet_skip_to_header(RAW_FILE, 'PRE TREATMENT DATA')
post = read_sheet_skip_to_header(RAW_FILE, 'POST TREATMENT DATA')

# Strip whitespace from column names (safety)
pre.columns = pre.columns.str.strip()
post.columns = post.columns.str.strip()

log.log("PRE columns:", list(pre.columns))
log.log("POST columns:", list(post.columns))

# ---- Fix POST sheet specific issues ----
# 1. Duplicate column: second 'LEFT ROTATION MOTION LIMIT POST' -> 'LEFT ROTATION PAIN LEVEL POST'
cols = post.columns.tolist()
seen = {}
for i, col in enumerate(cols):
    if col in seen:
        # It's a duplicate – assume it should be PAIN LEVEL
        cols[i] = 'LEFT ROTATION PAIN LEVEL POST'
    else:
        seen[col] = i
post.columns = cols

# 2. Misnamed column: 'SENSORY LEFT L4 PRE' -> 'SENSORY LEFT L4 POST'
post.rename(columns={'SENSORY LEFT L4 PRE': 'SENSORY LEFT L4 POST'}, inplace=True)

# ---- Clean data values ----
for df in [pre, post]:
    # For all object columns: strip whitespace and convert to uppercase
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip().str.upper()
    # Fix date column: remove any spaces (e.g., "21/11/ 1963" -> "21/11/1963")
    for col in df.columns:
        if 'D.O.B' in col.upper():
            df[col] = df[col].str.replace(' ', '', regex=False)
            break

# Merge on Patient ID#
merged = pd.merge(pre, post, on='Patient ID#', suffixes=('_PRE', '_POST'))

log.log(f"Merged data shape: {merged.shape}")
log.log(f"Merged columns ({len(merged.columns)}):")
for c in merged.columns:
    log.log(f"  {c}")

# Save
merged.to_csv(OUTPUT_FILE, index=False)
log.log(f"\nRaw merged data saved to {OUTPUT_FILE} with {len(merged)} patients.")