"""
utils.py
Shared helper functions for deriving composite clinical variables.
"""
import pandas as pd

# Ordered severity maps
LIMIT_ORDER = {'WNL': 0, 'MILD': 1, 'MOD': 2, 'SEVERE': 3, 'FUL': 3}
PAIN_ORDER  = {'NONE': 0, 'MILD': 1, 'MOD': 2, 'SEVERE': 3}
LIMIT_LABELS = ['WNL', 'MILD', 'MOD', 'SEVERE', 'FUL']
PAIN_LABELS  = ['NONE', 'MILD', 'MOD', 'SEVERE']

MOVEMENTS = ['FLEXION', 'EXTENSION',
             'RIGHT LATERAL FLEXION', 'LEFT LATERAL FLEXION',
             'RIGHT ROTATION', 'LEFT ROTATION']


def _max_ordinal(row, cols, ordered_labels):
    """Return the worst (highest severity) label among the given columns."""
    values = []
    for c in cols:
        val = row.get(c)
        if pd.notna(val):
            val_clean = str(val).strip().upper()
            # map back to canonical label
            for lbl in ordered_labels:
                if lbl.upper() == val_clean:
                    values.append(lbl)
                    break
    if not values:
        return None
    return max(values, key=lambda v: ordered_labels.index(v))


def derive_max_pain(row, suffix='PRE'):
    """Worst pain across all six movement planes."""
    cols = [f'{mvt} PAIN LEVEL {suffix}' for mvt in MOVEMENTS]
    return _max_ordinal(row, cols, PAIN_LABELS)


def derive_max_limit(row, suffix='PRE'):
    """Worst motion limitation across all six movement planes."""
    cols = [f'{mvt} MOTION LIMIT {suffix}' for mvt in MOVEMENTS]
    return _max_ordinal(row, cols, LIMIT_LABELS)


def derive_neuro_deficit(row, suffix='PRE'):
    """Returns 1 if any sensory, motor, or reflex abnormality exists."""
    # Sensory
    for side in ['LEFT', 'RIGHT']:
        for level in ['L4', 'L5', 'S1']:
            col = f'SENSORY {side} {level} {suffix}'
            val = row.get(col)
            if pd.notna(val) and str(val).strip().upper() not in ['WNL', 'NORMAL', '']:
                return 1
    # Motor
    for side in ['LEFT', 'RIGHT']:
        for level in ['L1-3', 'L4', 'L5', 'S1']:
            col = f'MOTOR {side} {level} {suffix}'
            val = row.get(col)
            if pd.notna(val):
                try:
                    if int(val) < 5:
                        return 1
                except ValueError:
                    if str(val).strip() == '-':
                        return 1
    # Reflexes (DTR)
    for side in ['LEFT', 'RIGHT']:
        for level in ['L4', 'S1']:
            col = f'DTR {side} {level} {suffix}'
            val = row.get(col)
            if pd.notna(val):
                try:
                    r = int(val)
                    if r < 2 or r > 2:
                        return 1
                except ValueError:
                    pass
    return 0