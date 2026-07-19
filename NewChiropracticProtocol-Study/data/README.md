\# New Chiropractic Protocol – Retrospective Cohort Study



\*\*Comparative Effectiveness of the New Chiropractic Protocol for Lumbar

Discogenic Conditions: A Retrospective Cohort Study Benchmarking Clinical

Outcomes Against Published Surgical Results and Developing a Predictive

Model of Treatment Success.\*\*



\## Repository Structure



| Folder    | Description |

|-----------|-------------|

| `data/`   | De-identified dataset (currently 20-patient pilot; full N≈160 to be added) and data dictionary. |

| `code/`   | Python scripts for data import, preprocessing, descriptive statistics, and all four research questions. |

| `output/` | Generated figures and tables. |

| `report/` | Final manuscript (if included). |



\## Data



\*\*Current status:\*\* 20 de-identified patients are included for code testing

and preliminary analysis. The full dataset of approximately 160 patients

will be uploaded after final data collection and cleaning.



\- `clinical\_data.csv` – Cleaned, analysis-ready dataset (one row per patient).

\- `data\_dictionary.csv` – Variable names, descriptions, types, and derivation rules.



Raw source data (Excel with two sheets) is NOT stored in this repository.

It is processed locally by `code/00\_import\_raw.py`.



\## Code



All analyses are written in \*\*Python 3.10+\*\* .



\### Dependencies



```bash

pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn shap openpyxl

