"""
p1_inspect_uci.py  --  Task 6, Phase 1 (P1-01, P1-02): first look at the RAW UCI Diabetes file.

Put this file in  yakubu_health_fairness\\src\\  and run from the project folder:
    python src\\p1_inspect_uci.py

It only LOOKS at the data. Nothing is cleaned, dropped or changed, and the raw file is never modified.
Outputs go to outputs\\uci_diabetes\\phase1\\ (overview.csv, columns.csv, target_by_race_counts.csv).
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(ROOT, "data", "uci_diabetes", "raw", "diabetic_data.csv")
OUT = os.path.join(ROOT, "outputs", "uci_diabetes", "phase1")
os.makedirs(OUT, exist_ok=True)

if not os.path.exists(RAW):
    raise SystemExit("File not found: %s\nCopy diabetic_data.csv into data\\uci_diabetes\\raw first." % RAW)

print("=" * 70)
print("P1-01  Source file:", RAW)
print("File size: %.1f MB" % (os.path.getsize(RAW) / 1e6))

# Only the literal '?' is treated as missing. keep_default_na=False stops pandas from turning real
# categories such as 'None' (test not performed) into missing values.
df = pd.read_csv(RAW, na_values=["?"], keep_default_na=False)

print("=" * 70)
print("P1-02  Shape (rows, columns):", df.shape)
print("One row means: ONE HOSPITAL ENCOUNTER (not necessarily one patient).")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 14)
print("\nFirst rows (first 12 columns):")
print(df.iloc[:, :12].head())
print("\nColumn names (%d):" % df.shape[1])
print(list(df.columns))
print("\nColumn types:")
print(df.dtypes.value_counts().to_string())

print("=" * 70)
n_enc = df["encounter_id"].nunique()
n_pat = df["patient_nbr"].nunique()
print("Encounters (rows):", len(df), "| unique encounter_id:", n_enc)
print("Unique patients (patient_nbr):", n_pat)
print("Extra encounters from repeat patients:", len(df) - n_pat,
      "-> the train/test split must keep each patient on ONE side (no leakage).")

print("=" * 70)
print("Readmission labels (readmitted):")
print(df["readmitted"].value_counts(dropna=False).to_string())
df["target_preview"] = (df["readmitted"] == "<30").astype(int)
print("\nProposed target (1 = readmitted within 30 days '<30', 0 = 'NO' or '>30'):")
print(df["target_preview"].value_counts().to_string())

print("=" * 70)
print("Race labels (counts, including missing '?'):")
print(df["race"].value_counts(dropna=False).to_string())
tab = pd.crosstab(df["race"].fillna("MISSING ('?')"), df["target_preview"], margins=True)
tab.columns = ["not_readmitted_30d", "readmitted_30d", "total"] if tab.shape[1] == 3 else tab.columns
tab["pct_readmitted_30d"] = (100.0 * tab["readmitted_30d"] / tab["total"]).round(2)
print("\nReadmission within 30 days by race (counts, not only percentages):")
print(tab.to_string())

# ---- save evidence files
overview = pd.DataFrame([
    {"item": "rows (encounters)", "value": len(df)},
    {"item": "columns", "value": df.shape[1]},
    {"item": "unique encounter_id", "value": n_enc},
    {"item": "unique patient_nbr", "value": n_pat},
    {"item": "extra encounters from repeat patients", "value": len(df) - n_pat},
    {"item": "readmitted <30 (target = 1)", "value": int(df["target_preview"].sum())},
    {"item": "readmitted NO or >30 (target = 0)", "value": int((1 - df["target_preview"]).sum())},
])
overview.to_csv(os.path.join(OUT, "overview.csv"), index=False)

cols = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes],
                     "missing_question_mark": df.isna().sum().values,
                     "n_unique": df.nunique(dropna=True).values})
cols.to_csv(os.path.join(OUT, "columns.csv"), index=False)
tab.reset_index().rename(columns={"race": "race"}).to_csv(os.path.join(OUT, "target_by_race_counts.csv"), index=False)

print("=" * 70)
print("Columns with the most missing values ('?'):")
print(cols.sort_values("missing_question_mark", ascending=False).head(8)[["column", "missing_question_mark"]].to_string(index=False))
print("=" * 70)
print("Saved in", os.path.relpath(OUT, ROOT), ": overview.csv, columns.csv, target_by_race_counts.csv")
print("DONE. Take your screenshots (raw folder, this terminal output).")
