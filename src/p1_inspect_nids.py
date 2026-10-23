"""
p1_inspect_nids.py  --  Task 6, Phase 1 (P1-01, P1-02): first look at the RAW NIDS Wave 5 Stata files.

Run from the project folder:
    python src\\p1_inspect_nids.py

It only inspects the files (nothing is cleaned, dropped or recoded). The health outcome is not chosen here;
it is selected from the codebook and approved separately. What it does:
  * finds the adult and individual-derived .dta files in data\\nids\\raw,
  * reports rows, columns and whether pid is unique,
  * counts the documented population group w5_best_race (raw codes + labels),
  * tests the one-to-one join on pid and reports the result honestly,
  * lists health-related variables (by keyword, using the codebook labels stored in the file) with how many
    answers are valid and how many are special codes (negative numbers such as refused / don't know),
    so you can propose a target from the REAL column names.
Outputs go to outputs\\nids\\phase1\\ . The raw files are never modified.
"""
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(ROOT, "data", "nids", "raw")
OUT = os.path.join(ROOT, "outputs", "nids", "phase1")
os.makedirs(OUT, exist_ok=True)

KEYWORDS = ["health", "hypertens", "blood pressure", "diabet", "doctor", "diagnos", "sugar", "cholesterol",
            "asthma", "cancer", "heart", "stroke", "tb ", "tubercul", "hiv", "smok", "alcohol", "disab",
            "illness", "sick", "clinic", "hospital", "medic", "weight", "height", "bmi", "depress", "cesd"]


def find_file(prefix):
    for f in sorted(os.listdir(RAW)):
        if f.lower().startswith(prefix) and f.lower().endswith(".dta"):
            return os.path.join(RAW, f)
    return None


def read_dta(path):
    """Read WITHOUT converting codes to text, so special codes such as -8 or -9 stay visible."""
    reader = pd.read_stata(path, iterator=True, convert_categoricals=False)
    var_labels = reader.variable_labels()
    try:
        value_labels = reader.value_labels()
    except Exception:
        value_labels = {}
    df = reader.read()
    try:
        reader.close()
    except Exception:
        pass
    return df, var_labels, value_labels


print("=" * 70)
print("P1-01  Raw folder:", RAW)
for f in sorted(os.listdir(RAW)):
    print("  %-50s %8.1f MB" % (f, os.path.getsize(os.path.join(RAW, f)) / 1e6))

adult_path = find_file("adult_w5")
ind_path = find_file("indderived_w5")
if adult_path is None or ind_path is None:
    raise SystemExit("Could not find both files (names starting with adult_w5 and indderived_w5, ending .dta) "
                     "in data\\nids\\raw. Check the file names.")

print("=" * 70)
print("P1-02  Adult file     :", os.path.basename(adult_path))
adult, adult_labels, adult_vlabels = read_dta(adult_path)
print("  rows, columns:", adult.shape)
print("Individual-derived file:", os.path.basename(ind_path))
ind, ind_labels, ind_vlabels = read_dta(ind_path)
print("  rows, columns:", ind.shape)

print("=" * 70)
rows = []
for name, d in [("adult", adult), ("indderived", ind)]:
    if "pid" not in d.columns:
        print("WARNING: no column called 'pid' in the", name, "file. Columns starting with p:", [c for c in d.columns if c.lower().startswith("p")][:10])
        continue
    n_unique = d["pid"].nunique()
    rows.append({"file": name, "rows": len(d), "unique_pid": n_unique, "duplicate_pid_rows": len(d) - n_unique,
                 "missing_pid": int(d["pid"].isna().sum())})
pid_audit = pd.DataFrame(rows)
print("pid audit (one row should mean one person):")
print(pid_audit.to_string(index=False))
pid_audit.to_csv(os.path.join(OUT, "pid_audit.csv"), index=False)

print("=" * 70)
race_col = "w5_best_race"
if race_col in ind.columns:
    vl_name = None
    # value labels are stored per label-set; show codes with their text where we can find them
    codes = ind[race_col].value_counts(dropna=False).sort_index()
    label_map = {}
    for key, mapping in ind_vlabels.items():
        if "race" in key.lower():
            label_map = mapping
            break
    if not label_map:   # fall back to any label set that looks like population groups
        for key, mapping in ind_vlabels.items():
            if any(w in str(v).lower() for v in mapping.values() for w in ("african", "coloured", "white")):
                label_map = mapping
                break
    race = pd.DataFrame({"code": codes.index, "participants": codes.values})
    race["label"] = race["code"].map(lambda c: label_map.get(c, "") if pd.notna(c) else "missing")
    print("Population group %s (raw codes; negative codes are special values, not answers):" % race_col)
    print(race.to_string(index=False))
    race.to_csv(os.path.join(OUT, "race_counts.csv"), index=False)
else:
    print("Column %s not found in the individual-derived file. Race-like columns:" % race_col)
    print([c for c in ind.columns if "race" in c.lower()])

print("=" * 70)
print("Join test (adult + w5_best_race on pid, must be one-to-one):")
join_note = ""
try:
    keep = ["pid"] + ([race_col] if race_col in ind.columns else [])
    joined = adult.merge(ind[keep], on="pid", how="left", validate="one_to_one")
    n_no_race = int(joined[race_col].isna().sum()) if race_col in joined.columns else -1
    join_note = "OK one-to-one: %d rows; adults without a group code after the join: %d" % (len(joined), n_no_race)
except Exception as err:
    join_note = "NOT one-to-one or failed: %s" % err
print(" ", join_note)
with open(os.path.join(OUT, "join_test.txt"), "w") as f:
    f.write(join_note + "\n")

print("=" * 70)
print("Health-related variables in the ADULT file (found by keyword in the codebook labels):")
cands = []
for col in adult.columns:
    label = str(adult_labels.get(col, ""))
    text = (col + " " + label).lower()
    if any(k in text for k in KEYWORDS):
        s = adult[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        n_special = int((s < 0).sum()) if numeric else 0
        n_valid = int(((s >= 0).sum()) if numeric else s.notna().sum())
        cands.append({"column": col, "codebook_label": label, "dtype": str(s.dtype),
                      "valid_answers": n_valid, "special_codes_negative": n_special,
                      "n_unique_valid": int(s[s >= 0].nunique()) if numeric else int(s.nunique())})
cand = pd.DataFrame(cands)
if len(cand):
    cand = cand.sort_values("valid_answers", ascending=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    pd.set_option("display.max_rows", 60)
    print("Found %d candidate columns. The 25 with the most valid answers:" % len(cand))
    print(cand.head(25)[["column", "codebook_label", "valid_answers", "special_codes_negative", "n_unique_valid"]].to_string(index=False))
    cand.to_csv(os.path.join(OUT, "health_variable_candidates.csv"), index=False)
else:
    print("No keyword matches. Open the codebook / user guide and search for the health module.")

print("=" * 70)
print("Saved in", os.path.relpath(OUT, ROOT), ": pid_audit.csv, race_counts.csv, join_test.txt, health_variable_candidates.csv")
print("Finished.")
