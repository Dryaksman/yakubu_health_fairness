"""
p1_inspect_nhanes.py  --  Task 6, Phase 1 (P1-01, P1-02): first look at the RAW NHANES 2017-2018 files.

Put this file in  yakubu_health_fairness\\src\\  and run from the project folder:
    python src\\p1_inspect_nhanes.py

It reads DEMO_J.xpt, BMX_J.xpt and GHB_J.xpt, joins them on SEQN (never by row position), and prints row
counts BEFORE and AFTER every join. Nothing is cleaned or dropped. Outputs go to outputs\\nhanes\\phase1\\.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(ROOT, "data", "nhanes", "raw")
OUT = os.path.join(ROOT, "outputs", "nhanes", "phase1")
os.makedirs(OUT, exist_ok=True)

NEEDED = ["DEMO_J.xpt", "BMX_J.xpt", "GHB_J.xpt"]
missing_files = [f for f in NEEDED if not os.path.exists(os.path.join(RAW, f))]
if missing_files:
    raise SystemExit("Missing in data\\nhanes\\raw: %s" % ", ".join(missing_files))

print("=" * 70)
print("P1-01  Raw folder:", RAW)
tables = {}
for f in NEEDED:
    tables[f] = pd.read_sas(os.path.join(RAW, f), format="xport")
    t = tables[f]
    print("%-10s rows %6d | columns %3d | unique SEQN %6d" % (f, len(t), t.shape[1], t["SEQN"].nunique()))
    if t["SEQN"].nunique() != len(t):
        print("  WARNING: SEQN is not unique in", f, "- investigate before joining")

demo, bmx, ghb = tables["DEMO_J.xpt"], tables["BMX_J.xpt"], tables["GHB_J.xpt"]

print("=" * 70)
print("P1-02  Joins on SEQN (row counts before and after):")
audit = []
keep_demo = ["SEQN", "RIDRETH3", "RIDAGEYR", "RIAGENDR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"]
keep_demo = [c for c in keep_demo if c in demo.columns]
df = demo[keep_demo].copy()
audit.append({"step": "start: DEMO_J", "rows_before": len(df), "rows_after": len(df)})
n0 = len(df)
df = df.merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="left")
audit.append({"step": "left join BMX_J (BMXBMI)", "rows_before": n0, "rows_after": len(df)})
n1 = len(df)
df = df.merge(ghb[["SEQN", "LBXGH"]], on="SEQN", how="left")
audit.append({"step": "left join GHB_J (LBXGH)", "rows_before": n1, "rows_after": len(df)})
audit = pd.DataFrame(audit)
print(audit.to_string(index=False))
print("One row means: ONE SURVEY PARTICIPANT (SEQN). Rows should not change in a left join.")

print("=" * 70)
print("Shape:", df.shape)
pd.set_option("display.width", 200)
print(df.head())
print("\nMissing values per column:")
miss = df.isna().sum()
print(miss.to_string())
print("\nMissing HbA1c (LBXGH) is NOT 'negative': those participants have no lab result.")

print("=" * 70)
print("Race/Hispanic-origin codes (RIDRETH3): 3 = non-Hispanic White, 4 = non-Hispanic Black")
print(df["RIDRETH3"].value_counts(dropna=False).sort_index().to_string())

has_lab = df["LBXGH"].notna()
print("\nParticipants with a measured HbA1c:", int(has_lab.sum()), "of", len(df))
tmp = df[has_lab].copy()
tmp["elevated_preview"] = (tmp["LBXGH"] >= 6.5).astype(int)
print("PREVIEW of the target (HbA1c >= 6.5). Confirm the threshold from medical references before using it:")
print(tmp["elevated_preview"].value_counts().to_string())
sub = tmp[tmp["RIDRETH3"].isin([3, 4])]
tab = pd.crosstab(sub["RIDRETH3"].map({3: "NH White (3)", 4: "NH Black (4)"}), sub["elevated_preview"], margins=True)
tab.columns = ["below_cutoff", "elevated", "total"]
tab["pct_elevated"] = (100.0 * tab["elevated"] / tab["total"]).round(2)
print("\nMain two-group audit (counts):")
print(tab.to_string())

# ---- save evidence files
audit.to_csv(os.path.join(OUT, "merge_audit.csv"), index=False)
pd.DataFrame({"column": df.columns, "missing": miss.values, "dtype": [str(t) for t in df.dtypes]}).to_csv(
    os.path.join(OUT, "columns.csv"), index=False)
df["RIDRETH3"].value_counts(dropna=False).sort_index().rename("participants").reset_index().rename(
    columns={"index": "RIDRETH3"}).to_csv(os.path.join(OUT, "race_counts.csv"), index=False)
tab.reset_index().to_csv(os.path.join(OUT, "target_preview_by_group.csv"), index=False)
print("=" * 70)
print("Saved in", os.path.relpath(OUT, ROOT), ": merge_audit.csv, columns.csv, race_counts.csv, target_preview_by_group.csv")
print("DONE. Take your screenshots (raw folder, this terminal output).")
