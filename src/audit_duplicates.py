"""
Duplicate audit: Nat Commun 2024 circadian_time_series_data.xlsx vs MSB 2025 raw_lumicycle_data.xlsx.

    python run.py audit

For every NC trace (sheet 'raw', experiment ID from sheet 'samples') search the MSB 'raw_full' traces of the same
cell line and date for the index offset (0-10 h) that minimises max|NC - MSB| / max|MSB|.
status = same_recording if that relative difference < 1 % (found: 135/164 identical with offset exactly 6.0 h),
         new_date_or_line if no MSB trace has the same cell line and date.
Output: results/data_audit/natcomm2024_vs_msb2025.csv (used by load_natcomm2024(keep_audit=...)).
"""
import argparse
import re

import numpy as np
import pandas as pd
import yaml

from .vabam_bridge import SPINOFF_ROOT, assert_inside_spinoff


def _line(s):
    s = s.replace("Neuroblastoma_", "")
    return s.split("_")[1].upper() if re.match(r"\d{8}_", s) else s.split("_")[0].upper()


def _date(s):
    m = re.match(r"(\d{8})_", s)
    return m.group(1) if m else None


def audit(nc_path, msb_path, max_offset_samples=60):
    smp = pd.read_excel(nc_path, sheet_name="samples")
    smp.columns = ["short", "experiment"]
    raw = pd.read_excel(nc_path, sheet_name="raw")
    raw = raw[raw["Time"].notna()]
    msb = pd.read_excel(msb_path, sheet_name="raw_full")
    msb = msb[msb["Time"].notna()]
    mcols = list(msb.columns[1:])
    rows = []
    for _, r in smp.iterrows():
        if r["short"] not in raw.columns:
            continue
        a = raw[r["short"]].to_numpy(float)
        a = a[np.isfinite(a)]
        cl, d = _line(r["experiment"]), _date(r["experiment"])
        best = (np.inf, None, None, np.nan)
        for c in mcols:
            if _line(c) != cl or _date(c) != d:
                continue
            b = msb[c].to_numpy(float)
            b = b[np.isfinite(b)]
            for off in range(0, max_offset_samples + 1):
                n = min(len(a), len(b) - off)
                if n < 300:
                    continue
                diff = np.max(np.abs(a[:n] - b[off:off + n])) / np.max(np.abs(b[off:off + n]))
                if diff < best[0]:
                    best = (diff, c, off, np.corrcoef(a[:n], b[off:off + n])[0, 1])
        rows.append({"nc_short": r["short"], "nc_experiment": r["experiment"], "cell_line": cl, "date": d,
                     "best_msb": best[1], "offset_h": None if best[2] is None else best[2] / 6.0,
                     "max_rel_diff": best[0], "raw_r": best[3]})
    df = pd.DataFrame(rows)
    df["status"] = np.where(df["max_rel_diff"] < 0.01, "same_recording",
                            np.where(df["best_msb"].isna(), "new_date_or_line", "different"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    nc = SPINOFF_ROOT / cfg["data"]["sources"]["natcomm2024"]["path"]
    msb = SPINOFF_ROOT / cfg["data"]["sources"]["msb2025"]["path"]
    df = audit(nc, msb)
    out = assert_inside_spinoff(SPINOFF_ROOT / "results" / "data_audit")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "natcomm2024_vs_msb2025.csv", index=False)
    print(df["status"].value_counts().to_string())
    print(df.groupby(["cell_line", "status"]).size().unstack(fill_value=0).to_string())
    print("offsets (h) of identical recordings:", df.loc[df.status == "same_recording", "offset_h"].value_counts().to_dict())


if __name__ == "__main__":
    main()
