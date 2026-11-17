"""
Duplicate audit: Nat Commun 2024 circadian_time_series_data.xlsx vs MSB 2025 raw_lumicycle_data.xlsx.

    python run.py audit

For every NC trace (sheet 'raw', experiment ID from sheet 'samples') search the MSB 'raw_full' traces of the same
cell line and date for the index offset (0-10 h) that minimises max|NC - MSB| / max|MSB|.
status = same_recording if that relative difference < 1 % (found: 135/164 identical with offset exactly 6.0 h),
         new_date_or_line if no MSB trace has the same cell line and date.
Output: results/data_audit/natcomm2024_vs_msb2025.csv (used by load_natcomm2024(keep_audit=...)).

Catalogue of all raw sources (no model, CPU, a few seconds):

    python run.py audit --catalog          (or notebooks/09_data_catalog.ipynb)

Writes data/catalog/:
    sources.csv            one row per source folder: role, files, size, records, cell models
    records.csv            one row per reporter time series: model, reporter, date, sampling, duration, value range,
                           and duplicate_of when the same values already occur in an earlier source
    models.csv             one row per cell model: cancer type, lineage group, CCLE id, records and dates
    circascope_sheets.csv  one row per Circa-SCOPE sheet: condition, time points, cells
    drug_files.csv         one row per drug-response file: kind, experiment, cell model, drug, sheets
    cohorts.csv            one row per patient-cohort file: samples, genes or probes, survival fields
    SUMMARY.md
Memory: workbook sheets are sized from the zip directory without loading them; only the small reporter tables are
read in full; the large expression matrices are opened for their header line only. Peak use stays far below 1 GB.
"""
import argparse
import gzip
import re
import zipfile

import numpy as np
import pandas as pd
import yaml

from src.vabam_bridge import SPINOFF_ROOT, assert_inside_spinoff


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


# ----------------------------------------------------------------------------- catalogue of raw sources
RAW = "data/raw"
SOURCES = {   # source id -> (folder under data/raw, role, cancer type, species, lab)
    "msb2025": ("msb2025_breast_cancer", "core: reporter recordings", "breast", "human", "Granada"),
    "natcomm2024": ("natcommun2024_tod", "core: reporter recordings + drug response", "breast/neuroblastoma", "human", "Granada"),
    "nb2026": ("cellreports2026_neuroblastoma", "core: reporter recordings + drug response", "neuroblastoma", "human", "Granada"),
    "kidney2025": ("natcommun2025_kidney", "extension: other laboratory", "kidney", "human", "Mello et al."),
    "4t1_2021": ("elife2021_4t1", "extension: other laboratory", "breast", "mouse", "Ogino et al."),
    "hes1_2021": ("hes1_sabherwal2021", "extension: single-cell HES1", "breast", "human", "Sabherwal et al."),
    "circascope2021": ("circascope2021", "method check: perturbation at 108 h", "fibroblast", "mouse", "Asher"),
    "msb2025_cisplatin": ("external/msb2025_cisplatin", "drug response", "breast", "human", "Granada"),
    "cbioportal": ("external/cbioportal", "omics and patient cohorts", "", "human", ""),
    "cohorts": ("external/cohorts", "patient cohorts", "", "human", ""),
}
LINEAGE = {"SY5Y": "SKNSH", "SKNSH": "SKNSH", "SKNBE": "SKNBE", "SKNBE2": "SKNBE"}   # same patient / parental line
CCLE_ALIAS = {"SY5Y": "SHSY5Y", "786-O": "786O", "U2OS": "U2OS"}
NON_CANCER = {"MCF10A"}


def _col(letters):
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def xlsx_sheets(path, parts=False):
    """[(sheet name, n_rows, n_cols)] from the workbook's zip directory; the sheets themselves are not loaded.
    parts=True appends the path of each sheet inside the zip."""
    z = zipfile.ZipFile(path)
    wb = z.read("xl/workbook.xml").decode("utf8", "replace")
    rel = z.read("xl/_rels/workbook.xml.rels").decode("utf8", "replace")
    target = {}
    for m in re.finditer(r"<Relationship\b[^>]*>", rel):
        i, t = re.search(r'\bId="([^"]+)"', m.group(0)), re.search(r'\bTarget="([^"]+)"', m.group(0))
        if i and t:
            target[i.group(1)] = t.group(1)
    out = []
    for m in re.finditer(r"<sheet\b[^>]*>", wb):
        name = re.search(r'\bname="([^"]*)"', m.group(0)).group(1)
        part = target[re.search(r'\br:id="([^"]*)"', m.group(0)).group(1)].lstrip("/")
        part = part if part.startswith("xl/") else "xl/" + part
        d = re.search(r'<dimension ref="[A-Z]+\d+:([A-Z]+)(\d+)"', z.open(part).read(4096).decode("utf8", "replace"))
        out.append((name, int(d.group(2)) if d else 0, _col(d.group(1)) if d else 0) + ((part,) if parts else ()))
    return out


def _fingerprint(x, k=6):
    x = np.round(np.asarray(x, float), 3)
    return {tuple(x[i:i + k]) for i in range(0, len(x) - k, k // 2) if np.isfinite(x[i:i + k]).all()}


def _series_row(source, rid, model, reporter, date, rep, t_h, x, condition=""):
    ok = np.isfinite(x)
    t, v = np.asarray(t_h, float)[ok], np.asarray(x, float)[ok]
    return dict(source=source, record_id=rid, cell_model=model, reporter=reporter, date=date, replicate=rep,
                condition=condition, n_points=int(ok.sum()), dt_h=float(np.median(np.diff(t))) if len(t) > 1 else np.nan,
                t_start_h=float(t[0]) if len(t) else np.nan, duration_h=float(t[-1] - t[0]) if len(t) else np.nan,
                min_value=float(v.min()) if len(v) else np.nan, frac_nonpositive=float((v <= 0).mean()) if len(v) else np.nan,
                raw_like=bool(len(v) and v.min() > 0), ge_72h=bool(len(t) and t[-1] - t[0] >= 72.0))


def _lumi_table(source, time, df):
    from src.data.loader import parse_trace_name

    rows, fps = [], {}
    for c in df.columns:
        try:
            m = parse_trace_name(re.sub(r"_BMAL_(?=\d)", "_BMAL1_", str(c).replace("Neuroblastoma_", "")))
        except ValueError:
            continue
        model = m["cell_line"] if m["genotype"] == "WT" else m["label"]
        rows.append(_series_row(source, "%s:%s" % (source, c), model, m["reporter"], m["date"], m["replicate"], time,
                                df[c].to_numpy(float), m["reset"]))
        fps[rows[-1]["record_id"]] = _fingerprint(df[c].to_numpy(float))
    return rows, fps


def _block_table(source, path, sheet, header_rows, to_hours, reporter):
    """Wide figure sheets: a time column ('Days' / 'time (day)') followed by replicate columns, groups named above."""
    d = pd.read_excel(path, sheet_name=sheet, header=None)
    hdr, body = d.iloc[:header_rows], d.iloc[header_rows:].apply(pd.to_numeric, errors="coerce")
    is_time = hdr.apply(lambda col: col.astype(str).str.contains(r"^\s*(?:Days|time)", case=False).any())
    group = hdr.iloc[-2].ffill()
    rows, tcol = [], None
    for j in d.columns:
        if is_time[j]:
            tcol = j
            continue
        x = body[j].to_numpy(float)
        if tcol is None or np.isfinite(x).sum() < 50:
            continue
        g = str(group[j]).strip()
        k = re.match(r"(786-O|A498|RCC4)\s*(.*)", g)
        model, cond = (k.group(1), k.group(2).strip()) if k else (g, "")
        if "VHL" in cond:                                # VHL re-expression is a genotype variant, not a condition
            model, cond = model + "_VHL", ""
        rid = "%s:%s:%s:%s" % (source, sheet, g, str(hdr.iloc[-1][j]).strip())
        rows.append(_series_row(source, rid, model if source == "kidney2025" else "4T1", reporter, None,
                                str(hdr.iloc[-1][j]).strip(), body[tcol].to_numpy(float) * to_hours, x,
                                cond if source == "kidney2025" else g))
    return rows


def _records(root):
    raw = root / RAW
    rows, fps = [], {}

    def add(source, time, df):
        r, f = _lumi_table(source, time, df)
        for row in r:                                   # duplicate = same values as a record of an earlier source
            mine, row["duplicate_of"] = f[row["record_id"]], ""
            for other, fp in fps.items():
                if mine and len(mine & fp) > 0.5 * len(mine):
                    row["duplicate_of"] = other
                    break
        rows.extend(r)
        fps.update(f)

    p = raw / "msb2025_breast_cancer/raw_lumicycle_data.xlsx"
    if p.exists():
        d = pd.read_excel(p, sheet_name="raw_full")
        d = d[d["Time"].notna()]
        add("msb2025", d["Time"].to_numpy(float), d.drop(columns="Time"))
    p = raw / "natcommun2024_tod/circadian_time_series_data.xlsx"
    if p.exists():
        d = pd.read_excel(p, sheet_name="raw")
        d = d[d["Time"].notna()]
        d = d[[c for c in d.columns if not str(c).endswith(".1")]]
        add("natcomm2024", d["Time"].to_numpy(float), d.drop(columns="Time"))
    for f in ("Bmal1_equallength.csv", "Per2_equallength.csv"):
        p = raw / "cellreports2026_neuroblastoma" / f
        if p.exists():
            d = pd.read_csv(p)
            add("nb2026", d["Time"].to_numpy(float), d.drop(columns="Time"))
    p = raw / "natcommun2025_kidney/41467_2025_60904_MOESM4_ESM.xlsx"
    if p.exists():
        for sheet in ("Figure 1f", "Figure 1g", "Figure S2B"):
            rows.extend(_block_table("kidney2025", p, sheet, 6, 24.0, "Per2"))
    p = raw / "elife2021_4t1/elife-66155-fig1-data1-v2.xlsx"
    if p.exists():
        rows.extend(_block_table("4t1_2021", p, "F", 3, 24.0, "E-box"))
    df = pd.DataFrame(rows)
    df["duplicate_of"] = df["duplicate_of"].fillna("") if "duplicate_of" in df else ""
    df["usable"] = (df["duplicate_of"] == "") & df["ge_72h"]    # new and at least three days; raw_like is reported separately
    return df


def _models(rec, root):
    ccle = []
    p = root / RAW / "external/cbioportal/ccle_broad_2019/data_mrna_seq_rpkm.txt"
    if p.exists():
        ccle = open(p, encoding="utf-8").readline().rstrip("\n").split("\t")[1:]
    by_name = {}
    for c in ccle:
        by_name.setdefault(c.split("_")[0], []).append(c)
    dep, dep_expr = {}, set()                        # DepMap release: model ids and which of them have expression
    d = root / RAW / "external/depmap"
    if (d / "Model.csv").exists():
        t = pd.read_csv(d / "Model.csv", usecols=["ModelID", "StrippedCellLineName"])
        dep = {re.sub(r"[^A-Z0-9]", "", str(n).upper()): i for i, n in zip(t.ModelID, t.StrippedCellLineName)}
        for f in sorted(d.glob("OmicsExpression*.csv")):
            with open(f, encoding="utf-8") as h:
                col = h.readline().split(",", 8).index("ModelID")
                dep_expr |= {line.split(",", col + 1)[col] for line in h}
    cancer = {k: v[2] for k, v in SOURCES.items()}
    rows = []
    new = rec[rec.duplicate_of == ""]
    for (src, model), g in rec.groupby(["source", "cell_model"]):
        base = model.split("_")[0]
        key = re.sub(r"[^A-Z0-9]", "", CCLE_ALIAS.get(base, base).upper())
        n = new[(new.source == src) & (new.cell_model == model)]
        ctype = cancer[src]
        if src == "natcomm2024":
            ctype = "neuroblastoma" if base in ("GIMEN", "SY5Y") else "osteosarcoma" if base == "U2OS" else "breast"
        if base == "U2OS":
            ctype = "osteosarcoma"
        rows.append(dict(source=src, cell_model=model, cancer_type="non-tumorigenic breast" if base in NON_CANCER else ctype,
                         species=SOURCES[src][3], lineage_group=LINEAGE.get(base, base), genotype_variant=model != base,
                         ccle_id=";".join(by_name.get(key, [])), depmap_id=dep.get(key, ""),
                         depmap_expression=dep.get(key, "") in dep_expr, reporters=",".join(sorted(g.reporter.unique())),
                         n_records=len(g), n_new_records=len(n), n_usable=int(g.usable.sum()), raw_like=bool(g.raw_like.all()),
                         n_dates=int(g.date.nunique()),
                         median_duration_h=float(g.duration_h.median()), median_dt_h=float(g.dt_h.median())))
    return pd.DataFrame(rows)


def _circascope(root):
    rows = []
    for p in sorted((root / RAW / "circascope2021").glob("RawData_*.xlsx")):
        for name, nr, nc in xlsx_sheets(p):
            if nr > 1:
                agent, _, dose = name.strip().partition(" ")
                rows.append(dict(file=p.name, sheet=name, agent=agent, dose=dose.strip(), n_timepoints=nr, n_cells=nc,
                                 size_mb=round(p.stat().st_size / 1e6, 1)))
    return pd.DataFrame(rows)


def _drug_files(root, models):
    names = sorted({m.split("_")[0] for m in models} | {"GIMEN-1"}, key=len, reverse=True)
    rows = []
    for src, sub in (("natcomm2024", "natcommun2024_tod"), ("nb2026", "cellreports2026_neuroblastoma/growth_tod"),
                     ("msb2025_cisplatin", "external/msb2025_cisplatin")):
        base = root / RAW / sub
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in (".xlsx", ".csv") or "_zip" in p.parts or p.parent == base:
                continue
            stem = re.sub(r"^\d{8}_(?=\d{4,})", "", p.stem)
            tok = re.split(r"[_ ]", stem)
            model = next((n for n in names if n.upper().replace("-", "") in [t.upper().replace("-", "") for t in tok]), "")
            sh = xlsx_sheets(p) if p.suffix.lower() == ".xlsx" else []
            rows.append(dict(source=src, folder=str(p.parent.relative_to(base)).replace("\\", "/"), file=p.name,
                             experiment=next((t for t in tok if re.fullmatch(r"\d{4,9}", t)), ""), cell_model=model,
                             drug=next((t for t in tok if t in ("Lorlatinib", "Cisplatin", "Doxorubicin")), ""),
                             n_sheets=len(sh), first_sheet_rows=sh[0][1] if sh else np.nan,
                             first_sheet_cols=sh[0][2] if sh else np.nan, size_kb=round(p.stat().st_size / 1e3, 1)))
    return pd.DataFrame(rows)


def _cohorts(root):
    surv = re.compile(r"surviv|\bos\b|efs|event|progression|death|vital|status", re.I)
    rows = []
    base = root / RAW / "external"
    for p in sorted((base / "cohorts").rglob("*.gz")):
        row = dict(cohort=p.parent.name, file=p.name, size_mb=round(p.stat().st_size / 1e6, 1), n_samples=np.nan,
                   n_features=np.nan, survival_fields="")
        with gzip.open(p, "rt", errors="replace") as h:
            if "series_matrix" in p.name:
                fields, n_tab, in_tab = [], 0, False
                for line in h:
                    if line.startswith("!Sample_geo_accession"):
                        row["n_samples"] = len(line.rstrip("\n").split("\t")) - 1
                    elif line.startswith("!Sample_characteristics_ch1"):
                        v = [x.strip('"').split(":")[0].strip() for x in line.rstrip("\n").split("\t")[1:] if ":" in x]
                        if v:
                            fields.append(max(set(v), key=v.count))
                    elif line.startswith("!series_matrix_table_begin"):
                        in_tab = True
                    elif in_tab and not line.startswith("!"):
                        n_tab += 1
                row["n_features"] = max(n_tab - 1, 0)
                row["survival_fields"] = "; ".join(f for f in fields if surv.search(f))
            else:                                   # expression matrix: header only (files can be several hundred MB)
                first = h.readline()
                row["n_samples"] = len(re.split(r"[\t,]", first.rstrip("\n"))) - 1
        rows.append(row)
    for d in sorted((base / "cbioportal").glob("*")):
        for p in sorted(d.glob("data_*.txt")):
            with open(p, encoding="utf-8", errors="replace") as h:
                first = h.readline().rstrip("\n").split("\t")
                n_lines = sum(1 for _ in h) if p.stat().st_size < 2e8 else np.nan
            clin = "clinical" in p.name
            rows.append(dict(cohort=d.name, file=p.name, size_mb=round(p.stat().st_size / 1e6, 1),
                             n_samples=n_lines - 4 if clin else len(first) - 2, n_features=np.nan if clin else n_lines,
                             survival_fields="OS/DFS columns in file" if clin else ""))
    return pd.DataFrame(rows)


def catalog(root=None, save=True):
    """Build every catalogue table. Returns a dict of DataFrames; with save=True also writes data/catalog/."""
    root = SPINOFF_ROOT if root is None else root
    rec = _records(root)
    mod = _models(rec, root)
    cs = _circascope(root)
    drug = _drug_files(root, mod.cell_model.tolist())
    coh = _cohorts(root)
    src = []
    for sid, (sub, role, cancer, species, lab) in SOURCES.items():
        d = root / RAW / sub
        files = [p for p in d.rglob("*") if p.is_file() and "_zip" not in p.parts] if d.exists() else []
        r = rec[rec.source == sid]
        src.append(dict(source=sid, folder=RAW + "/" + sub, role=role, cancer_type=cancer, species=species, lab=lab,
                        present=d.exists() and bool(files), n_files=len(files),
                        size_mb=round(sum(p.stat().st_size for p in files) / 1e6, 1), n_records=len(r),
                        n_new_records=int((r.duplicate_of == "").sum()), n_cell_models=int(r.cell_model.nunique())))
    out = dict(sources=pd.DataFrame(src), records=rec, models=mod, circascope_sheets=cs, drug_files=drug, cohorts=coh)
    if save:
        dst = assert_inside_spinoff(root / "data" / "catalog")
        dst.mkdir(parents=True, exist_ok=True)
        for k, v in out.items():
            v.to_csv(dst / (k + ".csv"), index=False)
        human = mod[~mod.genotype_variant & mod.source.isin(["msb2025", "natcomm2024", "nb2026"])]
        human = human.groupby("cell_model").agg(lineage_group=("lineage_group", "first"), ccle_id=("ccle_id", "max"),
                                                depmap_expression=("depmap_expression", "max"),
                                                n_usable=("n_usable", "sum")).reset_index()
        human = human[human.n_usable > 0]
        lines = ["# Data catalogue", "", "Generated by `python run.py audit --catalog`. Tables are in this folder.", "",
                 "- reporter records: %d (%d not duplicated in an earlier source)" % (len(rec), int((rec.duplicate_of == "").sum())),
                 "- core cell models (MSB 2025, Nat Commun 2024, Cell Reports 2026) with usable recordings: %d (%d lineage groups); with a CCLE id: %d"
                 % (human.cell_model.nunique(), human.lineage_group.nunique(), int((human.ccle_id != "").sum())),
                 "- of these, with expression in the DepMap release in data/raw/external/depmap: %d (missing: %s)"
                 % (int(human.depmap_expression.sum()), ", ".join(human.cell_model[~human.depmap_expression]) or "none"),
                 "- Circa-SCOPE: %d sheets, %d cells, agents: %s" % (len(cs), int(cs.n_cells.sum()) if len(cs) else 0,
                                                                   ", ".join(sorted(cs.agent.unique())) if len(cs) else ""),
                 "- drug-response files: %d" % len(drug), "- cohort files: %d" % len(coh), ""]
        (dst / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--catalog", action="store_true", help="write the catalogue of all raw sources to data/catalog/")
    args = ap.parse_args()
    if args.catalog:
        out = catalog()
        print(out["sources"].to_string(index=False))
        return
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
