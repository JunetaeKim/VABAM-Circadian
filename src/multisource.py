"""
Configuration B: multi-source training pool, validation set and external test sets.

    python run.py multi                    # builds data/processed/<multi.tag>/ from configs/circadian_vabam.yml

Sources (all put on a 10-min grid, hours since start of recording):
  msb2025          LumiCycle population bioluminescence, Bmal1/Per2-Luc, 180 traces, native 10 min
  circascope2021   single-cell Rev-VNP fluorescence, NIH-3T3 (Manella et al. 2021, Zenodo 5139746),
                   native 1 h -> PCHIP-interpolated to 10 min. Each cell gives two segments that never span the
                   drug addition at 108 h: 'pre' 0-108 h, 'post' 120 h-end (12 h after addition skipped).
  natcomm2024      LumiCycle traces NOT duplicated in msb2025 (GIMEN, SH-SY5Y; src/audit_duplicates.py)
  hes1_2021        single-cell mVenus-Hes1 (MCF7; Sabherwal et al. 2021), tracks >= window length only,
                   native 15/30 min -> PCHIP to 10 min

Every record is detrended and normalised exactly like preprocess.py (sinc trend, T_c 48 h; relative = detrended /
trend). Records are split train/val by GROUP (MSB: trace; Circa-SCOPE: cell, so both segments of a cell stay
together), stratified by label/condition. External sets are never used for training, scaling or selection.

Scaling to [0, 1] is per scale group: 'pop' (population bioluminescence) and 'sc' (single-cell fluorescence);
q = scale_quantile of |normalised value| over the TRAINING records of that group. External sets use the q of
the group named in the config.

Outputs in data/processed/<tag>/:
  traces.npz         Xn_full (n_records x max_len, NaN padded), lengths, plus per-record arrays
  records.csv        one row per record (source, split, group_id, label, condition, stride, ...)
  pool_train.npy     (record_index, start) of every training window at the source's stride
  val_windows.npz    X01 / Xn of every validation window, plus record index and start
  scaling.json       q per group, settings
  counts.csv         windows / records / groups per split and source
Compatibility files for exp01 and select (MSB windows only, 3 h stride, same split) are written as
data/processed/windows_<tag>.npz, windows_<tag>_meta.csv, scaling_<tag>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .loader import load_msb2025_raw, load_natcomm2024
from .preprocess import normalise_trace
from .vabam_bridge import SPINOFF_ROOT, assert_inside_spinoff

DT_H = 10.0 / 60.0

import sys as _sys
import time as _time

_T0 = [None]


def log(msg):
    """Progress line with elapsed time, flushed immediately (visible in Anaconda Prompt)."""
    if _T0[0] is None:
        _T0[0] = _time.time()
    el = _time.time() - _T0[0]
    print("[%02d:%02d] %s" % (el // 60, el % 60, msg), flush=True)
    _sys.stdout.flush()


# ------------------------------------------------------------------ helpers
def _pchip_to_grid(t, y, t0=None, t1=None):
    from scipy.interpolate import PchipInterpolator

    t0 = t[0] if t0 is None else t0
    t1 = t[-1] if t1 is None else t1
    grid = np.round(np.arange(t0, t1 + 1e-9, DT_H), 6)
    return grid, PchipInterpolator(t, y)(grid)


def _record(rid, source, group_id, t, x, **meta):
    return dict(record_id=rid, source=source, group_id=group_id, t=np.asarray(t, float),
                x=np.asarray(x, float), **meta)


# ------------------------------------------------------------------ source readers -> list of records
def read_msb(path, stride):
    ts = load_msb2025_raw(path)
    out = []
    for tid in ts.meta.index:
        x = ts.values[tid].to_numpy()
        ok = np.isfinite(x)
        m = ts.meta.loc[tid]
        out.append(_record(tid, "msb2025", tid, ts.time[ok], x[ok], label=m["label"], cell_line=m["cell_line"],
                           genotype=m["genotype"], reporter=m["reporter"], condition=m["reset"], date=m["date"],
                           modality="population_bioluminescence", native_dt_h=DT_H, stride=stride))
    return out


def read_natcomm(path, audit, stride):
    ts = load_natcomm2024(path, keep_audit=audit)
    out = []
    for tid in ts.meta.index:
        x = ts.values[tid].to_numpy()
        ok = np.isfinite(x)
        m = ts.meta.loc[tid]
        out.append(_record(tid, "natcomm2024", tid, ts.time[ok], x[ok], label=m["label"], cell_line=m["cell_line"],
                           genotype=m["genotype"], reporter=m["reporter"], condition=m["reset"], date=m["date"],
                           modality="population_bioluminescence", native_dt_h=DT_H, stride=stride))
    return out


def read_circascope(paths, stride, pre_end_h, post_start_h, native_dt_h=1.0):
    out = []
    for p in paths:
        p = Path(p)
        log("  reading %s ..." % p.name)
        xls = pd.ExcelFile(p, engine="openpyxl")
        for sh in xls.sheet_names:
            d = pd.read_excel(xls, sheet_name=sh, header=None)
            if d.size == 0:
                continue
            log("    sheet %-18s %d time points x %d cells" % (repr(sh), d.shape[0], d.shape[1]))
            V = d.apply(pd.to_numeric, errors="coerce").to_numpy(float)
            t = np.arange(V.shape[0]) * native_dt_h
            cond = "unknown" if sh.lower().startswith("sheet") else sh
            for j in range(V.shape[1]):
                y = V[:, j]
                if not np.isfinite(y).all() or (y <= 0).any():
                    continue  # complete, positive tracks only (all tracks in RawData_1/2 qualify)
                gid = "%s:%s:c%04d" % (p.stem, sh, j)
                for seg, (a, b) in (("pre", (0.0, pre_end_h)), ("post", (post_start_h, t[-1]))):
                    m = (t >= a - 1e-9) & (t <= b + 1e-9)
                    g, x = _pchip_to_grid(t[m], y[m])
                    out.append(_record("%s:%s" % (gid, seg), "circascope2021", gid, g, x, label="NIH3T3_RevVNP",
                                       cell_line="NIH3T3", genotype="WT", reporter="Rev-VNP", condition=cond,
                                       date=None, file=p.name, segment=seg, modality="single_cell_fluorescence",
                                       native_dt_h=native_dt_h, stride=stride))
    return out


def read_hes1(path, sheets, min_hours, stride):
    out = []
    xls = pd.ExcelFile(path, engine="openpyxl")
    for sh, header in sheets:
        d = pd.read_excel(xls, sheet_name=sh, header=header)
        t = pd.to_numeric(d.iloc[:, 0], errors="coerce")
        d, t = d[t.notna()], t[t.notna()].to_numpy(float)
        for c in d.columns[1:]:
            y = pd.to_numeric(d[c], errors="coerce").to_numpy(float)
            f = np.where(np.isfinite(y))[0]
            if len(f) == 0 or t[f[-1]] - t[f[0]] < min_hours:
                continue
            yy, tt = y[f[0]:f[-1] + 1], t[f[0]:f[-1] + 1]
            if not np.isfinite(yy).all() or (yy <= 0).any():
                continue
            g, x = _pchip_to_grid(tt, yy)
            rid = "hes1:%s:%s" % (sh, c)
            out.append(_record(rid, "hes1_2021", rid, g, x, label="MCF7_HES1", cell_line="MCF7", genotype="WT",
                               reporter="mVenus-Hes1", condition=sh, date=None, modality="single_cell_fluorescence",
                               native_dt_h=float(np.median(np.diff(tt))), stride=stride))
    return out


# ------------------------------------------------------------------ build
def _split_groups(recs, val_frac, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame([{k: r[k] for k in ("group_id", "source", "label", "condition")} for r in recs]).drop_duplicates(
        "group_id")
    df["stratum"] = df["source"] + "|" + df["label"] + "|" + df["condition"].astype(str)
    val = set()
    for _, g in df.groupby("stratum"):
        ids = g["group_id"].tolist()
        n = int(round(len(ids) * val_frac))
        if len(ids) >= 3 and n == 0:
            n = 1
        val.update(rng.choice(ids, size=n, replace=False).tolist() if n else [])
    return val


def build(cfg):
    mc = cfg["multi"]
    L = int(mc["window"])
    root = SPINOFF_ROOT
    tr = mc["train_sources"]
    log("configuration B preprocessing started (tag %s). Typical run time: 0.5-2 min." % mc["tag"])
    log("[1/6] reading training sources")
    recs = []
    log("  reading MSB 2025 raw_lumicycle_data.xlsx ...")
    recs += read_msb(root / tr["msb2025"]["path"], tr["msb2025"]["stride_samples"])
    log("  MSB: %d traces" % len(recs))
    cs = tr["circascope2021"]
    recs += read_circascope([root / p for p in cs["files"]], cs["stride_samples"], cs["pre_end_h"], cs["post_start_h"])
    log("  Circa-SCOPE (train/val files): %d segments" % (len(recs) - sum(r["source"] == "msb2025" for r in recs)))
    for r in recs:
        r["scale_group"] = tr[r["source"]]["scale_group"]
    val_groups = _split_groups(recs, mc["val_frac"], mc["seed"])
    for r in recs:
        r["split"] = "val" if r["group_id"] in val_groups else "train"

    log("[2/6] reading external sets")
    ext = []
    for name, sp in mc["external"].items():
        k = sp["kind"]
        log("  %s (%s)" % (name, k))
        try:
            if k == "circascope":
                missing = [p for p in sp["files"] if not (root / p).exists()]
                if missing:
                    raise FileNotFoundError(", ".join(missing))
                rr = read_circascope([root / p for p in sp["files"]], sp["stride_samples"], cs["pre_end_h"],
                                     cs["post_start_h"])
            elif k == "natcomm2024":
                rr = read_natcomm(root / sp["path"], root / sp["keep_audit"], sp["stride_samples"])
            elif k == "hes1":
                rr = read_hes1(root / sp["path"], [tuple(s) for s in sp["sheets"]], sp["min_hours"],
                               sp["stride_samples"])
            else:
                raise ValueError(k)
        except FileNotFoundError as e:
            print("external set %s skipped (file missing: %s)" % (name, e))
            continue
        for r in rr:
            r["split"], r["scale_group"] = name, sp["scale_group"]
        ext += rr
        log("    -> %d records" % len(rr))
    allrecs = [r for r in recs + ext if len(r["x"]) >= L]
    dropped = len(recs + ext) - len(allrecs)

    # QC (fixed before looking at any result): the trend under 'relative' normalisation must stay well above 0,
    # otherwise detrended/trend explodes. Exclude records whose trend minimum < min_trend_ratio x trend median.
    # The trend is computed once per record and reused for normalisation.
    from .preprocess import sinc_trend

    pre = {"detrend": cfg["preprocess"]["detrend"], "normalize": cfg["preprocess"]["normalize"]}
    thr = float(mc.get("min_trend_ratio", 0.1))
    kept, excluded = [], []
    n_all = len(allrecs)
    log("[3/6] detrending + normalising %d records (slowest step)" % n_all)
    for k, r in enumerate(allrecs, 1):
        tr_ = sinc_trend(r["x"], pre["detrend"]["T_c_h"], DT_H)
        ratio = tr_.min() / np.median(tr_)
        if not np.isfinite(ratio) or ratio < thr:
            excluded.append({"record_id": r["record_id"], "source": r["source"], "split": r["split"],
                             "min_over_median_trend": float(ratio)})
        else:
            if pre["normalize"]["method"] == "relative":
                r["xn"] = ((r["x"] - tr_) / tr_).astype(np.float32)
            else:
                r["xn"] = normalise_trace(r["x"], DT_H, pre)["norm"].astype(np.float32)
            kept.append(r)
        if k % 2000 == 0 or k == n_all:
            log("  %d / %d records (%.0f%%), excluded by QC so far: %d" % (k, n_all, 100.0 * k / n_all,
                                                                          len(excluded)))
    allrecs = kept

    log("[4/6] scale per group and window index")
    # per-group scale from training records
    q = {}
    for g in sorted({r["scale_group"] for r in allrecs}):
        vals = np.concatenate([np.abs(r["xn"]) for r in allrecs if r["scale_group"] == g and r["split"] == "train"])
        q[g] = float(np.quantile(vals, mc["scale_quantile"]))

    # arrays
    n, maxlen = len(allrecs), max(len(r["xn"]) for r in allrecs)
    Xf = np.full((n, maxlen), np.nan, np.float32)
    for i, r in enumerate(allrecs):
        Xf[i, :len(r["xn"])] = r["xn"]
    lens = np.array([len(r["xn"]) for r in allrecs])
    meta_cols = ["record_id", "source", "group_id", "split", "scale_group", "label", "cell_line", "genotype",
                 "reporter", "condition", "date", "modality", "native_dt_h", "stride", "file", "segment"]
    rec = pd.DataFrame([{c: r.get(c) for c in meta_cols} for r in allrecs])
    rec["length"] = lens
    rec["t0_h"] = [r["t"][0] for r in allrecs]
    rec["n_windows"] = [(l - L) // s + 1 for l, s in zip(lens, rec["stride"])]
    rec["q"] = rec["scale_group"].map(q)

    def windows_of(idx):
        rows = []
        for i in idx:
            for s in range(0, lens[i] - L + 1, int(rec.at[i, "stride"])):
                rows.append((i, s))
        return np.array(rows, dtype=np.int32).reshape(-1, 2)

    pool = windows_of(np.where(rec["split"] == "train")[0])
    vidx = windows_of(np.where(rec["split"] == "val")[0])
    Xn_val = np.stack([Xf[i, s:s + L] for i, s in vidx]).astype(np.float32)
    qv = rec["q"].to_numpy()[vidx[:, 0]][:, None]
    X01_val = np.clip(0.5 + Xn_val / (2 * qv), 0, 1).astype(np.float32)

    log("[5/6] writing data/processed/%s/ (traces.npz is ~40 MB, compression takes a few seconds)" % mc["tag"])
    out = assert_inside_spinoff(SPINOFF_ROOT / cfg["paths"]["processed"] / mc["tag"])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(excluded, columns=["record_id", "source", "split", "min_over_median_trend"]).to_csv(
        out / "qc_excluded.csv", index=False)
    np.savez_compressed(out / "traces.npz", Xn_full=Xf, lengths=lens)
    rec.to_csv(out / "records.csv", index=False)
    np.save(out / "pool_train.npy", pool)
    np.savez_compressed(out / "val_windows.npz", X01=X01_val, index=vidx)

    cnt = (rec.groupby(["split", "source"])
           .agg(records=("record_id", "size"), groups=("group_id", "nunique"), windows=("n_windows", "sum"),
                dates=("date", "nunique"))
           .reset_index())
    cnt.to_csv(out / "counts.csv", index=False)
    clipped = {g: float(np.mean(np.abs(np.concatenate([r["xn"] for r in allrecs if r["scale_group"] == g])) > q[g]))
               for g in q}
    info = {"tag": mc["tag"], "window": L, "dt_h": DT_H, "q": q, "scale_quantile": mc["scale_quantile"],
            "fraction_beyond_q": clipped, "val_frac": mc["val_frac"], "seed": mc["seed"],
            "n_train_pool_windows": int(len(pool)), "n_val_windows": int(len(vidx)),
            "records_dropped_shorter_than_window": int(dropped),
            "records_excluded_qc_trend": len(excluded), "min_trend_ratio": thr,
            "circascope_segments": {"pre_end_h": cs["pre_end_h"], "post_start_h": cs["post_start_h"]},
            "normalize": pre["normalize"], "detrend": pre["detrend"]}
    with open(out / "scaling.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # compatibility files for exp01 / select: MSB windows only, 3 h stride, same split, pop scale
    log("[6/6] compatibility files for exp01/select")
    write_compat(cfg, mc["tag"], allrecs, rec, q["pop"], L)
    log("done.")
    return rec, cnt, info


def write_compat(cfg, tag, allrecs, rec, q_pop, L, stride=18):
    rows, X = [], []
    for i, r in enumerate(allrecs):
        if r["source"] != "msb2025":
            continue
        for w, s in enumerate(range(0, len(r["xn"]) - L + 1, stride)):
            X.append(r["xn"][s:s + L])
            rows.append({"trace_id": r["record_id"], "t_start_h": float(r["t"][s]), "win_idx": w,
                         "source": "msb2025", "date": r["date"], "cell_line": r["cell_line"],
                         "genotype": r["genotype"], "reporter": r["reporter"], "reset": r["condition"],
                         "replicate": None, "label": r["label"], "split": r["split"]})
    Xn = np.stack(X).astype(np.float32)
    X01 = np.clip(0.5 + Xn / (2 * q_pop), 0, 1).astype(np.float32)
    meta = pd.DataFrame(rows)
    proc = SPINOFF_ROOT / cfg["paths"]["processed"]
    np.savez_compressed(proc / ("windows_%s.npz" % tag), X01=X01, Xn=Xn, split=meta["split"].to_numpy().astype(str))
    meta.to_csv(proc / ("windows_%s_meta.csv" % tag), index=False)
    with open(proc / ("scaling_%s.json" % tag), "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "q_abs": q_pop, "dt_h": DT_H, "length_samples": L, "stride_samples": stride,
                   "note": "compatibility file (MSB windows only) for exp01/select; training uses %s/" % tag}, f,
                  indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    rec, cnt, info = build(cfg)
    pd.set_option("display.width", 200)
    print(cnt.to_string(index=False))
    print(json.dumps({k: info[k] for k in ("q", "fraction_beyond_q", "n_train_pool_windows", "n_val_windows",
                                           "records_dropped_shorter_than_window", "records_excluded_qc_trend")},
                     indent=2))


if __name__ == "__main__":
    main()
