"""
Preprocessing: detrend -> normalise -> cut into windows of 2-3 circadian cycles -> scale to [0, 1].

1. Detrending: windowed-sinc low-pass (Blackman window) with cut-off period T_c (default 48 h) gives the
   trend; detrended = raw - trend. Same approach and settings as the authors' pyBOAT pipeline
   (sinc_smooth, T_c = 48 h; Moenke et al. 2020), implemented here in numpy. Filter length
   M = min(len - 1, 2000), even; mirror padding at both ends.
2. Normalisation (config preprocess.normalize.method):
     relative  : detrended / trend   -> removes reporter brightness and its slow drift, KEEPS the
                 oscillation's relative amplitude and damping (default; amplitude is part of the
                 biology we want theta to capture)
     envelope  : detrended / amplitude envelope (sliding max-min, window 48 h) -> removes amplitude,
                 shape only (what the authors call 'normalized')
     zscore    : (detrended - mean) / std per trace
3. Windows: length_samples (multiple of 50, VABAM frames the input in blocks of 50), stride_samples.
   With dt = 10 min, 400 samples = 66.7 h = 2.8 cycles of a 24-h rhythm (2.2-3.3 cycles for 20-30 h).
   Only finite samples are used (one MSB trace ends at 134.7 h).
4. Scaling to [0, 1] (VABAM's reconstructor ends in a sigmoid): x01 = 0.5 + x / (2 q), clipped,
   q = quantile(|x|, scale_quantile) over TRAINING windows only. Symmetric, so 0 maps to 0.5 and the
   relative amplitude stays comparable across windows and traces (one global scale, as in VABAM).
5. Train/validation split by TRACE (not window), so overlapping windows of one trace never sit on
   both sides.

Usage:
    python -m src.data.preprocess --config configs/circadian_vabam.yml
Outputs (data/processed/):
    windows_<tag>.npz      X01 (N, L) float32, Xn (N, L) normalised before scaling, split (N,) str
    windows_<tag>_meta.csv one row per window: trace_id, t_start_h, cell_line, genotype, reporter, ...
    scaling_<tag>.json     q, settings, counts
and results/preprocess/qc_*.png, results/preprocess/detrend_check.csv
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_traces, load_msb2025_processed
from src.vabam_bridge import SPINOFF_ROOT, assert_inside_spinoff

M_MAX = 2000


# ---------------------------------------------------------------- detrending / normalisation
def blackman_sinc_lowpass(M: int, f_c: float) -> np.ndarray:
    """Windowed-sinc low-pass kernel of length M+1 (M even), cut-off f_c in cycles/sample, unit DC gain."""
    assert M % 2 == 0, "M must be even"
    n = np.arange(M + 1) - M / 2
    h = np.where(n == 0, 2 * np.pi * f_c, np.sin(2 * np.pi * f_c * n) / np.where(n == 0, 1, n))
    x = np.arange(M + 1)
    h = h * (0.42 - 0.5 * np.cos(2 * np.pi * x / M) + 0.08 * np.cos(4 * np.pi * x / M))
    return h / h.sum()


def sinc_trend(signal: np.ndarray, T_c: float, dt: float, M: int = None) -> np.ndarray:
    """Low-frequency trend (periods > T_c) of `signal` via mirror-padded convolution."""
    x = np.asarray(signal, float)
    if M is None:
        M = min(len(x) - 1, M_MAX)
        M -= M % 2
    w = blackman_sinc_lowpass(M, dt / T_c)
    L = len(w)
    s = np.r_[x[L - 1:0:-1], x, x[-1:-L:-1]]
    y = np.convolve(w, s, mode="valid")
    h = (L - 1) // 2
    return y[h:len(y) - h]


def sliding_amplitude(signal: np.ndarray, window_h: float, dt: float) -> np.ndarray:
    """(max - min)/2 in a centred sliding window (window shrinks at the edges), Savitzky-Golay smoothed."""
    from scipy.signal import savgol_filter

    x = np.asarray(signal, float)
    w = int(window_h / dt)
    w += 1 - w % 2
    h = w // 2
    amp = np.array([(x[max(0, i - h):i + h + 1].max() - x[max(0, i - h):i + h + 1].min()) / 2 for i in range(len(x))])
    return savgol_filter(amp, window_length=min(w, len(x) - (1 - len(x) % 2)), polyorder=3)


def normalise_trace(raw: np.ndarray, dt: float, cfg_pre: dict) -> dict:
    T_c = cfg_pre["detrend"]["T_c_h"]
    trend = sinc_trend(raw, T_c, dt)
    detr = raw - trend
    method = cfg_pre["normalize"]["method"]
    if method == "relative":
        if np.any(trend <= 0):
            raise ValueError("Non-positive trend; 'relative' normalisation needs positive luminescence.")
        xn = detr / trend
    elif method == "envelope":
        env = sliding_amplitude(detr - detr.mean(), cfg_pre["normalize"].get("envelope_window_h", 48.0), dt)
        xn = detr / np.maximum(env, 1e-9)
    elif method == "zscore":
        xn = (detr - detr.mean()) / detr.std()
    else:
        raise ValueError("Unknown normalisation %r" % method)
    return {"trend": trend, "detrended": detr, "norm": xn}


# ---------------------------------------------------------------- windowing
def make_windows(x: np.ndarray, t: np.ndarray, length: int, stride: int):
    starts = np.arange(0, len(x) - length + 1, stride)
    W = np.stack([x[s:s + length] for s in starts]) if len(starts) else np.zeros((0, length))
    return W, t[starts]


def split_traces(meta: pd.DataFrame, val_frac: float, seed: int, exclude_labels=()):
    """Trace-level split, stratified by label so each cell line / genotype appears in both sets when possible."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=meta.index)
    for lab, idx in meta.groupby("label").groups.items():
        idx = list(idx)
        if lab in exclude_labels:
            split[idx] = "heldout"
            continue
        n_val = int(round(len(idx) * val_frac))
        if len(idx) >= 3 and n_val == 0:
            n_val = 1
        if n_val:
            split[list(rng.choice(idx, size=n_val, replace=False))] = "val"
    return split


def run(cfg: dict, tag: str = None):
    pre = cfg["preprocess"]
    tag = tag or pre["tag"]
    ts = load_traces(cfg, root=SPINOFF_ROOT)
    keep = ts.meta["reset"].isin(pre.get("resetting", ["Dex_1uM", "Dex_1.5uM", "serum_shock"]))
    ts = ts.select(ids=ts.meta.index[keep])
    dt = ts.dt_h
    if abs(dt * 60 - cfg["data"]["dt_min"]) > 0.1:
        raise ValueError("Sampling interval %.3f min differs from config dt_min" % (dt * 60))

    L, S = pre["window"]["length_samples"], pre["window"]["stride_samples"]
    if L % 50:
        raise ValueError("window.length_samples must be a multiple of 50 (VABAM frames inputs in blocks of 50)")

    split = split_traces(ts.meta, pre["split"]["val_frac"], pre["split"]["seed"], pre["split"].get("exclude_labels", []))

    Xs, metas, per_trace = [], [], {}
    for tid in ts.meta.index:
        raw = ts.values[tid].to_numpy()
        ok = np.isfinite(raw)
        if not ok.all():
            last = np.where(~ok)[0][0]
            if ok[last:].any():
                raise ValueError("Gap inside trace %s" % tid)
            raw = raw[:last]
        t = ts.time[:len(raw)]
        nt = normalise_trace(raw, dt, pre)
        per_trace[tid] = dict(t=t, raw=raw, **nt)
        W, t0 = make_windows(nt["norm"], t, L, S)
        Xs.append(W)
        m = pd.DataFrame({"trace_id": tid, "t_start_h": t0, "win_idx": np.arange(len(t0))})
        for c in ts.meta.columns:
            m[c] = ts.meta.at[tid, c]
        m["split"] = split[tid]
        metas.append(m)
    Xn = np.concatenate(Xs).astype(np.float32)
    meta = pd.concat(metas, ignore_index=True)

    tr = (meta["split"] == "train").to_numpy()
    q = float(np.quantile(np.abs(Xn[tr]), pre["scale_quantile"]))
    X01 = np.clip(0.5 + Xn / (2 * q), 0.0, 1.0).astype(np.float32)
    clipped = float(np.mean((X01 == 0) | (X01 == 1)))

    out = assert_inside_spinoff(SPINOFF_ROOT / cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / ("windows_%s.npz" % tag), X01=X01, Xn=Xn, split=meta["split"].to_numpy().astype(str))
    meta.to_csv(out / ("windows_%s_meta.csv" % tag), index=False)
    info = {
        "tag": tag, "q_abs": q, "scale_quantile": pre["scale_quantile"], "clipped_fraction": clipped,
        "dt_h": dt, "length_samples": L, "length_h": L * dt, "stride_samples": S, "stride_h": S * dt,
        "normalize": pre["normalize"], "detrend": pre["detrend"],
        "n_traces": int(len(ts)), "n_windows": int(len(meta)),
        "n_windows_by_split": meta["split"].value_counts().to_dict(),
        "n_traces_by_split": split.value_counts().to_dict(),
    }
    with open(out / ("scaling_%s.json" % tag), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # full normalised traces (NaN-padded) for on-the-fly window sampling in train.py (sampling: random_crop)
    ids = list(ts.meta.index)
    lens = np.array([len(per_trace[t]["norm"]) for t in ids])
    full = np.full((len(ids), lens.max()), np.nan, dtype=np.float32)
    for i, t in enumerate(ids):
        full[i, :lens[i]] = per_trace[t]["norm"]
    np.savez_compressed(out / ("traces_%s.npz" % tag), Xn_full=full, lengths=lens,
                        trace_id=np.array(ids, dtype=str), split=split[ids].to_numpy().astype(str),
                        label=ts.meta.loc[ids, "label"].to_numpy().astype(str))
    return ts, per_trace, meta, info


# ---------------------------------------------------------------- QC
def detrend_check(cfg, ts, per_trace):
    """Compare our detrended signal with the authors' pyBOAT output (MSB 2025 'detrended' sheet), if present."""
    spec = cfg["data"].get("authors_processed")
    if not spec:
        return None
    p = SPINOFF_ROOT / spec["path"]
    if not p.exists():
        return None
    auth = load_msb2025_processed(p, sheet="detrended", time=ts.time)
    rows = []
    for tid in per_trace:
        if tid not in auth.values.columns:
            continue
        a = auth.values[tid].to_numpy()
        b = per_trace[tid]["detrended"]
        n = min(np.isfinite(a).sum(), len(b))
        a, b = a[:n], b[:n]
        rows.append({"trace_id": tid, "pearson_r": float(np.corrcoef(a, b)[0, 1]),
                     "max_abs_diff_rel": float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-12))})
    return pd.DataFrame(rows)


def qc_figure(per_trace, meta, labels, path, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(labels), 3, figsize=(13, 2.4 * len(labels)), squeeze=False)
    for i, (lab, rep) in enumerate(labels):
        ids = meta.loc[(meta["label"] == lab) & (meta["reporter"] == rep), "trace_id"].unique()
        if len(ids) == 0:
            continue
        tid = ids[0]
        d = per_trace[tid]
        ax = axes[i, 0]
        ax.plot(d["t"], d["raw"], color="0.2", lw=1, label="raw")
        ax.plot(d["t"], d["trend"], color="C3", lw=1.2, label="trend (T_c 48 h)")
        ax.set_title("%s %s — %s" % (lab, rep, tid), fontsize=8, loc="left")
        ax.legend(fontsize=7, frameon=False)
        ax = axes[i, 1]
        for t2 in ids:
            ax.plot(per_trace[t2]["t"], per_trace[t2]["norm"], lw=0.8, alpha=0.8)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_title("normalised, all %d traces" % len(ids), fontsize=8, loc="left")
        ax = axes[i, 2]
        L = None
        sub = meta[(meta["trace_id"] == tid)]
        ax.set_title("windows of first trace (start times)", fontsize=8, loc="left")
        for _, r in sub.iloc[::4].iterrows():
            ax.axvspan(r["t_start_h"], r["t_start_h"] + 400 * dt, alpha=0.06, color="C0")
        ax.plot(d["t"], d["norm"], color="C0", lw=0.9)
    for ax in axes[-1]:
        ax.set_xlabel("time since start of recording [h]")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--length", type=int, default=None, help="override window.length_samples")
    ap.add_argument("--tag", default=None, help="override preprocess.tag (output file names)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.length:
        cfg["preprocess"]["window"]["length_samples"] = args.length
        if not args.tag:
            raise SystemExit("--length needs --tag so existing processed files are not overwritten")
    ts, per_trace, meta, info = run(cfg, tag=args.tag)
    print(json.dumps(info, indent=2))
    res = assert_inside_spinoff(SPINOFF_ROOT / cfg["paths"]["results"] / "preprocess")
    res.mkdir(parents=True, exist_ok=True)
    chk = detrend_check(cfg, ts, per_trace)
    if chk is not None and len(chk):
        chk.to_csv(res / "detrend_check.csv", index=False)
        print("detrend vs authors' pyBOAT output: median r=%.4f, min r=%.4f, median rel. max diff=%.4f (n=%d)" % (
            chk.pearson_r.median(), chk.pearson_r.min(), chk.max_abs_diff_rel.median(), len(chk)))
    qc_labels = [("U2OS", "Bmal1"), ("U2OS_CRY1_sKO", "Bmal1"), ("U2OS_CRY2_sKO", "Bmal1"),
                 ("U2OS_CRY1_2_dKO", "Bmal1")]
    qc_figure(per_trace, meta, qc_labels, res / ("qc_u2os_%s.png" % info["tag"]), info["dt_h"])
    print("wrote", res)


if __name__ == "__main__":
    main()
