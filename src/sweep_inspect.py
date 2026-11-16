"""
Per-window inspection of finished sweep runs (uses TensorFlow on CPU or GPU; read-only on checkpoints).

Same definitions as sweep.evaluate_run, but on every MSB validation window that is not U2OS (no subsampling),
kept per window so that results can be split by cell line. Cached to runs/<run_id>/per_window.csv.
Descriptive only: nothing here is used to choose a configuration.
"""
import json

import numpy as np
import pandas as pd

from . import sweep as sw


def _val_windows(cfg, mc):
    from .train import load_windows

    X01, _, meta = load_windows(cfg, mc["DataSource"])
    idx = np.where(((meta["split"] == "val") & (meta["cell_line"] != "U2OS")).to_numpy())[0]
    return X01[idx], meta.iloc[idx].reset_index(drop=True)


def _model(stage, run_id):
    cfg, scfg = sw.load_cfgs()
    rd = sw.run_dir(scfg, stage, run_id)
    p = json.load(open(rd / "params.json", encoding="utf-8"))
    mc = p["model_config"]
    ckpt = sw.stage_dir(scfg, stage) / "ckpt" / run_id / "model.hdf5"
    return cfg, scfg, mc, ckpt, rd


def theta_sweep(parts, theta, z, fc_limit, sweep_theta):
    """Generated signals (centred) for each theta_L in sweep_theta; other theta components as inferred."""
    from .exp01_u2os_cry import generate

    gens = []
    for th in sweep_theta:
        fc = theta * fc_limit
        fc[:, -1] = th * fc_limit
        gens.append(generate(parts, fc, z) - 0.5)
    return np.array(gens)          # (n_theta, n_windows, length)


def per_window(stage, run_id, refresh=False):
    from scipy.stats import spearmanr

    from .exp01_u2os_cry import encode_generate

    cfg, scfg, mc, ckpt, rd = _model(stage, run_id)
    cache = rd / "per_window.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)
    x, meta = _val_windows(cfg, mc)
    parts, lv = sw._load_parts(cfg, mc, x.shape[1], ckpt)
    theta, z, _, gen, _ = encode_generate(parts, lv, x, mc["FcLimit"])
    sel = scfg["selection"]
    sweep_theta = np.array(sel["sweep_theta"])
    gens = theta_sweep(parts, theta, z, mc["FcLimit"], sweep_theta)
    base = (gen - 0.5)
    amps = gens.std(2)
    ok = base.std(1) > 1e-6
    ratio = amps[-1] / np.maximum(amps[0], 1e-12)
    ctrl = np.array([spearmanr(sweep_theta, amps[:, k])[0] if np.ptp(amps[:, k]) > 0 else 0.0
                     for k in range(len(x))])
    shape = np.array([min(np.corrcoef(gens[j, k], base[k])[0, 1] for j in range(len(sweep_theta)))
                      if ok[k] else np.nan for k in range(len(x))])
    cp = sel["control_pass"]
    r2 = 1 - np.sum((gen - x) ** 2, 1) / np.maximum(np.sum((x - x.mean(1, keepdims=True)) ** 2, 1), 1e-12)
    df = meta[["trace_id", "cell_line", "t_start_h", "win_idx"]].copy()
    df["input_sd"] = x.std(1)
    df["recon_r2"] = r2
    df["amp_ratio"] = ratio
    df["thetaL_ctrl"] = ctrl
    df["shape_keep"] = shape
    df["pass_amp"] = ratio >= cp["amp_ratio_min"]
    df["pass_ctrl"] = ctrl >= cp["ctrl_min"]
    df["pass_shape"] = shape >= cp["shape_min"]
    df["control_pass"] = df.pass_amp & df.pass_ctrl & df.pass_shape & ok
    df.insert(0, "run_id", run_id)
    df.to_csv(cache, index=False)
    return df


def all_per_window(stage, configs=None, refresh=False):
    """Per-window metrics for the finished runs of a stage (only `configs` if given), with a progress line."""
    import time

    _, scfg = sw.load_cfgs()
    runs = [r for r in sw.load_plan(stage)
            if sw._done(sw.run_dir(scfg, stage, r["run_id"])) and (configs is None or r["config_name"] in configs)]
    out, t0 = [], time.time()
    for i, r in enumerate(runs, 1):
        cached = (sw.run_dir(scfg, stage, r["run_id"]) / "per_window.csv").exists() and not refresh
        print("[%d/%d] %s %s ..." % (i, len(runs), r["run_id"], "(cached)" if cached else ""), end=" ", flush=True)
        d = per_window(stage, r["run_id"], refresh=refresh)
        d["config_name"] = r["config_name"]
        out.append(d)
        print("done, %.0f s elapsed" % (time.time() - t0), flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def examples(stage, run_ids, n_windows=3, seed=1):
    """Arrays for plotting: input, reconstruction and theta_L sweep for the same windows in every run."""
    from .exp01_u2os_cry import encode_generate

    cfg, scfg = sw.load_cfgs()
    out = {}
    pick = None
    for n, rid in enumerate(run_ids, 1):
        print("[%d/%d] %s" % (n, len(run_ids), rid), flush=True)
        cfg, scfg, mc, ckpt, _ = _model(stage, rid)
        x, meta = _val_windows(cfg, mc)
        if pick is None:   # one window from each of n_windows different cell lines, same for every run
            rng = np.random.default_rng(seed)
            lines = rng.choice(meta["cell_line"].unique(), n_windows, replace=False)
            pick = [int(rng.choice(np.where(meta["cell_line"] == c)[0])) for c in lines]
        parts, lv = sw._load_parts(cfg, mc, x.shape[1], ckpt)
        theta, z, _, gen, _ = encode_generate(parts, lv, x[pick], mc["FcLimit"])
        sweep_theta = np.array(scfg["selection"]["sweep_theta"])
        out[rid] = {"x": x[pick], "gen": gen, "sweep": theta_sweep(parts, theta, z, mc["FcLimit"], sweep_theta),
                    "sweep_theta": sweep_theta, "cell_line": meta["cell_line"].iloc[pick].tolist(),
                    "trace_id": meta["trace_id"].iloc[pick].tolist()}
    return out
