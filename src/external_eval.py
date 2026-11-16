"""
Label-free evaluation on validation and external sets of configuration B (data/processed/<tag>/).

    python run.py external --model SKZFC_CIRC_1_30_200_MULTI
    python run.py external --models SKZFC_CIRC_1_30_200_POP SKZFC_CIRC_1_30_200_MULTI --gpu 0

Sets: 'val:<source>' (validation records of each training source) and every external split in records.csv
(E1 Circa-SCOPE RawData_7, E2 Nat Commun 2024 new cell lines, E3 HES1 long tracks).
For each set (random subset of at most --max_windows windows, fixed seed):
  recon_r2 / recon_mse   generated reconstruction (inferred z and theta), [0, 1] scale
  amp_ratio              generated amplitude at theta_L = 0.95 / at 0.05 (same z), median over --n_sweep windows
  thetaL_ctrl            Spearman(theta_L sweep, amplitude), median
  shape_keep             corr(generation at theta_L 0.5, at 0.95), median
Counts reported per set: windows available / evaluated, records, independent groups (trace or cell).
Outputs: results/external_<model>/per_set.csv, summary.json; results/external_comparison.csv when several models.
"""
import argparse
import json
import os

from . import vabam_bridge as vb  # before tensorflow

import numpy as np
import pandas as pd
import yaml


def window_index(rec, idx, L):
    rows = []
    for i in idx:
        s_ = int(rec.at[i, "stride"])
        rows += [(i, s) for s in range(0, int(rec.at[i, "length"]) - L + 1, s_)]
    return np.array(rows, dtype=np.int64).reshape(-1, 2)


def evaluate_model(cfg, name, max_windows, n_sweep, seed):
    from scipy.stats import spearmanr

    from .exp01_u2os_cry import encode_generate, generate, load_parts

    mc = vb.model_config(cfg, name)
    tag = cfg["multi"]["tag"]
    d = vb.SPINOFF_ROOT / cfg["paths"]["processed"] / tag
    rec = pd.read_csv(d / "records.csv", low_memory=False)
    Xf = np.load(d / "traces.npz")["Xn_full"]
    L = int(json.load(open(d / "scaling.json", encoding="utf-8"))["window"])
    q = rec["q"].to_numpy(np.float32)
    mc, parts, logvar = load_parts(cfg, name, L)
    sets = [("val:" + s_, rec.index[(rec["split"] == "val") & (rec["source"] == s_)]) for s_ in
            sorted(rec.loc[rec["split"] == "val", "source"].unique())]
    sets += [(e, rec.index[rec["split"] == e]) for e in cfg["multi"]["external"] if (rec["split"] == e).any()]
    rows = []
    sweep = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    offs = np.arange(L)
    for set_name, idx in sets:
        rng = np.random.default_rng(seed)
        W = window_index(rec, idx, L)
        sel = W if len(W) <= max_windows else W[np.sort(rng.choice(len(W), max_windows, replace=False))]
        Xn = Xf[sel[:, 0][:, None], sel[:, 1][:, None] + offs]
        qq = q[sel[:, 0]][:, None]
        X01 = np.clip(0.5 + Xn / (2 * qq), 0, 1).astype(np.float32)
        theta, z, _, gen, _ = encode_generate(parts, logvar, X01, mc["FcLimit"])
        sst = np.sum((X01 - X01.mean(1, keepdims=True)) ** 2, 1)
        r2 = 1 - np.sum((gen - X01) ** 2, 1) / np.maximum(sst, 1e-12)
        sub = rng.choice(len(X01), size=min(n_sweep, len(X01)), replace=False)
        amps, gens = [], []
        for th in sweep:
            fc = theta[sub] * mc["FcLimit"]
            fc[:, -1] = th * mc["FcLimit"]
            g = (generate(parts, fc, z[sub]) - 0.5) * 2 * qq[sub]
            gens.append(g)
            amps.append(np.sqrt(2) * g.std(1))
        amps = np.array(amps)
        ctrl = [spearmanr(sweep, amps[:, k])[0] if np.ptp(amps[:, k]) > 0 else 0.0 for k in range(len(sub))]
        shape = [np.corrcoef(gens[2][k], gens[4][k])[0, 1] for k in range(len(sub))]
        rows.append({
            "model": name, "set": set_name, "windows_available": int(len(W)), "windows_evaluated": int(len(sel)),
            "records": int(len(idx)), "independent_groups": int(rec.loc[idx, "group_id"].nunique()),
            "recon_r2_median": float(np.median(r2)), "recon_r2_q25": float(np.quantile(r2, 0.25)),
            "recon_mse_median": float(np.median(np.mean((gen - X01) ** 2, 1))),
            "amp_ratio": float(np.median(amps[-1] / np.maximum(amps[0], 1e-12))),
            "thetaL_ctrl": float(np.nanmedian(ctrl)), "shape_keep": float(np.nanmedian(shape)),
        })
        print(pd.DataFrame([rows[-1]]).round(4).to_string(index=False, header=len(rows) == 1))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--max_windows", type=int, default=20000)
    ap.add_argument("--n_sweep", type=int, default=300)
    ap.add_argument("--gpu", type=str, default="-1")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    import tensorflow as tf

    allres = []
    for name in args.models:
        tf.keras.backend.clear_session()
        df = evaluate_model(cfg, name, args.max_windows, args.n_sweep, cfg["train"]["seed"])
        out = vb.assert_inside_spinoff(vb.results_dir(cfg) / ("external_%s" % name))
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "per_set.csv", index=False)
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"model": name, "data_tag": cfg["multi"]["tag"], "max_windows": args.max_windows,
                       "n_sweep": args.n_sweep, "label_free": True, "sets": df.to_dict("records")}, f, indent=2)
        allres.append(df)
    if len(allres) > 1:
        comp = pd.concat(allres)
        comp.to_csv(vb.results_dir(cfg) / "external_comparison.csv", index=False)
        pd.set_option("display.width", 220)
        print("\n" + comp.pivot(index="set", columns="model", values=["recon_r2_median", "amp_ratio", "shape_keep"])
              .round(3).to_string())


if __name__ == "__main__":
    main()
