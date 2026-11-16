"""
Label-free model selection (decided BEFORE looking at genotype results).

    python run.py select                      # all models in the config that have a checkpoint
    python run.py select --models A B C

Uses only validation traces that are NOT U2OS (the exp01 traces never influence the choice) and never uses
genotype labels. Only models whose training finished (results/train/<model>_run.json) are compared unless
--models is given.

Note on theta: in VABAM the encoder's cut-off mean (FC_Mu) is pulled to the Bernoulli(0.5) prior (KL with a
tiny capacity) and theta is SAMPLED ~U(0,1) during training. The inferred theta is therefore (almost) constant
across inputs by design; theta is a generation-time amplitude control, not a descriptor of the input. The
criteria below evaluate it as a control.

Ranking criteria (pre-specified):
  1. recon_r2      median R^2 of the generated reconstruction (z, inferred theta), validation windows  [higher]
  2. amp_ratio     median generated amplitude at theta_L=0.95 / at theta_L=0.05 (same z)              [higher]
  3. thetaL_ctrl   median Spearman(theta_L sweep, generated amplitude): monotonic control             [higher]
  4. shape_keep    median corr between generations at theta_L = 0.5 and 0.95 (same z)                [higher]
Diagnostics (reported, not ranked): phase_R2, theta_from_z, thetaL_sd, z_active_dims.
Rule: rank each criterion across models, sum the ranks, lowest sum = selected. Ties -> higher recon_r2.
Output: results/model_selection.csv, results/model_selection.json
"""
import argparse
import json

from . import vabam_bridge as vb  # before tensorflow

import numpy as np
import pandas as pd
import yaml

from .exp01_u2os_cry import cosinor_periodogram, encode_generate, generate, load_parts, to_rel
from .train import load_windows

CRITERIA = [("recon_r2", False), ("amp_ratio", False), ("thetaL_ctrl", False), ("shape_keep", False)]
# (name, lower_is_better)


def evaluate(cfg, name, n_sweep, rng):
    mc = vb.model_config(cfg, name)
    X01, Xn, meta = load_windows(cfg, mc["DataSource"])
    scal = json.load(open(vb.SPINOFF_ROOT / cfg["paths"]["processed"] / ("scaling_%s.json" % mc["DataSource"])))
    q, dt_h = scal["q_abs"], scal["dt_h"]
    m = ((meta["split"] == "val") & (meta["cell_line"] != "U2OS")).to_numpy()
    iv = np.where(m)[0]
    mc, parts, logvar_model = load_parts(cfg, name, X01.shape[1])
    theta, z, _, gen, _ = encode_generate(parts, logvar_model, X01[iv], mc["FcLimit"])
    x = X01[iv]
    r2 = 1 - np.sum((gen - x) ** 2, 1) / np.maximum(np.sum((x - x.mean(1, keepdims=True)) ** 2, 1), 1e-12)

    # theta_L sweep on a random subset of validation windows
    sub = rng.choice(len(iv), size=min(n_sweep, len(iv)), replace=False)
    sweep = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    fcs0 = theta[sub] * mc["FcLimit"]
    amps, gens = [], []
    for th in sweep:
        fc = fcs0.copy()
        fc[:, -1] = th * mc["FcLimit"]
        g = to_rel(generate(parts, fc, z[sub]), q)
        gens.append(g)
        amps.append(np.sqrt(2) * g.std(1))
    amps = np.array(amps)  # (n_theta, n_win)
    from scipy.stats import spearmanr

    ctrl = np.array([spearmanr(sweep, amps[:, k])[0] if np.ptp(amps[:, k]) > 0 else 0.0 for k in range(amps.shape[1])])
    shape = np.array([np.corrcoef(gens[2][k], gens[4][k])[0, 1] for k in range(len(sub))])

    # phase leakage of theta_L within validation traces
    mv = meta.iloc[iv].reset_index(drop=True)
    cos = cosinor_periodogram(Xn[iv], dt_h)
    pr2 = []
    for t, idx in mv.groupby("trace_id").groups.items():
        idx = np.asarray(idx)
        if len(idx) < 6:
            continue
        ts_h = mv.loc[idx, "t_start_h"].to_numpy()
        rhythmic = cos["cos_r2"].to_numpy()[idx] > 0.5
        P = float(np.clip(np.median(cos["cos_period_h"].to_numpy()[idx][rhythmic]), 18, 30)) if rhythmic.any() else 24.0
        A0 = np.column_stack([np.ones(len(idx)), ts_h - ts_h.mean()])
        A1 = np.column_stack([A0, np.cos(2 * np.pi * ts_h / P), np.sin(2 * np.pi * ts_h / P)])
        y = theta[idx, -1]
        r0 = y - A0 @ np.linalg.lstsq(A0, y, rcond=None)[0]
        r1 = y - A1 @ np.linalg.lstsq(A1, y, rcond=None)[0]
        pr2.append(1 - np.sum(r1 ** 2) / max(np.sum(r0 ** 2), 1e-20))

    # how much of theta_L is linearly predictable from z
    Z1 = np.column_stack([np.ones(len(z)), z])
    y = theta[:, -1]
    res = y - Z1 @ np.linalg.lstsq(Z1, y, rcond=None)[0]
    tz = 1 - np.sum(res ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-20)

    return {
        "model": name, "window": int(X01.shape[1]), "FcLimit": mc["FcLimit"], "Sampling": mc.get("Sampling", "fixed"),
        "BatSize": mc["BatSize"], "n_val_windows": int(len(iv)), "n_val_traces": int(mv.trace_id.nunique()),
        "recon_r2": float(np.median(r2)), "thetaL_ctrl": float(np.median(ctrl)), "shape_keep": float(np.median(shape)),
        "phase_R2": float(np.median(pr2)) if pr2 else np.nan, "theta_from_z": float(tz),
        "thetaL_sd": float(theta[:, -1].std()), "z_active_dims": int(np.sum(np.var(z, 0) > 0.01)),
        "amp_ratio": float(np.median(amps[-1] / np.maximum(amps[0], 1e-9))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--n_sweep", type=int, default=200)
    ap.add_argument("--gpu", type=str, default="-1")
    args = ap.parse_args()
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    finished = lambda n: (vb.checkpoint_path(cfg, n).exists()
                          and (vb.results_dir(cfg) / "train" / ("%s_run.json" % n)).exists())
    names = args.models or [n for n in cfg["vabam"]["Models"] if finished(n)]
    if not names:
        raise SystemExit("No trained checkpoints found under results/ckpt/")
    rows = []
    for n in names:
        import tensorflow as tf

        tf.keras.backend.clear_session()
        rows.append(evaluate(cfg, n, args.n_sweep, np.random.default_rng(cfg["train"]["seed"])))
        print(pd.DataFrame([rows[-1]]).to_string(index=False))
    df = pd.DataFrame(rows)
    ranks = pd.DataFrame({c: df[c].rank(ascending=low) for c, low in CRITERIA})
    df["rank_sum"] = ranks.sum(1)
    df = df.sort_values(["rank_sum", "recon_r2"], ascending=[True, False]).reset_index(drop=True)
    out = vb.results_dir(cfg)
    df.to_csv(out / "model_selection.csv", index=False)
    rule = ("Lowest rank sum over recon_r2(+), amp_ratio(+), thetaL_ctrl(+), shape_keep(+) on non-U2OS "
            "validation windows; genotype labels not used.")
    with open(out / "model_selection.json", "w", encoding="utf-8") as f:
        json.dump({"selected": df.loc[0, "model"], "rule": rule, "table": df.to_dict("records")}, f, indent=2,
                  default=float)
    pd.set_option("display.width", 220)
    print("\n" + df.round(4).to_string(index=False))
    print("\nselected:", df.loc[0, "model"], "|", rule)


if __name__ == "__main__":
    main()
