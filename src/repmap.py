"""
theta + z vs single z: representations of one VABAM model for the exp02 downstream analysis.

For every window of data/processed/windows_msb_rel_L400_S18 (the windows exp02 uses) this writes three
representations of one VABAM model to results/exp02/codes/<rep>.npz (key 'z', rows aligned with exp02):

    <name>_z       z (encoder mean)                                        what exp02 already uses
    <name>_theta   theta response: how the generated signal reacts when theta_L is turned
                   (log amplitude relative to theta_L = 0.5, and shape correlation with it, at the other levels)
    <name>_ztheta  both, each block standardised

The inferred theta is the same for every input (exp01), so theta itself carries no information; the theta
response is what theta adds as a control knob. exp02's analysis code is used unchanged (src/exp02/analyze.py).

model: a model name in configs/circadian_vabam.yml (checkpoint results/ckpt/<SigType>/<name>.hdf5), or a sweep
run 'stage/run_id' (checkpoint results/sweep/<stage>/ckpt/<run_id>/model.hdf5). Sweep runs were trained on
multi_v1 scaling, so their input is rescaled from the same normalised windows (Xn) with multi_v1's q.
"""
import json

import numpy as np
import yaml

from . import vabam_bridge as vb

TAG = "msb_rel_L400_S18"
SWEEP_THETA = (0.05, 0.25, 0.5, 0.75, 0.95)


def _load(model):
    cfg = yaml.safe_load(open(vb.SPINOFF_ROOT / "configs" / "circadian_vabam.yml", encoding="utf-8"))
    proc = vb.SPINOFF_ROOT / cfg["paths"]["processed"]
    W = np.load(proc / ("windows_%s.npz" % TAG))
    if "/" in model:                                   # sweep run
        stage, rid = model.split("/")
        rd = vb.SPINOFF_ROOT / "results" / "sweep" / stage
        mc = json.load(open(rd / "runs" / rid / "params.json", encoding="utf-8"))["model_config"]
        ckpt = rd / "ckpt" / rid / "model.hdf5"
        q = json.load(open(proc / "multi_v1" / "scaling.json", encoding="utf-8"))["q"]["pop"]
        X01 = np.clip(0.5 + W["Xn"] / (2 * q), 0, 1).astype("float32")
        name = "VABAM_" + rid.replace("_s0", "").replace(stage + "_", stage + "-")
    else:                                              # model in the main config
        mc = vb.model_config(cfg, model)
        ckpt = vb.checkpoint_path(cfg, model)
        X01 = W["X01"].astype("float32")
        name = "VABAM_" + model
    return cfg, mc, ckpt, X01, name


def theta_response(model, sweep_theta=SWEEP_THETA, bs=1024):
    """Returns (z, response, name). response columns: dlogamp at each theta_L != 0.5, then shape corr."""
    import tensorflow as tf

    from .exp01_u2os_cry import generate

    cfg, mc, ckpt, X01, name = _load(model)
    v = vb.import_vabam(cfg)
    _, parts = v.ModelCall(mc, X01.shape[1], 1000, LoadWeight=True, ReturnModelPart=True, Reparam=False,
                           ModelSaveName=str(ckpt), ModelSummary=False)
    _, fcs, z = parts[0].predict(X01, batch_size=bs, verbose=0)
    gens = []
    for th in sweep_theta:
        fc = fcs.copy()
        fc[:, -1] = th * mc["FcLimit"]
        gens.append(generate(parts, fc, z, bs) - 0.5)
    gens = np.array(gens)                              # (n_theta, n_windows, length)
    ref = list(sweep_theta).index(0.5)
    amp = np.log(gens.std(2) + 1e-6)
    others = [k for k in range(len(sweep_theta)) if k != ref]
    g0 = gens[ref] - gens[ref].mean(1, keepdims=True)
    corr = []
    for k in others:
        gk = gens[k] - gens[k].mean(1, keepdims=True)
        corr.append((gk * g0).sum(1) / np.sqrt((gk ** 2).sum(1) * (g0 ** 2).sum(1) + 1e-12))
    resp = np.column_stack([amp[k] - amp[ref] for k in others] + corr)
    tf.keras.backend.clear_session()
    return z, resp, name


def _std(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-9)


def write_codes(model):
    """Write <name>_z, <name>_theta, <name>_ztheta to results/exp02/codes/. Returns {rep: path}."""
    z, resp, name = theta_response(model)
    out = vb.assert_inside_spinoff(vb.SPINOFF_ROOT / "results" / "exp02" / "codes")
    out.mkdir(parents=True, exist_ok=True)
    reps = {name + "_z": z, name + "_theta": resp, name + "_ztheta": np.hstack([_std(z), _std(resp)])}
    paths = {}
    for rep, arr in reps.items():
        paths[rep] = out / ("%s.npz" % rep)
        np.savez_compressed(paths[rep], z=arr.astype("float32"))
        print("%-45s %d x %d" % (rep, *arr.shape))
    return paths


def table(reps=None):
    """Main numbers of results/exp02/<rep>/summary.json (same columns as exp02 compare)."""
    import pandas as pd

    from .exp02 import compare
    from .exp02.common import RES

    rows = []
    for p in sorted(RES.glob("*/summary.json")):
        s = json.load(open(p, encoding="utf-8"))
        if reps is None or s["rep"] in reps:
            rows.append({"rep": s["rep"], "dim": s["latent_dim"], **compare.row(s, "rep")})
    if rows:
        rows.append({"rep": "classical (reference)", "dim": 6, **compare.row(s, "classical")})
    return pd.DataFrame(rows)


def plot_table(T, save=True):
    import matplotlib.pyplot as plt

    m = [("phenotype_R2", "phenotype R²"), ("line_id_acc_Per2", "cell-line ID (Per2)"),
         ("cisplatin_LOO_rho", "cisplatin LOO ρ"), ("gdsc_drugs_LOO_gt_0.5", "GDSC drugs ρ>0.5")]
    m = [(k, t) for k, t in m if k in T and T[k].notna().any()]
    col = ["#1d4ed8" if r.endswith("_ztheta") else "#60a5fa" if r.endswith("_theta") else
           "#93c5fd" if "VABAM" in r else "#9ca3af" if "classical" in r else "#f59e0b" for r in T.rep]
    fig, axes = plt.subplots(1, len(m), figsize=(3.6 * len(m), 0.38 * len(T) + 1.2), sharey=True, squeeze=False)
    for ax, (k, t) in zip(axes[0], m):
        ax.barh(range(len(T)), T[k], color=col)
        if k == "gdsc_drugs_LOO_gt_0.5" and "gdsc_null_95pct" in T:
            ax.scatter(T["gdsc_null_95pct"], range(len(T)), marker="|", color="k", s=80, label="null 95%")
            ax.legend(fontsize=7, frameon=False)
        ax.set_title(t, fontsize=9)
        ax.grid(axis="x", alpha=0.25)
    axes[0, 0].set_yticks(range(len(T)))
    axes[0, 0].set_yticklabels(T.rep, fontsize=8)
    axes[0, 0].invert_yaxis()
    fig.tight_layout()
    if save:
        out = vb.assert_inside_spinoff(vb.SPINOFF_ROOT / "results" / "figures" / "repmap")
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "theta_z_vs_single_z.png", dpi=130, bbox_inches="tight")
    return fig


def pca_codes(dim=30):
    """Linear floor without training: PCA of the raw windows (same as run_exp02.py pca)."""
    X = np.load(vb.SPINOFF_ROOT / "data" / "processed" / ("windows_%s.npz" % TAG))["X01"].astype(float)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    out = vb.assert_inside_spinoff(vb.SPINOFF_ROOT / "results" / "exp02" / "codes")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / ("PCA%d_raw.npz" % dim), z=(Xc @ Vt[:dim].T).astype("float32"))


def _fast_loo_ridge(X, y, alpha=5.0):
    """Same result as exp02.analyze.loo_ridge (StandardScaler + Ridge(alpha=5), leave one out), in plain numpy:
    about 25x faster because it skips scikit-learn's per-fit overhead (13 tiny fits per drug, 130k drugs x perms)."""
    from scipy.stats import spearmanr

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    pred, I = np.empty(n), alpha * np.eye(p)
    for i in range(n):
        tr = np.arange(n) != i
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 10 * np.finfo(float).eps] = 1.0
        Z, ym = (X[tr] - mu) / sd, y[tr].mean()
        pred[i] = ym + ((X[i] - mu) / sd) @ np.linalg.solve(Z.T @ Z + I, Z.T @ (y[tr] - ym))
    return spearmanr(pred, y)[0]


def _timed(name, fn):
    import time

    def wrap(*a, **k):
        t0 = time.time()
        print("    %s %s ..." % (time.strftime("%H:%M:%S"), name), flush=True)
        out = fn(*a, **k)
        print("    %s %s done (%.1f min)" % (time.strftime("%H:%M:%S"), name, (time.time() - t0) / 60), flush=True)
        return out
    return wrap


def analyze_missing(n_perm=200, n_random=300, redo=()):
    """Run exp02's analysis for every results/exp02/codes/<rep>.npz without a summary yet, with progress lines.
    exp02's files are not changed; while this runs, its leave-one-out ridge is swapped in memory for
    _fast_loo_ridge (checked to give the same numbers here first) and each step prints its time."""
    import time

    from .exp02 import analyze
    from .exp02.common import RES

    rng = np.random.default_rng(0)
    for _ in range(20):                                  # identical results, or do not swap
        X, y = rng.normal(size=(13, 8)), rng.normal(size=13)
        assert abs(_fast_loo_ridge(X, y) - analyze.loo_ridge(X, y)) < 1e-9, "fast ridge differs; not used"
    orig = {k: getattr(analyze, k) for k in ("loo_ridge", "clustering", "drugs", "bridge")}
    analyze.loo_ridge = _fast_loo_ridge
    analyze.clustering = _timed("phenotype / cell-line ID", orig["clustering"])
    analyze.drugs = _timed("drugs (cisplatin, GDSC x %d permutations)" % n_perm, orig["drugs"])
    analyze.bridge = _timed("expression -> survival (%d random signatures)" % n_random, orig["bridge"])
    todo = [p for p in sorted((RES / "codes").glob("*.npz"))
            if not (RES / p.stem / "summary.json").exists() or p.stem in redo]
    print("to analyse (%d): %s" % (len(todo), [p.stem for p in todo]), flush=True)
    t0 = time.time()
    try:
        for i, p in enumerate(todo, 1):
            print("\n=== [%d/%d] %s  (start %s)" % (i, len(todo), p.stem, time.strftime("%H:%M:%S")), flush=True)
            analyze.run(p.stem, p, n_perm=n_perm, n_random=n_random)
            print("=== [%d/%d] %s done, total %.1f min" % (i, len(todo), p.stem, (time.time() - t0) / 60), flush=True)
    finally:
        for k, v in orig.items():
            setattr(analyze, k, v)
    return [p.stem for p in todo]
