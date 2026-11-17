"""
Experiment 01 — U-2 OS wild type vs CRY1 single KO vs CRY1/2 double KO (Bmal1-Luc).

Question: are VABAM's amplitude code (theta, filter cut-offs) and shape code (z, latent) different between
the genotypes, and do the two codes carry different information on circadian recordings?

    python -m src.analysis.exp01_u2os_cry --config configs/circadian_vabam.yml --model SKZFC_CIRC_1_30_200

Design
  * Codes come from the trained model run deterministically (Reparam=False): z = Z_Mu, theta = FC_Mu
    (FCs / FcLimit, in (0, 1)); Depth 1 -> theta = (theta_H, theta_L) = (high-pass, low-pass cut-off).
  * Unit of inference = trace (recording). Windows of one trace overlap and are not independent, so
    tests use trace-mean codes (6 traces per genotype, 3 per experiment date).
  * PERMANOVA (Euclidean on standardised trace means); genotype labels permuted WITHIN experiment date.
  * Classification: leave-one-date-out (train on 2022-06-28, test on 2022-07-05 and vice versa) and
    leave-one-trace-out; LDA with shrinkage on standardised window features; trace call = mean posterior.
  * Baseline: classical per-window features from a cosinor periodogram (period 16-36 h, amplitude, R^2).
  * Generation check: hold z, sweep theta (and vice versa along z-PC1) and measure amplitude, period and
    shape similarity of the generated signal.
Outputs: results/exp01/ (tables, figures, summary.json). Nothing is written elsewhere.
"""
import argparse
import itertools
import json
import os

from src import vabam_bridge as vb  # before tensorflow

import numpy as np
import pandas as pd
import yaml

from src.model.train import load_windows

GROUP_COLORS = {"WT": "#1f77b4", "CRY1_sKO": "#d62728", "CRY1_2_dKO": "#7f7f7f"}


# ---------------------------------------------------------------- model / codes
def load_parts(cfg, model_name, sig_dim, ckpt=None):
    import tensorflow as tf

    v = vb.import_vabam(cfg)
    mc = vb.model_config(cfg, model_name)
    ck = str(ckpt or vb.checkpoint_path(cfg, model_name))
    _, parts = v.ModelCall(mc, sig_dim, 1000, LoadWeight=True, ReturnModelPart=True, Reparam=False,
                           ModelSaveName=ck, ModelSummary=False)
    enc = parts[0]
    logvar = tf.keras.Model(enc.input, enc.get_layer("Z_Log_Sigma").output)
    return mc, parts, logvar


def encode_generate(parts, logvar_model, X01, fc_limit, bs=1024):
    enc, fext, fgen, recon = parts
    _, fcs, zs = enc.predict(X01, batch_size=bs, verbose=0)
    z_logvar = logvar_model.predict(X01, batch_size=bs, verbose=0)
    gen = generate(parts, fcs, zs, bs)
    ext = recon.predict(fext.predict([X01, fcs], batch_size=bs, verbose=0), batch_size=bs, verbose=0)
    return fcs / fc_limit, zs, z_logvar, gen, ext


def generate(parts, fcs, zs, bs=1024):
    _, fext, fgen, recon = parts
    ne = len(fext.output)
    feats = fgen.predict([fcs[:, :-ne], fcs[:, -ne:], zs], batch_size=bs, verbose=0)
    return recon.predict(feats, batch_size=bs, verbose=0)


# ---------------------------------------------------------------- classical features
def cosinor_periodogram(Xn, dt_h, periods=np.arange(16.0, 36.01, 0.1)):
    """Per window: best-fitting period (max R^2 of a + linear trend + cos + sin), its amplitude and R^2."""
    n = Xn.shape[1]
    t = np.arange(n) * dt_h
    Y = Xn.T - Xn.mean(axis=1)
    sst = np.sum(Y ** 2, axis=0)
    best_r2 = np.full(Xn.shape[0], -np.inf)
    best_p = np.zeros(Xn.shape[0])
    best_a = np.zeros(Xn.shape[0])
    for P in periods:
        A = np.column_stack([np.ones(n), t - t.mean(), np.cos(2 * np.pi * t / P), np.sin(2 * np.pi * t / P)])
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        r2 = 1 - np.sum((Y - A @ coef) ** 2, axis=0) / np.maximum(sst, 1e-12)
        upd = r2 > best_r2
        best_r2[upd], best_p[upd] = r2[upd], P
        best_a[upd] = np.hypot(coef[2], coef[3])[upd]
    return pd.DataFrame({"cos_period_h": best_p, "cos_amp_rel": best_a, "cos_r2": best_r2})


def to_rel(x01, q):
    return (x01 - 0.5) * 2 * q


# ---------------------------------------------------------------- statistics
def pseudo_F(X, labels):
    D2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    n, groups = len(labels), np.unique(labels)
    iu = np.triu_indices(n, 1)
    sst = D2[iu].sum() / n
    ssw = 0.0
    for g in groups:
        idx = np.where(labels == g)[0]
        sub = D2[np.ix_(idx, idx)]
        ssw += sub[np.triu_indices(len(idx), 1)].sum() / len(idx)
    a = len(groups)
    return ((sst - ssw) / (a - 1)) / (ssw / (n - a)), 1 - ssw / sst


def permanova(X, labels, strata, n_perm, rng):
    """PERMANOVA on standardised X; labels permuted within strata. Exact if the permutation set is small."""
    keep = X.std(0) > 1e-9                      # drop constant features
    if not keep.any():
        return {"pseudo_F": np.nan, "R2": np.nan, "p": np.nan, "n_perm": 0, "exact": False}
    X = X[:, keep]
    Xs = (X - X.mean(0)) / X.std(0)
    F0, R2 = pseudo_F(Xs, labels)
    blocks = [np.where(strata == s)[0] for s in np.unique(strata)]
    perms_per_block = [list(set(itertools.permutations(labels[b]))) if len(b) <= 8 else None for b in blocks]
    n_exact = np.prod([len(p) for p in perms_per_block]) if all(p is not None for p in perms_per_block) else np.inf
    Fs = []
    if n_exact <= n_perm:
        for combo in itertools.product(*perms_per_block):
            lab = labels.copy()
            for b, pl in zip(blocks, combo):
                lab[b] = pl
            Fs.append(pseudo_F(Xs, lab)[0])
        exact = True
    else:
        for _ in range(n_perm):
            lab = labels.copy()
            for b in blocks:
                lab[b] = rng.permutation(labels[b])
            Fs.append(pseudo_F(Xs, lab)[0])
        exact = False
    Fs = np.array(Fs)
    p = (np.sum(Fs >= F0 - 1e-12) + (0 if exact else 1)) / (len(Fs) + (0 if exact else 1))
    return {"pseudo_F": float(F0), "R2": float(R2), "p": float(p), "n_perm": int(len(Fs)), "exact": exact}


def classify(Xw, yw, groups_w, folds):
    """folds: list of (train_mask, test_mask) over windows. Returns window and trace accuracy."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    win_correct, trace_rows = [], []
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        clf.fit(Xw[tr], yw[tr])
        proba = clf.predict_proba(Xw[te])
        pred = clf.classes_[proba.argmax(1)]
        win_correct.extend(list(pred == yw[te]))
        for tid in np.unique(groups_w[te]):
            m = groups_w[te] == tid
            trace_rows.append((tid, yw[te][m][0], clf.classes_[proba[m].mean(0).argmax()]))
    tr_df = pd.DataFrame(trace_rows, columns=["trace_id", "true", "pred"])
    return float(np.mean(win_correct)), float(np.mean(tr_df["true"] == tr_df["pred"])), tr_df


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", default=None, help="checkpoint override (must be inside the repository)")
    ap.add_argument("--out", default=None, help="output sub-folder of results/ (default exp01_<model>)")
    ap.add_argument("--gpu", type=str, default="-1", help="GPU id; default '-1' = CPU (recommended for precision)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ex = cfg["exp01"]
    rng = np.random.default_rng(cfg["train"]["seed"])
    if args.ckpt:
        if not os.path.isabs(args.ckpt):
            args.ckpt = str(vb.SPINOFF_ROOT / args.ckpt)
        vb.assert_inside_spinoff(args.ckpt)

    mc = vb.model_config(cfg, args.model)
    X01, Xn, meta = load_windows(cfg, mc["DataSource"])
    scal = json.load(open(vb.SPINOFF_ROOT / cfg["paths"]["processed"] / ("scaling_%s.json" % mc["DataSource"])))
    q, dt_h = scal["q_abs"], scal["dt_h"]
    out = vb.assert_inside_spinoff(vb.results_dir(cfg) / (args.out or "exp01_%s" % args.model))
    out.mkdir(parents=True, exist_ok=True)

    mc, parts, logvar_model = load_parts(cfg, args.model, X01.shape[1], ckpt=args.ckpt)
    theta, z, zlv, gen, ext = encode_generate(parts, logvar_model, X01, mc["FcLimit"])
    np.savez_compressed(out / "codes_all_windows.npz", theta=theta, z=z, z_logvar=zlv)

    # reconstruction quality (all windows)
    sse_gen = np.sum((gen - X01) ** 2, 1)
    sst = np.sum((X01 - X01.mean(1, keepdims=True)) ** 2, 1)
    meta = meta.copy()
    meta["mse_gen"] = sse_gen / X01.shape[1]
    meta["mse_ext"] = np.mean((ext - X01) ** 2, 1)
    meta["r2_gen"] = 1 - sse_gen / np.maximum(sst, 1e-12)
    meta["theta_H"], meta["theta_L"] = theta[:, 0], theta[:, -1]
    active = np.var(z, 0) / np.mean(np.exp(zlv), 0)
    n_active = int(np.sum(np.var(z, 0) > 0.01))

    # select U2OS groups
    gmask = np.zeros(len(meta), bool)
    gname = np.array([""] * len(meta), dtype=object)
    for g, sel in ex["groups"].items():
        m = np.ones(len(meta), bool)
        for k, val in sel.items():
            m &= (meta[k] == val).to_numpy()
        gmask |= m
        gname[m] = g
    meta["group"] = gname
    U = meta[gmask].copy()
    iU = np.where(gmask)[0]
    cls = cosinor_periodogram(Xn[iU], dt_h)
    for c in cls.columns:
        U[c] = cls[c].to_numpy()
    U["rel_amp_rms"] = np.sqrt(2) * Xn[iU].std(1)

    # brightness: mean luminescence trend under each window (raw RLU; relative normalisation divides it out)
    from src.data.loader import load_traces
    from src.data.preprocess import normalise_trace

    ts = load_traces(cfg, root=vb.SPINOFF_ROOT)
    stride = int(scal["stride_samples"])
    L = X01.shape[1]
    trend_cache = {}
    bright = np.zeros(len(U))
    for k, (t_id, w) in enumerate(zip(U["trace_id"], U["win_idx"])):
        if t_id not in trend_cache:
            raw = ts.values[t_id].to_numpy()
            raw = raw[np.isfinite(raw)]
            trend_cache[t_id] = normalise_trace(raw, dt_h, cfg["preprocess"])["trend"]
        s0 = int(w) * stride
        bright[k] = trend_cache[t_id][s0:s0 + L].mean()
    U["log10_brightness"] = np.log10(bright)

    # z PCA on the U2OS windows (for display and the z sweep)
    zU = z[iU]
    zc = zU - zU.mean(0)
    _, svals, Vt = np.linalg.svd(zc, full_matrices=False)
    evr = svals ** 2 / np.sum(svals ** 2)
    pcs = zc @ Vt[:3].T
    for k in range(3):
        U["zPC%d" % (k + 1)] = pcs[:, k]
    U.drop(columns=[c for c in U.columns if c.startswith("Unnamed")], errors="ignore").to_csv(
        out / "u2os_windows.csv", index=False)

    theta_sd = theta[iU].std(0)
    theta_constant = bool(np.all(theta_sd < 1e-6))
    feats = {
        "theta (amplitude code, 2-d)": theta[iU],
        "z (shape code, %d-d)" % z.shape[1]: zU,
        "theta+z": np.hstack([theta[iU], zU]),
        "classical (period, amplitude, R2)": U[["cos_period_h", "cos_amp_rel", "cos_r2"]].to_numpy(),
    }

    # trace-level tables
    tid = U["trace_id"].to_numpy()
    traces = U.groupby("trace_id").agg(group=("group", "first"), date=("date", "first"),
                                       replicate=("replicate", "first")).reset_index()
    order = {t: i for i, t in enumerate(traces["trace_id"])}
    labels_t = traces["group"].to_numpy()
    strata_t = traces["date"].astype(str).to_numpy()

    def trace_mean(F):
        M = np.zeros((len(traces), F.shape[1]))
        for t, i in order.items():
            M[i] = F[tid == t].mean(0)
        return M

    tm_cols = {}
    for name, F in feats.items():
        tm_cols[name] = trace_mean(F)
    tr_table = traces.copy()
    for c in ["theta_H", "theta_L", "cos_period_h", "cos_amp_rel", "cos_r2", "rel_amp_rms", "r2_gen", "mse_gen"]:
        tr_table[c] = trace_mean(U[[c]].to_numpy())[:, 0]
    tr_table.to_csv(out / "u2os_traces.csv", index=False)

    # 1) PERMANOVA (3 groups and pairwise), permutations within date
    sep_rows = []
    pairs = [tuple(ex["groups"])] + list(itertools.combinations(ex["groups"], 2))
    for name, M in tm_cols.items():
        for pg in pairs:
            m = np.isin(labels_t, pg)
            r = permanova(M[m], labels_t[m], strata_t[m], ex["n_permutations"], rng)
            sep_rows.append({"features": name, "groups": " vs ".join(pg), "n_traces": int(m.sum()), **r})
    sep = pd.DataFrame(sep_rows)
    sep.to_csv(out / "permanova.csv", index=False)

    # 2) classification
    yw = U["group"].to_numpy()
    date_w = U["date"].astype(str).to_numpy()
    lodo = [(date_w != d, date_w == d) for d in np.unique(date_w)]
    loto = [(tid != t, tid == t) for t in np.unique(tid)]
    cls_rows, calls = [], []
    for name, F in feats.items():
        if F.std(0).max() < 1e-6:
            cls_rows.append({"features": name, "cv": "not applicable (feature constant across inputs)",
                             "window_acc": np.nan, "trace_acc": np.nan, "n_traces": 0})
            continue
        for cv_name, folds in (("leave-one-date-out", lodo), ("leave-one-trace-out", loto)):
            wa, ta, tdf = classify(F, yw, tid, folds)
            cls_rows.append({"features": name, "cv": cv_name, "window_acc": wa, "trace_acc": ta,
                             "n_traces": len(tdf)})
            tdf["features"], tdf["cv"] = name, cv_name
            calls.append(tdf)
    clsdf = pd.DataFrame(cls_rows)
    clsdf.to_csv(out / "classification.csv", index=False)
    pd.concat(calls).to_csv(out / "classification_trace_calls.csv", index=False)

    # 3) what do the codes track? Spearman over U2OS windows
    from scipy.stats import spearmanr

    code_cols = ["theta_H", "theta_L", "zPC1", "zPC2", "zPC3"]
    feat_cols = ["rel_amp_rms", "cos_amp_rel", "cos_period_h", "cos_r2", "t_start_h"]
    corr = pd.DataFrame(index=code_cols, columns=feat_cols, dtype=float)
    for a in code_cols:
        for b in feat_cols:
            corr.loc[a, b] = spearmanr(U[a], U[b])[0]
    corr.to_csv(out / "code_feature_spearman.csv")
    corr.loc[:, "log10_brightness"] = [spearmanr(U[a], U["log10_brightness"])[0] for a in code_cols]
    corr.to_csv(out / "code_feature_spearman.csv")

    # 3b) brightness vs codes within genotype: window-level per group, and trace-level after
    #     removing genotype means (pooled within-group correlation)
    bright_rows = []
    for a in code_cols:
        row = {"code": a}
        for g in ex["groups"]:
            m = (U["group"] == g).to_numpy()
            row["spearman_windows_%s" % g] = spearmanr(U[a][m], U["log10_brightness"][m])[0]
        tmc = U.groupby("trace_id")[[a, "log10_brightness", "group"]].agg(
            {a: "mean", "log10_brightness": "mean", "group": "first"})
        resid_c = tmc[a] - tmc.groupby("group")[a].transform("mean")
        resid_b = tmc["log10_brightness"] - tmc.groupby("group")["log10_brightness"].transform("mean")
        row["spearman_traces_within_group"] = spearmanr(resid_c, resid_b)[0]
        bright_rows.append(row)
    bright_df = pd.DataFrame(bright_rows)
    bright_df.to_csv(out / "brightness_vs_codes.csv", index=False)

    # 3c) phase leakage: within each trace, how much of a code's variation over window start times is explained
    #     by a sinusoid at the trace's own period (partial R^2 of cos/sin terms beyond intercept + linear drift)
    phase_rows = []
    for t in np.unique(tid):
        m = tid == t
        ts_h = U["t_start_h"].to_numpy()[m]
        rhythmic = U["cos_r2"].to_numpy()[m] > 0.5
        P = float(np.clip(np.median(U["cos_period_h"].to_numpy()[m][rhythmic]), 18, 30)) if rhythmic.any() else 24.0
        A0 = np.column_stack([np.ones(m.sum()), ts_h - ts_h.mean()])
        A1 = np.column_stack([A0, np.cos(2 * np.pi * ts_h / P), np.sin(2 * np.pi * ts_h / P)])
        row = {"trace_id": t, "group": U["group"].to_numpy()[m][0], "period_used_h": P}
        for a in code_cols:
            y = U[a].to_numpy()[m]
            r0 = y - A0 @ np.linalg.lstsq(A0, y, rcond=None)[0]
            c1 = np.linalg.lstsq(A1, y, rcond=None)[0]
            r1 = y - A1 @ c1
            row["phaseR2_" + a] = float(1 - np.sum(r1 ** 2) / max(np.sum(r0 ** 2), 1e-20))
            row["phaseAmp_" + a] = float(np.hypot(c1[2], c1[3]))
        phase_rows.append(row)
    phase_df = pd.DataFrame(phase_rows)
    phase_df.to_csv(out / "phase_leakage_by_trace.csv", index=False)
    # compare the size of the within-trace phase oscillation with the spread of genotype means
    phase_summary = {}
    for a in code_cols:
        gm = U.groupby("group")[a].mean()
        phase_summary[a] = {
            "median_phaseR2_by_group": phase_df.groupby("group")["phaseR2_" + a].median().round(3).to_dict(),
            "median_phase_amplitude": float(phase_df["phaseAmp_" + a].median()),
            "range_of_group_means": float(gm.max() - gm.min()),
        }

    theta_z_corr = np.array([[spearmanr(theta[iU][:, i], zU[:, j])[0] for j in range(zU.shape[1])]
                             for i in range(theta.shape[1])])

    # 4) generation sweeps on a representative WT window (closest to the WT median code, mid-recording)
    wt = np.where(U["group"].to_numpy() == "WT")[0]
    med = np.median(np.hstack([theta[iU][wt], zU[wt]]), 0)
    rep = wt[np.argmin(np.sum((np.hstack([theta[iU][wt], zU[wt]]) - med) ** 2, 1))]
    i_rep = iU[rep]
    fc0, z0 = theta[i_rep] * mc["FcLimit"], z[i_rep]
    sweep_rows, sweep_sig = [], {}
    base = to_rel(generate(parts, fc0[None], z0[None])[0], q)
    base_cos = cosinor_periodogram(base[None], dt_h).iloc[0]

    def describe(sig, kind, k, val):
        c = cosinor_periodogram(sig[None], dt_h).iloc[0]
        sweep_rows.append({"sweep": kind, "value": val, "rel_amp_rms": np.sqrt(2) * sig.std(),
                           "cos_period_h": c.cos_period_h, "cos_amp_rel": c.cos_amp_rel,
                           "shape_corr_with_base": float(np.corrcoef(sig, base)[0, 1])})
        sweep_sig[(kind, k)] = sig

    for k, th in enumerate(ex["theta_sweep"]):
        for j, nm in ((0, "theta_H"), (fc0.shape[0] - 1, "theta_L")):
            fc = fc0.copy()
            fc[j] = th * mc["FcLimit"]
            describe(to_rel(generate(parts, fc[None], z0[None])[0], q), nm, k, th)
    sd1 = pcs[:, 0].std()
    for k, s in enumerate([-2, -1, 0, 1, 2]):
        zz = z0 + s * sd1 * Vt[0]
        describe(to_rel(generate(parts, fc0[None], zz[None])[0], q), "zPC1", k, s)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(out / "generation_sweeps.csv", index=False)

    def sweep_summary(kind):
        s = sweep[sweep["sweep"] == kind]
        return {"amp_ratio_max_over_min": float(s.rel_amp_rms.max() / max(s.rel_amp_rms.min(), 1e-12)),
                "period_range_h": [float(s.cos_period_h.min()), float(s.cos_period_h.max())],
                "min_shape_corr": float(s.shape_corr_with_base.min())}

    # ---------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gU = U["group"].to_numpy()
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    a = ax[0, 0]
    for g, col in GROUP_COLORS.items():
        m = gU == g
        a.scatter(U["theta_L"][m], U["theta_H"][m], s=6, alpha=0.25, color=col)
        tm = tr_table[tr_table.group == g]
        for d, mk in zip(sorted(tr_table.date.astype(str).unique()), ["o", "s"]):
            mm = tm.date.astype(str) == d
            a.scatter(tm.theta_L[mm], tm.theta_H[mm], s=70, marker=mk, color=col, edgecolor="k",
                      label="%s (%s)" % (g, d))
    a.set_xlabel("theta_L (low-pass cut-off / FcLimit)")
    a.set_ylabel("theta_H (high-pass cut-off / FcLimit)")
    a.set_title("amplitude code theta (dots = windows, markers = trace means)", fontsize=9, loc="left")
    a.legend(fontsize=7, frameon=False, ncol=2)
    a = ax[0, 1]
    for g, col in GROUP_COLORS.items():
        m = gU == g
        a.scatter(U["zPC1"][m], U["zPC2"][m], s=6, alpha=0.25, color=col, label=g)
        for t in np.unique(tid[m]):
            mt = tid == t
            a.scatter(U["zPC1"][mt].mean(), U["zPC2"][mt].mean(), s=70, color=col, edgecolor="k")
    a.set_xlabel("z PC1 (%.0f%%)" % (100 * evr[0]))
    a.set_ylabel("z PC2 (%.0f%%)" % (100 * evr[1]))
    a.set_title("shape code z (PCA over U2OS windows)", fontsize=9, loc="left")
    a.legend(fontsize=7, frameon=False)
    a = ax[1, 0]
    for g, col in GROUP_COLORS.items():
        for t in np.unique(tid[gU == g]):
            mt = tid == t
            a.plot(U["t_start_h"][mt], U["theta_L"][mt], color=col, lw=1, alpha=0.8)
    a.set_xlabel("window start [h]")
    a.set_ylabel("theta_L")
    a.set_title("theta_L along the recording (one line per trace)", fontsize=9, loc="left")
    a = ax[1, 1]
    for g, col in GROUP_COLORS.items():
        m = gU == g
        a.scatter(U["cos_period_h"][m], U["cos_amp_rel"][m], s=6, alpha=0.3, color=col, label=g)
    a.set_xlabel("cosinor period [h]")
    a.set_ylabel("cosinor amplitude (relative)")
    a.set_yscale("log")
    a.set_title("classical features (baseline)", fontsize=9, loc="left")
    a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_codes.png", dpi=130)
    plt.close(fig)

    t_axis = np.arange(X01.shape[1]) * dt_h
    fig, ax = plt.subplots(3, 3, figsize=(13, 7.5), sharex=True)
    for r, g in enumerate(GROUP_COLORS):
        ids = np.where(gU == g)[0]
        pick = ids[np.argsort(U["t_start_h"].to_numpy()[ids])][[0, len(ids) // 2, -1]]
        for c, k in enumerate(pick):
            i = iU[k]
            ax[r, c].plot(t_axis, to_rel(X01[i], q), color="0.3", lw=1, label="input")
            ax[r, c].plot(t_axis, to_rel(gen[i], q), color=GROUP_COLORS[g], lw=1.3, label="VABAM (z, theta)")
            ax[r, c].set_title("%s  %s  t0=%.0f h  R2=%.2f" % (g, U["trace_id"].iloc[k], U["t_start_h"].iloc[k],
                                                               U["r2_gen"].iloc[k]), fontsize=7, loc="left")
    ax[0, 0].legend(fontsize=7, frameon=False)
    for a in ax[-1]:
        a.set_xlabel("time in window [h]")
    fig.tight_layout()
    fig.savefig(out / "fig_reconstruction.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    cmap = plt.get_cmap("viridis")
    for a, kind, vals in ((ax[0], "theta_L", ex["theta_sweep"]), (ax[1], "theta_H", ex["theta_sweep"]),
                          (ax[2], "zPC1", [-2, -1, 0, 1, 2])):
        for k, val in enumerate(vals):
            a.plot(t_axis, sweep_sig[(kind, k)], color=cmap(k / (len(vals) - 1)), lw=1.2, label=str(val))
        a.set_title("vary %s (others fixed)" % kind, fontsize=9, loc="left")
        a.legend(fontsize=7, frameon=False, title=kind if kind != "zPC1" else "z PC1 [SD]", title_fontsize=7)
        a.set_xlabel("time in window [h]")
    ax[0].set_ylabel("generated signal (relative units)")
    fig.tight_layout()
    fig.savefig(out / "fig_generation_sweeps.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- summary
    recon = meta.groupby("label")[["mse_gen", "r2_gen"]].median()
    split_u2os = U.groupby("split")["trace_id"].nunique().to_dict()
    summary = {
        "evaluation_type": ("transductive representation analysis: the encoder was trained without labels on "
                            "windows that include these U2OS traces (train/val split by trace: %s). Date-wise "
                            "classification here is NOT a test of generalisation to unseen experiments." % split_u2os),
        "counts": {"u2os_windows": int(len(U)), "u2os_traces": int(len(traces)),
                   "independent_experiment_dates": int(traces.date.nunique()),
                   "training_windows_total": int((meta["split"] == "train").sum()),
                   "training_traces_total": int(meta.loc[meta["split"] == "train", "trace_id"].nunique())},
        "theta_inferred_sd": theta_sd.tolist(),
        "theta_inferred_constant": theta_constant,
        "theta_note": ("Inferred theta (FC_Mu) is pulled to the Bernoulli(0.5) prior and theta is sampled ~U(0,1) "
                       "during training; it is a generation-time amplitude control, not an input descriptor. "
                       "Genotype tests on theta are therefore not applicable when it is constant; evaluate theta "
                       "by the generation sweeps instead." if theta_constant else ""),
        "model": args.model, "FcLimit": mc["FcLimit"], "n_u2os_traces": int(len(traces)),
        "n_u2os_windows": int(len(U)), "traces_per_group": traces.group.value_counts().to_dict(),
        "dates": sorted(traces.date.astype(str).unique().tolist()),
        "z_active_dims(var>0.01)": n_active, "z_active_ratio_top5": np.sort(active)[::-1][:5].round(3).tolist(),
        "z_PCA_explained_var_top3": evr[:3].round(3).tolist(),
        "reconstruction_median_by_label": {k: {kk: round(float(vv), 4) for kk, vv in v.items()}
                                           for k, v in recon.loc[[l for l in recon.index if "U2OS" in l]].to_dict("index").items()},
        "reconstruction_median_all": {"mse_gen": float(meta.mse_gen.median()), "r2_gen": float(meta.r2_gen.median())},
        "trace_means": {g: {"%s_%s" % (c, s): round(float(v), 4) for (c, s), v in row.items()}
                        for g, row in tr_table.groupby("group")[["theta_H", "theta_L", "cos_period_h",
                                                                  "cos_amp_rel", "cos_r2"]]
                        .agg(["mean", "std"]).iterrows()},
        "permanova": sep.to_dict("records"),
        "classification": clsdf.to_dict("records"),
        "code_feature_spearman": corr.round(3).to_dict(),
        "max_abs_spearman_theta_vs_z_dims": float(np.nanmax(np.abs(theta_z_corr))),
        "brightness_vs_codes": bright_df.round(3).to_dict("records"),
        "phase_leakage": phase_summary,
        "representative_window": {"trace_id": U["trace_id"].iloc[rep], "t_start_h": float(U["t_start_h"].iloc[rep]),
                                  "theta": theta[i_rep].round(4).tolist(),
                                  "base_period_h": float(base_cos.cos_period_h)},
        "sweeps": {k: sweep_summary(k) for k in ("theta_L", "theta_H", "zPC1")},
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.set_option("display.width", 200)
    print(sep.to_string(index=False))
    print(clsdf.to_string(index=False))
    print(corr.round(2).to_string())
    print(sweep.round(3).to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("z_active_dims(var>0.01)", "reconstruction_median_all",
                                              "max_abs_spearman_theta_vs_z_dims", "sweeps")}, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
