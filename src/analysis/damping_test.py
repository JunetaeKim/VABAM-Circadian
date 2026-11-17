"""
Damping test: does lowering theta_L reproduce the natural amplitude loss within a recording?

Within one LumiCycle recording the rhythm damps over days, mostly because cells drift out of synchrony; the
waveform keeps its period and phase while its amplitude shrinks. If theta is a biologically meaningful amplitude
control, then the early window's shape code z combined with a lower theta_L should resemble a later window of the
same recording.

Design (fixed before results are seen)
  Traces   : MSB recordings excluding U-2 OS (which="all": training and validation; "val": validation only).
  Early    : first window starting >= early_min_h (default 12 h; skips the transient right after synchronisation).
  Period P : cosinor period (16-36 h) of the early window.
  Late     : window whose start is closest to early start + cycles*P (default one cycle; same phase as early).
             Pairs whose start mismatch exceeds max_shift_h (default 2 h) are dropped.
  Known before the run (from the inputs alone): after one cycle the amplitude is about 0.55x (median), while the
  theta_L range of the s1 models spans about 0.8-1.3x, so theta* is expected to hit the lower bound often.
  Models of the late window, all built from the early window only (each fits at most one number):
    unchanged   early reconstruction (inferred z and theta)
    scaled      early reconstruction multiplied by the least-squares scalar (plain amplitude scaling)
    theta       generation with early z and theta_L on a grid (0.02-0.98), best theta_L by MSE
    ceiling     late window's own reconstruction (best the model can do)
  Outcomes per trace: MSE of each model vs the late input (all centred), observed amplitude ratio late/early,
  amplitude ratio implied by theta*, theta* relative to the inferred theta_L, theta* at the grid bound,
  coverage = (1 - ratio implied by theta*) / (1 - observed ratio) for damping traces,
  shape: best correlation with the late input over theta_L (scale-free) minus that of the unchanged reconstruction.
  Hypotheses: H1 late/early amplitude < 1; H2 theta* < inferred theta_L when H1 holds; H3 MSE(theta) < MSE(unchanged);
  H4 shape correlation improves when theta_L is lowered (theta of best correlation < inferred theta_L).
  Comparison without a directional prior: MSE(theta) vs MSE(scaled) (paired Wilcoxon).
Outputs: results/damping/<run_id>/per_trace.csv, summary.json, figures.
"""
import json

from src import vabam_bridge as vb  # before tensorflow

import numpy as np
import pandas as pd

DT_H = 1 / 6.0
GRID = np.round(np.arange(0.02, 0.981, 0.02), 2)


def cosinor_period(x, dt_h=DT_H, periods=np.arange(16.0, 36.01, 0.1)):
    t = np.arange(len(x)) * dt_h
    y = x - x.mean()
    best, bp = -1.0, np.nan
    for p in periods:
        C = np.c_[np.cos(2 * np.pi * t / p), np.sin(2 * np.pi * t / p)]
        b = np.linalg.lstsq(C, y, rcond=None)[0]
        r2 = 1 - np.sum((y - C @ b) ** 2) / max(np.sum(y ** 2), 1e-12)
        if r2 > best:
            best, bp = r2, p
    return float(bp), float(best)


def pick_pairs(meta, early_min_h=12.0, max_shift_h=2.0, xn=None, cycles=1):
    """(trace_id, early row index, late row index, period, k, shift) per trace; rows index into meta."""
    out = []
    for tid, g in meta.groupby("trace_id"):
        g = g.sort_values("t_start_h")
        e = g[g.t_start_h >= early_min_h]
        if e.empty:
            continue
        ie = int(e.index[0])
        t0 = float(meta.loc[ie, "t_start_h"])
        p, r2 = cosinor_period(xn[ie])
        best = None
        target = t0 + cycles * p
        d = (g.t_start_h - target).abs()
        j = int(d.idxmin())
        if d.loc[j] <= max_shift_h:
            best = (j, cycles, float(meta.loc[j, "t_start_h"] - target))
        if best:
            out.append(dict(trace_id=tid, i_early=ie, i_late=best[0], period_h=p, cos_r2=r2, k=best[1],
                            shift_h=best[2], t_early=t0, t_late=float(meta.loc[best[0], "t_start_h"])))
    return pd.DataFrame(out)


def evaluate_pair(xe, xl, rec_e, rec_l, gens, theta_l0):
    """All arrays centred (mean removed inside). gens: (len(GRID), L) generations from early z."""
    c = lambda v: v - v.mean()
    xe, xl, rec_e, rec_l = c(xe), c(xl), c(rec_e), c(rec_l)
    gens = gens - gens.mean(1, keepdims=True)
    mse = lambda a: float(np.mean((a - xl) ** 2))
    s = float(np.dot(rec_e, xl) / max(np.dot(rec_e, rec_e), 1e-12))
    m_theta = np.array([mse(g) for g in gens])
    k = int(np.argmin(m_theta))
    a0 = gens[int(np.argmin(np.abs(GRID - theta_l0)))].std()
    corrs = np.array([np.corrcoef(g, xl)[0, 1] for g in gens])
    kc = int(np.argmax(corrs))
    obs = float(xl.std() / max(xe.std(), 1e-12))
    imp = float(gens[k].std() / max(a0, 1e-12))
    return dict(coverage=float((1 - imp) / (1 - obs)) if obs < 1 else float("nan"),
                theta_star_corr=float(GRID[kc]), corr_theta_best=float(corrs[kc]),
                delta_corr=float(corrs[kc] - np.corrcoef(rec_e, xl)[0, 1]),mse_unchanged=mse(rec_e), mse_scaled=mse(s * rec_e), mse_theta=float(m_theta[k]),
                mse_ceiling=mse(rec_l), scale=s, theta_star=float(GRID[k]), theta_l0=float(theta_l0),
                theta_at_bound=bool(k in (0, len(GRID) - 1)),
                amp_ratio_obs=obs, amp_ratio_theta=imp,
                corr_unchanged=float(np.corrcoef(rec_e, xl)[0, 1]), corr_theta=float(np.corrcoef(gens[k], xl)[0, 1]),
                best_idx=k)


def run(run, which="all", early_min_h=12.0, max_shift_h=2.0, cycles=1):
    from src.model import sweep as sw
    from src.analysis.exp01_u2os_cry import encode_generate
    from src.model.sweep_inspect import _model, theta_sweep
    from src.model.train import load_windows

    stage, rid = run.split("/")
    cfg, scfg, mc, ckpt, _ = _model(stage, rid)
    X01, Xn, meta = load_windows(cfg, mc["DataSource"])
    keep = (meta["cell_line"] != "U2OS") & ((meta["split"] == "val") if which == "val" else True)
    idx = np.where(keep.to_numpy())[0]
    X01, Xn, meta = X01[idx], Xn[idx], meta.iloc[idx].reset_index(drop=True)
    pairs = pick_pairs(meta, early_min_h, max_shift_h, xn=Xn, cycles=cycles)
    rows = np.unique(np.r_[pairs.i_early.to_numpy(), pairs.i_late.to_numpy()])
    parts, lv = sw._load_parts(cfg, mc, X01.shape[1], ckpt)
    theta, z, _, gen, _ = encode_generate(parts, lv, X01[rows], mc["FcLimit"])
    pos = {r: n for n, r in enumerate(rows)}
    ie = np.array([pos[i] for i in pairs.i_early])
    sweep = theta_sweep(parts, theta[ie], z[ie], mc["FcLimit"], GRID)        # (len(GRID), n_pairs, L), centred
    res, ex = [], {}
    for n, p in pairs.reset_index(drop=True).iterrows():
        a, b = pos[p.i_early], pos[p.i_late]
        r = evaluate_pair(X01[p.i_early] - 0.5, X01[p.i_late] - 0.5, gen[a] - 0.5, gen[b] - 0.5, sweep[:, n],
                          theta[a, -1])
        ex[p.trace_id] = dict(xe=X01[p.i_early] - 0.5, xl=X01[p.i_late] - 0.5, rec_e=gen[a] - 0.5,
                              rec_l=gen[b] - 0.5, best=sweep[r["best_idx"], n], scale=r["scale"])
        res.append({**p.to_dict(), "cell_line": meta.loc[p.i_early, "cell_line"],
                    "reporter": meta.loc[p.i_early, "reporter"], "split": meta.loc[p.i_early, "split"], **r})
    df = pd.DataFrame(res)
    return df, ex


def summarise(df):
    from scipy.stats import wilcoxon

    def w(a, b):
        d = df[a] - df[b]
        return float(wilcoxon(d).pvalue) if (d != 0).any() and len(d) >= 5 else float("nan")

    damp = df.amp_ratio_obs < 1
    s = dict(n_traces=int(len(df)), n_val=int((df.split == "val").sum()),
             H1_frac_damping=float(damp.mean()), median_amp_ratio_obs=float(df.amp_ratio_obs.median()),
             H2_frac_theta_lower_given_damping=float((df.theta_star[damp] < df.theta_l0[damp]).mean()) if damp.any() else None,
             frac_theta_at_bound=float(df.theta_at_bound.mean()),
             H3_frac_theta_better_than_unchanged=float((df.mse_theta < df.mse_unchanged).mean()),
             H3_p_wilcoxon=w("mse_theta", "mse_unchanged"),
             frac_theta_better_than_scaled=float((df.mse_theta < df.mse_scaled).mean()),
             p_theta_vs_scaled=w("mse_theta", "mse_scaled"),
             median_gap_closed_theta=float(np.median((df.mse_unchanged - df.mse_theta) /
                                                     np.maximum(df.mse_unchanged - df.mse_ceiling, 1e-12))),
             median_gap_closed_scaled=float(np.median((df.mse_unchanged - df.mse_scaled) /
                                                      np.maximum(df.mse_unchanged - df.mse_ceiling, 1e-12))),
             median_coverage=float(df.coverage[damp].median()) if damp.any() else None,
             H4_frac_corr_best_below_inferred=float((df.theta_star_corr < df.theta_l0).mean()),
             median_delta_corr=float(df.delta_corr.median()),
             spearman_amp_obs_vs_theta=float(df[["amp_ratio_obs", "amp_ratio_theta"]].corr("spearman").iloc[0, 1]))
    return s


def save(run, df, summary):
    out = vb.assert_inside_spinoff(vb.SPINOFF_ROOT / "results" / "damping" / run.replace("/", "__"))
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["best_idx"]).to_csv(out / "per_trace.csv", index=False)
    json.dump({"run": run, **summary}, open(out / "summary.json", "w", encoding="utf-8"), indent=1)
    return out


def plot_examples(df, ex, trace_ids=None, n=4):
    import matplotlib.pyplot as plt

    if trace_ids is None:
        d = df[df.amp_ratio_obs < 1].sort_values("amp_ratio_obs")
        trace_ids = d.trace_id.iloc[np.linspace(0, len(d) - 1, min(n, len(d))).astype(int)].tolist() if len(d) else []
    fig, axes = plt.subplots(1, len(trace_ids), figsize=(4.3 * len(trace_ids), 3.2), squeeze=False)
    for a, tid in zip(axes[0], trace_ids):
        e, r = ex[tid], df.set_index("trace_id").loc[tid]
        t = np.arange(len(e["xl"])) * DT_H
        c = lambda v: v - v.mean()
        a.plot(t, c(e["xe"]), color="#c7c7c7", lw=1.4, label="early input")
        a.plot(t, c(e["xl"]), color="#6b7280", lw=1.4, label="late input")
        a.plot(t, c(e["rec_e"]), color="#2563eb", lw=1.1, ls=":", label="early recon")
        a.plot(t, c(e["best"]), color="#dc2626", lw=1.6, label="early z, θ$_L$*=%.2f" % r.theta_star)
        a.set_title("%s %s\nlate/early amp %.2f, k=%d" % (r.cell_line, r.reporter, r.amp_ratio_obs, r.k), fontsize=9)
        a.set_xlabel("hours in window", fontsize=8)
        a.tick_params(labelsize=7)
    if len(trace_ids):
        axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def plot_summary(df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    a = ax[0]
    a.scatter(df.amp_ratio_obs, df.amp_ratio_theta, c=np.where(df.theta_at_bound, "#dc2626", "#2563eb"), s=22)
    lim = [0, max(1.6, df[["amp_ratio_obs", "amp_ratio_theta"]].max().max() * 1.05)]
    a.plot(lim, lim, color="#9ca3af", lw=0.8, ls=":")
    a.set_xlabel("observed amplitude ratio (late / early)")
    a.set_ylabel("amplitude ratio implied by θ$_L$*")
    a.set_title("red: θ$_L$* at grid bound", fontsize=9)
    a = ax[1]
    a.scatter(df.theta_l0, df.theta_star, c=np.where(df.amp_ratio_obs < 1, "#2563eb", "#f59e0b"), s=22)
    a.plot([0, 1], [0, 1], color="#9ca3af", lw=0.8, ls=":")
    a.set_xlabel("inferred θ$_L$ (early window)")
    a.set_ylabel("best θ$_L$* for the late window")
    a.set_title("blue: damping, orange: amplitude grew", fontsize=9)
    a = ax[2]
    rel = pd.DataFrame({k: df["mse_" + k] / df.mse_unchanged for k in ["scaled", "theta", "ceiling"]})
    a.boxplot([rel[k] for k in rel], showfliers=True)
    a.set_xticks([1, 2, 3], ["scaled", "θ", "ceiling"])
    a.axhline(1, color="#9ca3af", lw=0.8, ls=":")
    a.set_ylabel("MSE / MSE(unchanged)")
    a.set_title("late window error relative to unchanged early recon", fontsize=9)
    for a in ax:
        a.tick_params(labelsize=8)
    fig.tight_layout()
    return fig
