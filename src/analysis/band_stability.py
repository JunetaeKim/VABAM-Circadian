"""
Reference-band clock stability: is the next cycle inside the amplitude-variant band of the previous one?

Adapted from the monitoring concept of the VABAM paper (reference bands built from a single preceding observation
by generating amplitude variants; conformity = stable, isolated departures = noise, sustained departures = true
disturbance). Here the "next beat" is the same recording one circadian period later.

Design (fixed before results are seen)
  Traces    : MSB recordings excluding U-2 OS (which="all": training and validation; "val": validation only).
  Period P  : cosinor period (16-36 h) of the first window starting >= early_min_h (default 12 h); one P per trace.
  Pairs     : every reference window starting >= early_min_h that has a target window starting within
              max_shift_h (default 1.5 h, half the window stride) of reference start + P. The known residual
              offset is removed by shifting the comparison by round(offset / dt) samples (overlap only).
  Band      : reference window -> z, theta. Members = generations with the same z and theta_L on a grid
              (0.02-0.98); band = pointwise 5-95 % range of the members, widened by Z_TOL * sigma, where sigma is
              the SD of the reference window's own reconstruction residual (the model's noise level there).
              band="theta_z" additionally draws z from the encoder posterior for every member.
  Per pair  : conf_theta  fraction of target samples inside the band
              conf_shape  fraction of target samples within Z_TOL * sigma of scale * reference reconstruction
                          (scale = least-squares scalar; amplitude is free, shape and phase are not)
              state       no_rhythm  reconstruction SD < snr_min * sigma (nothing to follow)
                          conform    conf_theta >= tau
                          amplitude  conf_shape >= tau and scale >= scale_min (amplitude left the theta range,
                                     waveform kept)
                          break      otherwise (shape or phase changed, or rhythm lost)
  Per trace : stable_frac (primary) = fraction of pairs in state conform or amplitude
              conformity_theta, conformity_shape = means of conf_theta, conf_shape
                       (a flat, noisy trace sits inside its own noise band, so these two are high for
                       arrhythmic traces too; read them together with frac_no_rhythm)
              hold_h = hours from the first reference start to the target start of the first run of run_len
                       (default 2) consecutive unstable pairs (break or no_rhythm); censored if none
              calib_self = fraction of the reference input inside its own band (calibration check, expect ~0.9+)
  Comparators (same pairs, no model): acf = correlation of reference input and target input; log_amp = log
              median SD of the trace's windows (normalised units); also self_sim = correlation of the reference
              reconstruction and the target input.
  Evaluation: published circadian phenotype (Functional / Unstable / Weak / Dysfunctional) at cell-line level
              (median over traces): Kruskal-Wallis, AUC Functional vs rest, and partial Spearman with the
              Functional indicator controlling log_amp. The band score must be compared with acf and log_amp.
  Caveats   : pairs of one trace overlap in time and are not independent; n = 15 cell lines.
Outputs: results/band_stability/<run_id>/per_pair.csv, per_trace.csv, per_line.csv, compare.csv, summary.json.
"""
import json

from src import vabam_bridge as vb  # before tensorflow

import numpy as np
import pandas as pd

from src.analysis.damping_test import DT_H, GRID, cosinor_period

Z_TOL = 1.645
STATES = ["conform", "amplitude", "break", "no_rhythm"]
SCORES = ["stable_frac", "conformity_shape", "conformity_theta", "hold_h", "self_sim", "acf", "log_amp"]


def pick_chain(meta, xn, early_min_h=12.0, max_shift_h=1.5, cycles=1):
    """All (reference, target) window pairs per trace; rows index into meta."""
    out = []
    for tid, g in meta.groupby("trace_id"):
        g = g.sort_values("t_start_h")
        e = g[g.t_start_h >= early_min_h]
        if e.empty:
            continue
        p, r2 = cosinor_period(xn[int(e.index[0])])
        for ir, t0 in zip(e.index, e.t_start_h):
            d = g.t_start_h - (t0 + cycles * p)
            j = d.abs().idxmin()
            if abs(d.loc[j]) <= max_shift_h:
                out.append(dict(trace_id=tid, i_ref=int(ir), i_tgt=int(j), period_h=p, cos_r2=r2,
                                t_ref=float(t0), t_tgt=float(g.t_start_h.loc[j]), offset_h=float(d.loc[j])))
    return pd.DataFrame(out)


def _overlap(n, s):
    """Index slices (target, reference) so that target sample k is compared with reference sample k + s."""
    return (slice(0, n - s), slice(s, n)) if s >= 0 else (slice(-s, n), slice(0, n + s))


def evaluate_pair(x_ref, x_tgt, rec, members, offset_h, tau=0.8, scale_min=0.25, snr_min=1.0):
    """x_ref, x_tgt, rec: (L,); members: (n_theta, L) ordered like GRID. Everything is centred inside."""
    c = lambda v: v - v.mean(-1, keepdims=True)
    x_ref, x_tgt, rec, members = c(x_ref), c(x_tgt), c(rec), c(members)
    sig = float((x_ref - rec).std())
    tol = Z_TOL * sig
    lo, hi = np.quantile(members, [0.05, 0.95], axis=0)
    calib = float(np.mean((x_ref >= lo - tol) & (x_ref <= hi + tol)))
    st, sr = _overlap(len(x_tgt), int(round(offset_h / DT_H)))
    xt, xr, rc, mem, lo, hi = x_tgt[st], x_ref[sr], rec[sr], members[:, sr], lo[sr], hi[sr]
    conf_theta = float(np.mean((xt >= lo - tol) & (xt <= hi + tol)))
    scale = float(np.dot(rc, xt) / max(np.dot(rc, rc), 1e-12))
    conf_shape = float(np.mean(np.abs(xt - scale * rc) <= tol))
    snr = float(rec.std() / max(sig, 1e-12))
    if snr < snr_min:
        state = "no_rhythm"
    elif conf_theta >= tau:
        state = "conform"
    elif conf_shape >= tau and scale >= scale_min:
        state = "amplitude"
    else:
        state = "break"
    k = int(np.argmin(np.mean((mem - xt) ** 2, axis=1)))
    cc = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return dict(conf_theta=conf_theta, conf_shape=conf_shape, state=state, scale=scale, snr=snr, sigma=sig,
                calib_self=calib, theta_star=float(GRID[k]), theta_at_bound=bool(k in (0, len(GRID) - 1)),
                amp_ratio_obs=float(xt.std() / max(xr.std(), 1e-12)), acf=cc(xr, xt), self_sim=cc(rc, xt),
                band_width=float(np.mean(hi - lo)))


def _members(parts, theta, z, z_logvar, fc_limit, band, seed):
    """(len(GRID), n, L) generations (centred at 0.5 removed) for every reference window."""
    from src.analysis.exp01_u2os_cry import generate

    rng = np.random.default_rng(seed)
    out = []
    for th in GRID:
        fc = theta * fc_limit
        fc[:, -1] = th * fc_limit
        zz = z + np.exp(0.5 * z_logvar) * rng.standard_normal(z.shape).astype(z.dtype) if band == "theta_z" else z
        out.append(generate(parts, fc, zz) - 0.5)
    return np.array(out)


def run(run, which="all", band="theta", early_min_h=12.0, max_shift_h=1.5, cycles=1, tau=0.8, scale_min=0.25,
        snr_min=1.0, seed=0):
    from src.model import sweep as sw
    from src.analysis.exp01_u2os_cry import encode_generate
    from src.model.sweep_inspect import _model
    from src.model.train import load_windows

    stage, rid = run.split("/")
    cfg, scfg, mc, ckpt, _ = _model(stage, rid)
    X01, Xn, meta = load_windows(cfg, mc["DataSource"])
    keep = (meta["cell_line"] != "U2OS") & ((meta["split"] == "val") if which == "val" else True)
    if "source" in meta:
        keep &= meta["source"] == "msb2025"
    idx = np.where(keep.to_numpy())[0]
    X01, Xn, meta = X01[idx], Xn[idx], meta.iloc[idx].reset_index(drop=True)
    pairs = pick_chain(meta, Xn, early_min_h, max_shift_h, cycles)
    refs = np.unique(pairs.i_ref.to_numpy())
    parts, lv = sw._load_parts(cfg, mc, X01.shape[1], ckpt)
    theta, z, z_logvar, gen, _ = encode_generate(parts, lv, X01[refs], mc["FcLimit"])
    mem = _members(parts, theta, z, z_logvar, mc["FcLimit"], band, seed)
    pos = {r: n for n, r in enumerate(refs)}
    res, ex = [], {}
    for p in pairs.itertuples(index=False):
        a = pos[p.i_ref]
        r = evaluate_pair(X01[p.i_ref] - 0.5, X01[p.i_tgt] - 0.5, gen[a] - 0.5, mem[:, a], p.offset_h, tau,
                          scale_min, snr_min)
        m = meta.loc[p.i_ref]
        res.append({**p._asdict(), "cell_line": m["cell_line"], "reporter": m["reporter"], "split": m["split"],
                    "theta_l0": float(theta[a, -1]), **r})
        ex[(p.trace_id, p.t_ref)] = dict(x_ref=X01[p.i_ref] - 0.5, x_tgt=X01[p.i_tgt] - 0.5, rec=gen[a] - 0.5,
                                         members=mem[:, a], offset_h=p.offset_h)
    amp = pd.Series(Xn.std(axis=1)).groupby(meta["trace_id"].to_numpy()).median()
    return pd.DataFrame(res), ex, amp


def per_trace(pp, amp=None, run_len=2):
    from src.analysis.exp02_analyze import PHENO

    rows = []
    for tid, g in pp.groupby("trace_id"):
        g = g.sort_values("t_ref")
        bad = g.state.isin(["break", "no_rhythm"]).to_numpy()
        runs = np.convolve(bad.astype(int), np.ones(run_len, dtype=int), "valid") == run_len if len(bad) >= run_len else []
        first = int(np.argmax(runs)) if np.any(runs) else None
        span = float(g.t_tgt.iloc[-1] - g.t_ref.iloc[0])
        r = dict(trace_id=tid, cell_line=g.cell_line.iloc[0], reporter=g.reporter.iloc[0], split=g.split.iloc[0],
                 phenotype=PHENO.get(g.cell_line.iloc[0], "NA"), n_pairs=len(g), period_h=g.period_h.iloc[0],
                 stable_frac=float((~bad).mean()), conformity_theta=float(g.conf_theta.mean()),
                 conformity_shape=float(g.conf_shape.mean()),
                 hold_h=float(g.t_tgt.iloc[first] - g.t_ref.iloc[0]) if first is not None else span,
                 hold_censored=first is None, self_sim=float(g.self_sim.mean()), acf=float(g.acf.mean()),
                 amp_ratio=float(g.amp_ratio_obs.median()), theta_star=float(g.theta_star.median()),
                 theta_bound_frac=float(g.theta_at_bound.mean()), calib_self=float(g.calib_self.mean()),
                 snr=float(g.snr.median()))
        r.update({"frac_" + s: float((g.state == s).mean()) for s in STATES})
        r["log_amp"] = float(np.log(amp[tid])) if amp is not None and tid in amp.index else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def per_line(pt):
    num = [c for c in pt.columns if pt[c].dtype.kind in "fb" and c != "hold_censored"]
    g = pt.groupby(["cell_line", "phenotype"])
    return g[num].median().join(g.size().rename("n_traces")).reset_index()


def _auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def _partial_spearman(x, y, c):
    r = lambda v: pd.Series(v).rank().to_numpy()
    x, y, c = r(x), r(y), r(c)
    res = lambda a: a - np.polyval(np.polyfit(c, a, 1), c)
    return float(np.corrcoef(res(x), res(y))[0, 1])


def compare(pl, scores=SCORES):
    """Cell-line level: how well does each score follow the published phenotype? pl from per_line()."""
    from scipy.stats import kruskal

    d = pl[pl.phenotype != "NA"]
    f = (d.phenotype == "Functional").to_numpy()
    rows = []
    for s in scores:
        v = d[s].to_numpy()
        ok = np.isfinite(v)
        groups = [v[ok & (d.phenotype == p).to_numpy()] for p in d.phenotype.unique()]
        kw = kruskal(*[g for g in groups if len(g)]) if np.ptp(v[ok]) > 0 else None
        rows.append(dict(score=s, n_lines=int(ok.sum()), kruskal_H=float(kw.statistic) if kw else np.nan,
                         kruskal_p=float(kw.pvalue) if kw else np.nan, auc_functional=_auc(v[ok & f], v[ok & ~f]),
                         partial_rho_given_amp=_partial_spearman(v[ok], f[ok].astype(float), d.log_amp.to_numpy()[ok])
                         if s != "log_amp" and np.isfinite(d.log_amp).all() else np.nan,
                         **{"median_" + p: float(np.median(v[ok & (d.phenotype == p).to_numpy()]))
                            for p in ["Functional", "Unstable", "Weak", "Dysfunctional"]
                            if (ok & (d.phenotype == p).to_numpy()).any()}))
    return pd.DataFrame(rows)


def summarise(pp, pt, cmp):
    return dict(n_pairs=int(len(pp)), n_traces=int(len(pt)), n_lines=int(pt.cell_line.nunique()),
                state_fractions={s: float((pp.state == s).mean()) for s in STATES},
                median_calib_self=float(pt.calib_self.median()),
                median_band_width_over_sigma=float((pp.band_width / pp.sigma.clip(lower=1e-12)).median()),
                theta_bound_frac=float(pp.theta_at_bound.mean()),
                frac_hold_censored=float(pt.hold_censored.mean()),
                auc_functional={r.score: r.auc_functional for r in cmp.itertuples()},
                kruskal_p={r.score: r.kruskal_p for r in cmp.itertuples()})


def save(run, pp, pt, pl, cmp, summary, band="theta"):
    out = vb.assert_inside_spinoff(vb.SPINOFF_ROOT / "results" / "band_stability" /
                                   (run.replace("/", "__") + ("" if band == "theta" else "__" + band)))
    out.mkdir(parents=True, exist_ok=True)
    for name, d in (("per_pair", pp), ("per_trace", pt), ("per_line", pl), ("compare", cmp)):
        d.to_csv(out / (name + ".csv"), index=False)
    json.dump({"run": run, "band": band, **summary}, open(out / "summary.json", "w", encoding="utf-8"), indent=1)
    return out


COL = {"conform": "#2563eb", "amplitude": "#f59e0b", "break": "#dc2626", "no_rhythm": "#9ca3af"}
PCOL = {"Functional": "#2563eb", "Unstable": "#f59e0b", "Weak": "#9ca3af", "Dysfunctional": "#dc2626", "NA": "#d1d5db"}


def plot_examples(pp, ex, keys=None):
    """One pair per phenotype by default (the trace with the median stable_frac, its first pair)."""
    import matplotlib.pyplot as plt

    if keys is None:
        pt = per_trace(pp)
        keys = []
        for ph in ["Functional", "Unstable", "Weak", "Dysfunctional"]:
            d = pt[pt.phenotype == ph].sort_values("stable_frac")
            if len(d):
                tid = d.trace_id.iloc[len(d) // 2]
                keys.append((tid, float(pp[pp.trace_id == tid].t_ref.min())))
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.3), squeeze=False)
    for a, key in zip(axes[0], keys):
        e = ex[key]
        r = pp[(pp.trace_id == key[0]) & (pp.t_ref == key[1])].iloc[0]
        c = lambda v: v - v.mean(-1, keepdims=True)
        mem, rec, xt = c(e["members"]), c(e["rec"]), c(e["x_tgt"])
        lo, hi = np.quantile(mem, [0.05, 0.95], axis=0)
        tol = Z_TOL * r.sigma
        st, sr = _overlap(len(xt), int(round(e["offset_h"] / DT_H)))
        t = np.arange(len(xt))[st] * DT_H
        a.fill_between(t, (lo - tol)[sr], (hi + tol)[sr], color="#bfdbfe", lw=0, label="reference band")
        a.plot(t, rec[sr], color="#2563eb", lw=1.0, ls=":", label="reference recon")
        a.plot(t, xt[st], color=COL[r.state], lw=1.5, label="one cycle later")
        a.set_title("%s %s (%s)\n%s, in band %.2f, shape %.2f" % (r.cell_line, r.reporter, r.state,
                    key[0][:8], r.conf_theta, r.conf_shape), fontsize=9)
        a.set_xlabel("hours in window", fontsize=8)
        a.tick_params(labelsize=7)
        a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def plot_timeline(pp):
    """State of every pair, traces grouped by phenotype and cell line."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from src.analysis.exp02_analyze import PHENO

    d = pp.assign(phenotype=pp.cell_line.map(PHENO).fillna("NA"))
    order = {p: n for n, p in enumerate(["Functional", "Unstable", "Weak", "Dysfunctional", "NA"])}
    tr = d.drop_duplicates("trace_id").assign(o=lambda x: x.phenotype.map(order)).sort_values(["o", "cell_line", "trace_id"])
    ypos = {t: n for n, t in enumerate(tr.trace_id)}
    fig, a = plt.subplots(figsize=(8, max(3, 0.09 * len(tr))))
    a.scatter(d.t_tgt, d.trace_id.map(ypos), c=d.state.map(COL), s=14, marker="s", lw=0)
    for ln, g in tr.groupby("cell_line", sort=False):
        y = [ypos[t] for t in g.trace_id]
        a.text(d.t_tgt.max() + 1.5, np.mean(y), "%s (%s)" % (ln, g.phenotype.iloc[0]), fontsize=6, va="center",
               color=PCOL[g.phenotype.iloc[0]])
        a.axhline(max(y) + 0.5, color="#e5e7eb", lw=0.5)
    a.set_yticks([])
    a.invert_yaxis()
    a.set_xlabel("start of the compared window (h)")
    a.legend(handles=[Patch(color=COL[s], label=s) for s in STATES], fontsize=7, frameon=False, ncol=4,
             loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    return fig


def plot_scores(pl, scores=("stable_frac", "conformity_shape", "self_sim", "acf", "log_amp")):
    """Cell-line medians by published phenotype."""
    import matplotlib.pyplot as plt

    ph = [p for p in ["Functional", "Unstable", "Weak", "Dysfunctional"] if (pl.phenotype == p).any()]
    fig, axes = plt.subplots(1, len(scores), figsize=(3.1 * len(scores), 3.3), squeeze=False)
    rng = np.random.default_rng(0)
    for a, s in zip(axes[0], scores):
        for n, p in enumerate(ph):
            v = pl.loc[pl.phenotype == p, s].to_numpy()
            a.scatter(n + rng.uniform(-0.12, 0.12, len(v)), v, color=PCOL[p], s=30)
            a.plot([n - 0.25, n + 0.25], [np.median(v)] * 2, color="#111827", lw=1.2)
        a.set_xticks(range(len(ph)), ph, fontsize=7, rotation=20)
        a.set_title(s, fontsize=9)
        a.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    return fig
