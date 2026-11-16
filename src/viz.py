"""
Plots for the notebooks (no TensorFlow). Every function returns the matplotlib Figure and, when save=True,
writes a PNG under results/figures/<subdir>/.
"""
import json

import numpy as np
import pandas as pd

from . import vabam_bridge as vb
from . import sweep as sw


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _save(fig, subdir, name, save):
    if save:
        out = vb.assert_inside_spinoff(vb.SPINOFF_ROOT / "results" / "figures" / subdir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / name, dpi=130, bbox_inches="tight")
    return fig


def live_curves(stage, metric="val_OrigRecLoss", runs=None, logy=True, save=True):
    """Validation curves of every run in a stage, read live from RelLossWeight logs (finished or running)."""
    plt = _plt()
    _, scfg = sw.load_cfgs()
    sd = sw.stage_dir(scfg, stage)
    plan = sw.load_plan(stage, scfg)
    ids = [r["run_id"] for r in plan if runs is None or r["run_id"] in runs]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    n_drawn = 0
    for rid in ids:
        log = sw._parse_log(sd / "Logs" / rid / "Logs_model.txt")
        if len(log) and metric in log:
            ax.plot(log["epoch"], log[metric], lw=1.2, marker="." if len(log) < 3 else None,
                    label=rid.replace(stage + "_", ""))
            n_drawn += 1
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric + " (validation, unit loss)")
    if logy:
        ax.set_yscale("log")
    ax.set_title("%s — %s" % (stage, metric), loc="left", fontsize=10)
    if n_drawn:
        ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    return _save(fig, "sweep", "%s_curves_%s.png" % (stage, metric), save)


def factor_effects(df, factors=None, metrics=("recon_r2", "control_pass", "amp_ratio", "shape_keep"), save=True,
                   name="factor_effects"):
    """Mean of each metric per level of each factor (main effects) for finished runs."""
    plt = _plt()
    factors = [f for f in (factors or sw.factors(df)) if f in df]
    if not factors or df.empty:
        print("no swept factor with more than one finished level yet")
        return None
    df = df.assign(**{f: df[f].astype(str) for f in factors})
    fig, axes = plt.subplots(len(metrics), len(factors), figsize=(3.2 * len(factors), 2.4 * len(metrics)),
                             squeeze=False)
    for i, m in enumerate(metrics):
        for j, f in enumerate(factors):
            ax = axes[i, j]
            g = df.groupby(f)[m]
            mu, sd = g.mean(), g.std()
            ax.bar([str(k) for k in mu.index], mu.values, yerr=sd.reindex(mu.index).fillna(0).values,
                   color="C%d" % j, alpha=0.8, capsize=3)
            for k, (lvl, grp) in enumerate(df.groupby(f)):
                ax.scatter([k] * len(grp), grp[m], color="k", s=8, zorder=3)
            if i == 0:
                ax.set_title(f, fontsize=9)
            if j == 0:
                ax.set_ylabel(m, fontsize=9)
            ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _save(fig, "sweep", "%s.png" % name, save)


def ranking_bars(ranked, top=15, save=True, name="ranking"):
    """Criteria of the top configurations (after rank-sum)."""
    plt = _plt()
    _, scfg = sw.load_cfgs()
    crit = scfg["selection"]["criteria"]
    if ranked is None or ranked.empty:
        print("no finished runs yet")
        return None
    r = ranked.head(top)
    fig, axes = plt.subplots(1, len(crit), figsize=(3.0 * len(crit), 0.35 * len(r) + 1.5), sharey=True)
    for ax, c in zip(axes, crit):
        err = r[c + "_sd"].fillna(0).values if c + "_sd" in r else None
        ax.barh(range(len(r)), r[c].values, xerr=err, color="C0", alpha=0.8)
        ax.set_title(c, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_yticks(range(len(r)))
    axes[0].set_yticklabels(r["config_name"], fontsize=8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return _save(fig, "sweep", "%s.png" % name, save)


def eta_table(stage):
    """Estimated remaining time per running/pending run from the speed of finished/running ones."""
    st = sw.status(stage)
    _, scfg = sw.load_cfgs()
    sd = sw.stage_dir(scfg, stage)
    spe = []
    for rid in st.loc[st.state == "done", "run_id"]:
        rj = sd / "runs" / rid / "run.json"
        if rj.exists():
            spe.append(json.load(open(rj, encoding="utf-8")).get("sec_per_epoch", np.nan))
    s = float(np.nanmedian(spe)) if spe else np.nan
    st["sec_per_epoch_est"] = s
    st["remaining_min_upper"] = (st["max_epochs"] - st["epochs"]).clip(lower=0) * s / 60.0
    return st


# ------------------------------------------------------------------ stage analysis (s1 and later)
_FC_LABEL = {0.0208333: "cut-off 8 h", 0.0138889: "cut-off 12 h"}


def _run_factors(stage):
    """run_id -> dict(data, FcLimit, LatDim, label) from the frozen plan and params."""
    cfg, scfg = sw.load_cfgs()
    out = {}
    for r in sw.load_plan(stage, scfg):
        mc = sw.run_model_config(cfg, scfg, r)
        data = "MULTI" if "circascope2021" in mc.get("SourceWeights", {}) else "POP"
        out[r["run_id"]] = {"data": data, "FcLimit": mc["FcLimit"], "LatDim": mc["LatDim"],
                            "label": r["config_name"]}
    return out


def curves_by_factor(stage, metric="val_OrigRecLoss", save=True):
    """History curves: colour = cut-off, line style = data mix, line width = latent size."""
    plt = _plt()
    _, scfg = sw.load_cfgs()
    fac = _run_factors(stage)
    colors = {0.0208333: "#c2410c", 0.0138889: "#1d4ed8"}
    styles = {"POP": "-", "MULTI": "--"}
    widths = {10: 0.8, 30: 1.4, 50: 2.2}
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for rid, f in fac.items():
        p = sw.run_dir(scfg, stage, rid) / "history.csv"
        if not p.exists():
            continue
        h = pd.read_csv(p)
        y = h[metric].cummin()
        ax.plot(np.arange(1, len(y) + 1), y, color=colors.get(f["FcLimit"], "k"), ls=styles[f["data"]],
                lw=widths.get(f["LatDim"], 1.2), alpha=0.9)
    from matplotlib.lines import Line2D
    handles = ([Line2D([], [], color=c, label=_FC_LABEL.get(k, k)) for k, c in colors.items()] +
               [Line2D([], [], color="k", ls=s, label=d) for d, s in styles.items()] +
               [Line2D([], [], color="grey", lw=w, label="Z%d" % z) for z, w in widths.items()])
    ax.legend(handles=handles, fontsize=8, frameon=False, ncol=3)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("best %s so far (MSB validation)" % metric)
    ax.set_title("%s learning curves" % stage, loc="left", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    return _save(fig, "sweep", "%s_curves_by_factor_%s.png" % (stage, metric), save)


def tradeoff(df, stage="s1", save=True):
    """Left: reconstruction vs control pass. Right: amplitude ratio vs shape keep with the pass thresholds."""
    plt = _plt()
    _, scfg = sw.load_cfgs()
    cp = scfg["selection"]["control_pass"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = {"FC8": "#c2410c", "FC12": "#1d4ed8"}
    for _, r in df.iterrows():
        fc = "FC12" if "FC12" in r["config_name"] else "FC8"
        mk = "s" if r["data"] == "MULTI" else "o"
        size = {10: 35, 30: 70, 50: 120}.get(r.get("LatDim"), 60)
        for ax, (xk, yk) in zip(axes, [("recon_r2", "control_pass"), ("amp_ratio", "shape_keep")]):
            ax.scatter(r[xk], r[yk], c=colors[fc], marker=mk, s=size, alpha=0.85, edgecolor="white", lw=0.6)
            ax.annotate(r["config_name"].replace("_", " "), (r[xk], r[yk]), fontsize=6.5,
                        xytext=(4, 3), textcoords="offset points", color="#444")
    axes[0].set_xlabel("reconstruction R² (median window)")
    axes[0].set_ylabel("θ control pass rate (all 3 conditions)")
    axes[1].axvline(cp["amp_ratio_min"], color="grey", ls=":", lw=1)
    axes[1].axhline(cp["shape_min"], color="grey", ls=":", lw=1)
    axes[1].set_xlabel("amplitude ratio θ=0.95 / θ=0.05 (median)")
    axes[1].set_ylabel("shape kept: min corr. with own reconstruction (median)")
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color="grey", label="POP"),
                            Line2D([], [], marker="s", ls="", color="grey", label="MULTI"),
                            Line2D([], [], marker="o", ls="", color=colors["FC8"], label="cut-off 8 h"),
                            Line2D([], [], marker="o", ls="", color=colors["FC12"], label="cut-off 12 h")],
                   fontsize=8, frameon=False, loc="lower left")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("%s: reconstruction and θ control trade-off (marker size = latent size)" % stage, x=0.01,
                 ha="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, "sweep", "%s_tradeoff.png" % stage, save)


def pass_breakdown(df, stage="s1", save=True):
    """Share of validation windows meeting each control condition, per configuration (sorted by control_pass)."""
    plt = _plt()
    d = df.sort_values("control_pass", ascending=True)
    cols = [("pass_amp", "amplitude ≥ 1.5×", "#f59e0b"), ("pass_ctrl", "monotone in θ", "#10b981"),
            ("pass_shape", "shape kept", "#6366f1"), ("control_pass", "all three", "#111827")]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(d) + 1.4))
    y = np.arange(len(d))
    h = 0.2
    for i, (c, lab, col) in enumerate(cols):
        ax.barh(y + (i - 1.5) * h, d[c], height=h, color=col, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels(d["config_name"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of MSB validation windows")
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save(fig, "sweep", "%s_pass_breakdown.png" % stage, save)


def cell_line_heatmap(pw, value="control_pass", stage="s1", order=None, save=True):
    """Mean of a per-window metric by cell line (rows) and configuration (columns)."""
    plt = _plt()
    t = pw.pivot_table(index="cell_line", columns="config_name", values=value, aggfunc="mean")
    if order is not None:
        t = t[[c for c in order if c in t.columns]]
    t = t.loc[t.mean(1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(0.62 * t.shape[1] + 2.4, 0.33 * t.shape[0] + 1.6))
    vmin, vmax = (0, 1) if value != "recon_r2" else (max(-0.2, float(np.nanmin(t.values))), 1)
    im = ax.imshow(t.values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=6.5,
                    color="white" if v < (vmin + vmax) / 2 else "black")
    ax.set_xticks(range(t.shape[1]))
    ax.set_xticklabels([c.replace("_", "\n") for c in t.columns], fontsize=7)
    ax.set_yticks(range(t.shape[0]))
    ax.set_yticklabels(t.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=value)
    ax.set_title("%s: %s by cell line (MSB validation, 24 windows per line)" % (stage, value), loc="left",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, "sweep", "%s_cell_line_%s.png" % (stage, value), save)


def plot_examples(ex, stage="s1", dt_h=1 / 6, save=True):
    """Rows = runs, columns = windows. Grey: input; black dashed: reconstruction; coloured: θ_L sweep."""
    plt = _plt()
    import matplotlib as mpl

    rids = list(ex)
    nw = len(ex[rids[0]]["x"])
    fig, axes = plt.subplots(len(rids), nw, figsize=(4.2 * nw, 2.3 * len(rids)), sharex=True, squeeze=False)
    cmap = mpl.colormaps["viridis"]
    for i, rid in enumerate(rids):
        e = ex[rid]
        t = np.arange(e["x"].shape[1]) * dt_h
        for j in range(nw):
            ax = axes[i, j]
            ax.plot(t, e["x"][j] - 0.5, color="#9ca3af", lw=2.2, label="input")
            for k, th in enumerate(e["sweep_theta"]):
                ax.plot(t, e["sweep"][k, j], color=cmap(k / max(1, len(e["sweep_theta"]) - 1)), lw=0.9,
                        label="θ_L=%.2f" % th)
            ax.plot(t, e["gen"][j] - 0.5, color="k", ls="--", lw=1.0, label="reconstruction")
            if i == 0:
                ax.set_title("%s (%s)" % (e["cell_line"][j], e["trace_id"][j][:28]), fontsize=8)
            if j == 0:
                ax.set_ylabel(rid.replace(stage + "_", "").replace("_s0", ""), fontsize=8)
            ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("hours in window")
    axes[0, -1].legend(fontsize=6.5, frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    return _save(fig, "sweep", "%s_examples.png" % stage, save)
