"""
Overview figure for the README (used from notebooks/06_readme_figure.ipynb).

(a) one LumiCycle recording with the model input window shaded; (b) the window, its reconstruction and
generations at fixed z with theta_L from 0.05 to 0.95; (c) generated amplitude relative to theta_L = 0.5 over all
non-U2OS MSB validation windows (median and interquartile range).
Without --trace, the window is the one with the largest input SD among windows with reconstruction R^2 >= 0.85 and
amplitude increasing monotonically with theta_L.
Output: results/figures/readme/overview.png
"""
import argparse
import os

from . import vabam_bridge as vb  # before tensorflow

import numpy as np

DT_H = 1 / 6.0
THETA = np.round(np.linspace(0.05, 0.95, 7), 2)


def compute(run, trace=None):
    from . import sweep as sw
    from .exp01_u2os_cry import encode_generate
    from .loader import load_msb2025_raw
    from .sweep_inspect import _model, _val_windows, theta_sweep

    stage, rid = run.split("/")
    cfg, scfg, mc, ckpt, _ = _model(stage, rid)
    x, meta = _val_windows(cfg, mc)
    parts, lv = sw._load_parts(cfg, mc, x.shape[1], ckpt)
    theta, z, _, gen, _ = encode_generate(parts, lv, x, mc["FcLimit"])
    gens = theta_sweep(parts, theta, z, mc["FcLimit"], THETA)          # (n_theta, n_windows, L), centred
    amp = gens.std(2)
    rel = amp / np.maximum(amp[len(THETA) // 2], 1e-12)
    r2 = 1 - np.sum((gen - x) ** 2, 1) / np.maximum(np.sum((x - x.mean(1, keepdims=True)) ** 2, 1), 1e-12)
    if trace:
        cand = np.where(meta["trace_id"].to_numpy() == trace)[0]
        if len(cand) == 0:
            raise ValueError("trace %r is not among the validation windows of %s" % (trace, run))
        k = int(cand[len(cand) // 2])
    else:
        mono = np.all(np.diff(amp, axis=0) > 0, axis=0)
        cand = np.where((r2 >= 0.85) & mono)[0]
        cand = cand if len(cand) else np.arange(len(x))
        k = int(cand[np.argmax(x[cand].std(1))])
    tid, t0 = meta["trace_id"].iloc[k], float(meta["t_start_h"].iloc[k])
    ts = load_msb2025_raw(vb.SPINOFF_ROOT / cfg["data"]["sources"]["msb2025"]["path"])
    return dict(run=run, trace_id=tid, cell_line=meta["cell_line"].iloc[k], reporter=meta["reporter"].iloc[k],
                t0=t0, raw_t=np.asarray(ts.time, float), raw_y=ts.values[tid].to_numpy(float),
                x=x[k] - 0.5, gen=gen[k] - 0.5, sweep=gens[:, k], rel=rel, r2=float(r2[k]), n_windows=len(x),
                val_traces=sorted(meta["trace_id"].unique()))


def plot(d, out=None):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    cmap = matplotlib.colormaps["viridis"]
    norm = Normalize(THETA[0], THETA[-1])
    L = len(d["x"])
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6), gridspec_kw={"width_ratios": [1.35, 1.35, 0.85]})

    a = ax[0]
    a.plot(d["raw_t"], d["raw_y"] / 1e3, color="#374151", lw=1.0)
    a.axvspan(d["t0"], d["t0"] + L * DT_H, color="#fbbf24", alpha=0.25, lw=0)
    a.set_xlabel("time since start of recording (h)")
    a.set_ylabel("bioluminescence (10$^3$ counts)")
    a.set_title("%s, %s-Luc" % (d["cell_line"], d["reporter"]), fontsize=10, loc="left")

    b = ax[1]
    t = d["t0"] + np.arange(L) * DT_H
    b.plot(t, d["x"], color="#9ca3af", lw=2.4, label="input window")
    for k, th in enumerate(THETA):
        b.plot(t, d["sweep"][k], color=cmap(norm(th)), lw=1.1)
    b.plot(t, d["gen"], color="k", ls="--", lw=1.1, label="reconstruction")
    b.set_xlabel("time since start of recording (h)")
    b.set_ylabel("normalised signal")
    b.set_title("generation at fixed z, θ$_L$ swept", fontsize=10, loc="left")
    b.legend(fontsize=8, frameon=False, loc="upper right")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=b, fraction=0.04, pad=0.02)
    cb.set_label("θ$_L$", fontsize=9)

    c = ax[2]
    q25, q50, q75 = np.percentile(d["rel"], [25, 50, 75], axis=1)
    c.fill_between(THETA, q25, q75, color="#93c5fd", alpha=0.5, lw=0, label="IQR")
    c.plot(THETA, q50, color="#1d4ed8", marker="o", ms=4, lw=1.6, label="median")
    c.axhline(1, color="#9ca3af", lw=0.8, ls=":")
    c.set_xlabel("θ$_L$")
    c.set_ylabel("generated amplitude / at θ$_L$ = 0.5")
    c.set_title("all validation windows (n = %d)" % d["n_windows"], fontsize=10, loc="left")
    c.legend(fontsize=8, frameon=False, loc="upper left")

    for a, lab in zip(ax, "abc"):
        a.text(-0.12, 1.06, lab, transform=a.transAxes, fontsize=13, fontweight="bold")
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)
    fig.tight_layout()
    if out is not None:
        out = vb.assert_inside_spinoff(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


OUT = vb.SPINOFF_ROOT / "results" / "figures" / "readme" / "overview.png"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="sweep run as stage/run_id, e.g. s1/s1_MULTI_FC12_Z10_s0")
    ap.add_argument("--trace", default=None, help="trace_id of a validation recording (optional)")
    ap.add_argument("--gpu", default="-1", help="GPU id; -1 = CPU (default)")
    a = ap.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    d = compute(a.run, a.trace)
    plot(d, OUT)
    print(describe(d))


def describe(d):
    return "%s | %s (%s), window %.0f-%.0f h, reconstruction R2 %.2f -> %s" % (
        d["run"], d["trace_id"], d["cell_line"], d["t0"], d["t0"] + len(d["x"]) * DT_H, d["r2"], OUT)


if __name__ == "__main__":
    main()
