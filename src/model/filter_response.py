"""
Frequency response of VABAM's cascading filters (official GenLowFilter / GenHighFilter, imported unchanged)
for circadian sampling, used to choose FcLimit.

In VABAM the cut-off is in cycles/sample (normalised frequency, Nyquist = 0.5) and the encoder maps
theta in (0, 1) to a cut-off in (0, FcLimit). With 10-min sampling, a period P [h] corresponds to
f = dt_h / P cycles/sample (24 h -> 1/144 = 0.00694).

Usage:
    python -m src.model.filter_response --config configs/circadian_vabam.yml
"""
import argparse

from src import vabam_bridge as vb  # must come before tensorflow

import numpy as np
import yaml


def kernel_size(sig_dim, comp_size, depth):
    return (sig_dim - comp_size) // depth + 1


def gain_at(h, f):
    n = np.arange(len(h))
    return np.abs(np.sum(h * np.exp(-2j * np.pi * f * n)))


def response_table(cfg, fc_limits, depth=1):
    import tensorflow as tf  # noqa: F401

    v = vb.import_vabam(cfg)
    dt_h = cfg["data"]["dt_min"] / 60.0
    sig_dim = cfg["preprocess"]["window"]["length_samples"]
    comp = cfg["vabam"]["Common_Param"]["CompSize"]
    N = kernel_size(sig_dim, comp, depth)
    periods = np.array([48.0, 24.0, 12.0, 8.0, 4.0])
    thetas = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    rows = []
    for fl in fc_limits:
        for th in thetas:
            fc = np.array([[th * fl]], dtype=np.float32)
            lp = v.GenLowFilter(fc, N).numpy().ravel()
            hp = v.GenHighFilter(fc, N).numpy().ravel()
            for P in periods:
                f = dt_h / P
                rows.append((fl, th, th * fl, dt_h / (th * fl), P, gain_at(lp, f), gain_at(hp, f)))
    return N, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fc_limits", type=float, nargs="*", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    dt_h = cfg["data"]["dt_min"] / 60.0
    fls = args.fc_limits or [1.0, dt_h / 4.0, dt_h / 8.0, dt_h / 12.0, cfg["vabam"]["Common_Param"]["FcLimit"]]
    for depth in sorted({m["Depth"] for m in cfg["vabam"]["Models"].values()}):
        N, rows = response_table(cfg, fls, depth=depth)
        print("\nDepth=%d  kernel length N=%d samples (%.1f h)" % (depth, N, N * dt_h))
        print("%-9s %-5s %-9s %-10s %-6s %-8s %-8s" % ("FcLimit", "theta", "fc[c/s]", "Pc[h]", "P[h]", "|LPF|", "|HPF|"))
        for fl, th, fc, pc, P, gl, gh in rows:
            if P in (24.0, 12.0, 48.0):
                print("%-9.5f %-5.2f %-9.5f %-10.1f %-6.0f %-8.3f %-8.3f" % (fl, th, fc, pc, P, gl, gh))


if __name__ == "__main__":
    main()
