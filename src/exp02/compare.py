"""Collect results/exp02/<rep>/summary.json for all representations into one table + figure.

    python run_exp02.py compare
Output: results/exp02/comparison.csv, results/exp02/comparison.png
Rows = representations (VABAM, BaseVAE seeds, TCVAE, ConVAE ...); a 'classical' row (same for every rep) is
taken from the first summary for reference.
"""
import json
import numpy as np, pandas as pd
from .common import RES, outdir


def pick(recs, key, val, field):
    for r in recs or []:
        if r.get(key) == val:
            return r.get(field)
    return np.nan


def row(s, which):
    cl = s.get("cluster"); idn = s.get("identification"); gd = s.get("gdsc"); cis = s.get("cisplatin")
    sv = [r for r in (s.get("survival") or []) if r["feature"].startswith("rep_" if which == "rep" else "cl_")]
    return {
        "phenotype_R2": pick(cl, "features", which, "phenotype_R2"),
        "phenotype_p": pick(cl, "features", which, "p_R2"),
        "phenotype_ARI": pick(cl, "features", which, "ARI_ward"),
        "line_id_acc_Per2": next((r["acc"] for r in idn or [] if r["features"] == which and r["reporter"] == "Per2"), np.nan),
        "cisplatin_LOO_rho": pick(cis, "feature", "LOO_ridge_" + which, "rho"),
        "gdsc_drugs_LOO_gt_0.5": pick(gd, "features", which, "n_drugs_LOO_rho_gt_0_5"),
        "gdsc_null_95pct": pick(gd, "features", which, "null_95pct"),
        "gdsc_p": pick(gd, "features", which, "p"),
        "survival_best_pct_vs_random": max([r["pct_vs_random"] for r in sv], default=np.nan),
        "survival_n_adj_p_lt_0.05": sum(r["p_adj"] < 0.05 for r in sv),
    }


def main(argv=None):
    S = [json.load(open(p, encoding="utf-8")) for p in sorted(RES.glob("*/summary.json"))]
    if not S:
        raise SystemExit("no results/exp02/<rep>/summary.json yet - run analyze first")
    rows = [{"rep": s["rep"], "latent_dim": s["latent_dim"], **row(s, "rep")} for s in S]
    rows.append({"rep": "classical (reference)", "latent_dim": 6, **row(S[0], "classical")})
    T = pd.DataFrame(rows)
    out = outdir()
    T.to_csv(out / "comparison.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        m = ["phenotype_R2", "line_id_acc_Per2", "cisplatin_LOO_rho", "gdsc_drugs_LOO_gt_0.5"]
        fig, ax = plt.subplots(1, len(m), figsize=(4 * len(m), 3.6))
        for a, k in zip(ax, m):
            a.barh(T.rep, T[k], color=["#4c72b0" if "VABAM" in r else "#999999" if "classical" in r else "#dd8452"
                                        for r in T.rep])
            a.set_title(k, fontsize=9); a.tick_params(labelsize=7)
        plt.tight_layout(); plt.savefig(out / "comparison.png", dpi=150)
    except Exception as e:  # figure is optional
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
