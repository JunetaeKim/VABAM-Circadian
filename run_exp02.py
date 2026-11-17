"""
exp02 entry point: does a disentangled representation (VABAM: theta + z) map onto drug response, published
circadian phenotypes and patient-level expression better than a mixed single-z VAE?
Runs from any working directory; reads and writes inside the repository only.

    python run_exp02.py fetch                                  # public data -> data/raw/external (~1.4 GB, once)
    python run_exp02.py tables                                 # harmonised drug-response and expression tables -> data/processed/external_v1
    python run_exp02.py vabam                                  # analyze VABAM z (needs results/exp01_<model>/codes_all_windows.npz)
    python run_exp02.py pca                                    # linear floor: PCA-30 of the raw windows (no training)
    python run_exp02.py baseline --type BaseVAE --seed 0 --gpu 0   # train + encode a single-z VAE (GPU)
    python run_exp02.py analyze --rep BaseVAE_CIRC_30_s0       # same downstream analysis for that baseline
    python run_exp02.py compare                                # table + figure across all analysed reps
    python run_exp02.py all --gpu 0                            # everything above, 3 BaseVAE seeds + TCVAE + ConVAE

Options: --model <VABAM model> (vabam/all; default SKZFC_CIRC_1_30_200), --epochs N (baseline),
         --n_perm / --n_random (analysis permutation counts), --skip drugs bridge.
Requires tensorflow for 'baseline' only; the analysis needs scikit-learn, scipy, pandas, openpyxl, lifelines.
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "tables", "vabam", "pca", "baseline", "analyze", "compare", "all"])
    ap.add_argument("--model", default="SKZFC_CIRC_1_30_200")
    ap.add_argument("--type", default="BaseVAE", choices=["BaseVAE", "TCVAE", "ConVAE"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--rep", default=None)
    ap.add_argument("--codes", default=None)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--n_random", type=int, default=300)
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()

    if a.step == "tables":                      # needs pandas and openpyxl only
        from src.data import external as fetch_external
        fetch_external.tables()
        fetch_external.cohorts()
        return

    from src.data import external as fetch_external
    from src.analysis import exp02_analyze as analyze
    from src.analysis import exp02_compare as compare

    def ana(rep, codes=None):
        args = ["--rep", rep, "--n_perm", str(a.n_perm), "--n_random", str(a.n_random)]
        args += ["--codes", codes] if codes else []
        args += ["--skip"] + a.skip if a.skip else []
        analyze.main(args)

    def vabam():
        codes = os.path.join(HERE, "results", "exp01_%s" % a.model, "codes_all_windows.npz")
        if not os.path.exists(codes):
            raise SystemExit("missing %s - run: python run.py exp01 --model %s" % (codes, a.model))
        ana("VABAM_%s" % a.model, codes)

    def pca():
        import numpy as np
        X = np.load(os.path.join(HERE, "data", "processed", "windows_msb_rel_L400_S18.npz"))["X01"].astype(float)
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        d = os.path.join(HERE, "results", "exp02", "codes"); os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "PCA30_raw.npz"), z=(Xc @ Vt[:30].T).astype("float32"))
        ana("PCA30_raw")

    def baseline(typ, seed):
        from src.model import baseline as train_baseline
        args = ["--type", typ, "--seed", str(seed)]
        args += ["--gpu", a.gpu] if a.gpu is not None else []
        args += ["--epochs", str(a.epochs)] if a.epochs else []
        train_baseline.main(args)
        return "%s_CIRC_30_s%d" % (typ, seed)

    if a.step == "fetch":
        fetch_external.main()
    elif a.step == "vabam":
        vabam()
    elif a.step == "pca":
        pca()
    elif a.step == "baseline":
        baseline(a.type, a.seed)
    elif a.step == "analyze":
        if not a.rep:
            raise SystemExit("--rep is required (e.g. BaseVAE_CIRC_30_s0)")
        ana(a.rep, a.codes)
    elif a.step == "compare":
        compare.main()
    elif a.step == "all":
        fetch_external.main()
        vabam()
        pca()
        for typ, seed in [("BaseVAE", 0), ("BaseVAE", 1), ("BaseVAE", 2), ("TCVAE", 0), ("ConVAE", 0)]:
            ana(baseline(typ, seed))
        compare.main()


if __name__ == "__main__":
    main()
