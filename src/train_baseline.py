"""
Train a single-z baseline VAE from the official VABAM repository (Benchmarks/Models, imported unchanged) on the
same circadian windows as VABAM, then encode every window (z = posterior mean) for exp02.

    python run_exp02.py baseline --type BaseVAE --seed 0 --gpu 0
    python run_exp02.py baseline --type TCVAE   --seed 0 --gpu 0
    python run_exp02.py baseline --type ConVAE  --seed 0 --gpu 0

Types (all LatDim 30 like VABAM, same windows/split, batch 256, max 1500 epochs, patience 150):
  BaseVAE  plain VAE, one mixed z                              (Benchmarks BaseVAE)
  TCVAE    beta-TC-VAE, one z with total-correlation penalty   (Benchmarks TCVAE; input PSD condition as in the paper)
  ConVAE   conditional VAE: z + PSD (amplitude) condition      (Benchmarks ConVAE)

Writes only under results/:
  results/ckpt/CIRC/<name>.hdf5, results/Logs/CIRC/Logs_<name>.txt   (RelLossWeight, as for VABAM)
  results/train/<name>_history.csv, results/train/<name>_run.json
  results/exp02/codes/<name>.npz   (z for all windows, rows aligned with windows_<tag>_meta.csv)
"""
import argparse
import json
import os
import random
import time

from . import vabam_bridge as vb  # before tensorflow

import numpy as np
import pandas as pd
import yaml

COMMON = dict(SigType="CIRC", ReparaStd=0.1, Capacity_Z=0.05, WRec=1, WZ=1, MnWRec=1, MxWRec=1,
              MnWZ=0.00001, MxWZ=0.00001)
EXTRA = {"BaseVAE": {}, "ConVAE": {},
         "TCVAE": dict(Capacity_TC=0.01, Capacity_MI=0.1, WTC=1, WMI=1, MnWTC=0.00001, MxWTC=0.00001,
                       MnWMI=0.00001, MxWMI=0.00001)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=list(EXTRA))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latdim", type=int, default=30)
    ap.add_argument("--tag", default="msb_rel_L400_S18")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--encode_only", action="store_true", help="skip training; load checkpoint and encode")
    ap.add_argument("--config", default=str(vb.SPINOFF_ROOT / "configs" / "circadian_vabam.yml"))
    ap.add_argument("--verbose", type=int, default=2)
    a = ap.parse_args(argv)

    if a.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    import tensorflow as tf

    random.seed(a.seed); np.random.seed(a.seed); tf.random.set_seed(a.seed); os.environ["PYTHONHASHSEED"] = str(a.seed)
    for g in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    vb.import_vabam(cfg)  # puts the official repo on sys.path (import only)
    from Benchmarks.Models import BenchmarkModels as BM, BenchmarkCaller as BC  # noqa: E402 (official code)

    name = "%s_CIRC_%d_s%d" % (a.type, a.latdim, a.seed)
    proc = vb.SPINOFF_ROOT / cfg["paths"]["processed"]
    X01 = np.load(proc / ("windows_%s.npz" % a.tag))["X01"].astype("float32")
    meta = pd.read_csv(proc / ("windows_%s_meta.csv" % a.tag))
    tr, va = (meta.split == "train").to_numpy(), (meta.split == "val").to_numpy()
    spec = {**COMMON, **EXTRA[a.type], "LatDim": a.latdim, "BatSize": a.batch, "NEpochs": a.epochs,
            "DataSource": a.tag, "SigDim": X01.shape[1], "DataSize": int(tr.sum())}

    def inputs(X):
        if a.type == "BaseVAE":
            return X
        return [X, BC.FFT_PSD_Vanilla(X)[:, 0]]

    res = vb.results_dir(cfg)
    rel_ckpt = "./ckpt/CIRC/%s.hdf5" % name
    t0 = time.time()
    with vb.working_dir(res):
        os.makedirs(os.path.dirname(rel_ckpt), exist_ok=True)
        if a.type == "BaseVAE":
            model = BM.BaseVAE(X01.shape[1], spec, Reparam=True)
        else:
            model, _, _ = BC.ModelCall(spec, name, X01[tr], X01[va], Reparam=True, ModelSaveName=rel_ckpt)
        if not a.encode_only:
            rel = BC.DCLCall(spec, name, rel_ckpt, ToSaveLoss=None, SaveWay="max", Patience=a.patience)
            hist = model.fit(inputs(X01[tr]), batch_size=a.batch, epochs=a.epochs, shuffle=True,
                             validation_data=(inputs(X01[va]), None), callbacks=[rel], verbose=a.verbose)
        model.load_weights(rel_ckpt)
        enc = tf.keras.Model(model.inputs, model.get_layer("Z_Mu").output)
        Z = enc.predict(inputs(X01), batch_size=1024, verbose=0)
    elapsed = time.time() - t0

    out = vb.assert_inside_spinoff(res / "exp02" / "codes"); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / ("%s.npz" % name), z=Z.astype("float32"))
    if not a.encode_only:
        tdir = vb.assert_inside_spinoff(res / "train"); tdir.mkdir(parents=True, exist_ok=True)
        h = pd.DataFrame(hist.history); h.index.name = "epoch"; h.to_csv(tdir / ("%s_history.csv" % name))
        json.dump({"model": name, "type": a.type, "spec": spec, "tensorflow": tf.__version__, "seed": a.seed,
                   "epochs_run": int(len(h)), "elapsed_s": round(elapsed, 1),
                   "final": {k: float(h[k].iloc[-1]) for k in h.columns}},
                  open(tdir / ("%s_run.json" % name), "w", encoding="utf-8"), indent=2, default=str)
    print("codes:", out / ("%s.npz" % name), Z.shape)


if __name__ == "__main__":
    main()
