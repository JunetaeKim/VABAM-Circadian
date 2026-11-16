"""
Train VABAM (official ModelCall / DCLCall, unchanged) on preprocessed circadian windows.

    python -m src.train --config configs/circadian_vabam.yml --model SKZFC_CIRC_1_30_200

Writes only under results/:
    results/ckpt/<SigType>/<model>.hdf5      best model (RelLossWeight, max of val_FeatRecLoss/val_OrigRecLoss)
    results/Logs/<SigType>/Logs_<model>.txt  RelLossWeight's own log
    results/train/<model>_history.csv        Keras history
    results/train/<model>_run.json           settings and data counts
"""
import argparse
import json
import os
import random
import time

from . import vabam_bridge as vb  # must be imported before tensorflow

import numpy as np
import pandas as pd
import yaml


def load_windows(cfg, tag):
    d = np.load(vb.SPINOFF_ROOT / cfg["paths"]["processed"] / ("windows_%s.npz" % tag), allow_pickle=False)
    meta = pd.read_csv(vb.SPINOFF_ROOT / cfg["paths"]["processed"] / ("windows_%s_meta.csv" % tag))
    return d["X01"], d["Xn"], meta


def make_random_crop_sequence(cfg, tag, crops_per_trace, batch_size, seed):
    """Keras Sequence: every epoch draws `crops_per_trace` random contiguous windows from EACH training trace
    (trace-balanced; no wrapping, no concatenation of different traces) and scales them like preprocess.py."""
    import tensorflow as tf

    proc = vb.SPINOFF_ROOT / cfg["paths"]["processed"]
    d = np.load(proc / ("traces_%s.npz" % tag), allow_pickle=False)
    info = __import__("json").load(open(proc / ("scaling_%s.json" % tag), encoding="utf-8"))
    tr = d["split"] == "train"
    Xf, lens = d["Xn_full"][tr], d["lengths"][tr]
    L, q = int(info["length_samples"]), float(info["q_abs"])

    class RandomCrops(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.n = int(len(lens) * crops_per_trace)
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(self.n / batch_size))

        def on_epoch_end(self):
            rows = []
            for x, n in zip(Xf, lens):
                for s in self.rng.integers(0, n - L + 1, size=crops_per_trace):
                    rows.append(x[s:s + L])
            X = np.clip(0.5 + np.stack(rows) / (2 * q), 0.0, 1.0).astype(np.float32)
            self.X = X[self.rng.permutation(len(X))]

        def __getitem__(self, i):
            return self.X[i * batch_size:(i + 1) * batch_size]

    return RandomCrops(), int(len(lens) * crops_per_trace), int(len(lens))


def make_pool_sequence(cfg, tag, weights, epoch_windows, batch_size, seed, val_sources=None):
    """Configuration B: every epoch draw `epoch_windows` windows from the training pool
    (data/processed/<tag>/pool_train.npy), split across sources by `weights`; uniform within a source.
    Returns (sequence, validation X01 restricted to the weighted sources, info dict)."""
    import tensorflow as tf

    d = vb.SPINOFF_ROOT / cfg["paths"]["processed"] / tag
    Xf = np.load(d / "traces.npz")["Xn_full"]
    rec = pd.read_csv(d / "records.csv", low_memory=False)
    pool = np.load(d / "pool_train.npy")
    L = int(json.load(open(d / "scaling.json", encoding="utf-8"))["window"])
    q = rec["q"].to_numpy(np.float32)
    src_of_pool = rec["source"].to_numpy()[pool[:, 0]]
    wsum = float(sum(weights.values()))
    groups, n_per = {}, {}
    for s_, w in weights.items():
        groups[s_] = np.where(src_of_pool == s_)[0]
        if len(groups[s_]) == 0:
            raise ValueError("Source %r has no training windows in %s" % (s_, d))
        n_per[s_] = int(round(epoch_windows * w / wsum))
    offs = np.arange(L)

    class PoolWindows(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.n = int(sum(n_per.values()))
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(self.n / batch_size))

        def on_epoch_end(self):
            sel = np.concatenate([self.rng.choice(groups[s_], size=n_per[s_], replace=n_per[s_] > len(groups[s_]))
                                  for s_ in groups])
            rows = pool[sel]
            X = Xf[rows[:, 0][:, None], rows[:, 1][:, None] + offs]
            X = np.clip(0.5 + X / (2 * q[rows[:, 0]][:, None]), 0.0, 1.0).astype(np.float32)
            self.X = X[self.rng.permutation(len(X))]

        def __getitem__(self, i):
            return self.X[i * batch_size:(i + 1) * batch_size]

    v = np.load(d / "val_windows.npz")
    vsrc = rec["source"].to_numpy()[v["index"][:, 0]]
    val_sources = list(val_sources or weights)
    vm = np.isin(vsrc, val_sources)
    info = {"pool_windows_by_source": {s_: int(len(g)) for s_, g in groups.items()},
            "epoch_windows_by_source": n_per,
            "val_sources": val_sources,
            "val_windows_by_source": {s_: int((vsrc == s_).sum()) for s_ in val_sources},
            "train_records_by_source": rec[rec["split"] == "train"].groupby("source").size().to_dict(),
            "train_groups_by_source": rec[rec["split"] == "train"].groupby("source")["group_id"].nunique().to_dict()}
    return PoolWindows(), v["X01"][vm], info


def reset_all_seeds(seed):
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_gpu(gpu=None):
    """Select the GPU (CUDA_VISIBLE_DEVICES) before TensorFlow starts and enable memory growth."""
    if gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print("[train] visible GPUs: %s" % ([g.name for g in gpus] or "none (CPU)"), flush=True)
    return tf


def fit_model(cfg, mc, res_dir, rel_ckpt, n_epochs=None, seed=None, verbose=2):
    """Train one VABAM model (official ModelCall / DCLCall). Runs with cwd = res_dir so that the checkpoint
    (rel_ckpt, e.g. './ckpt/<dir>/<name>.hdf5') and RelLossWeight's './Logs/<dir>/' stay inside res_dir.
    Returns (history DataFrame, info dict)."""
    import tensorflow as tf

    seed = cfg["train"]["seed"] if seed is None else seed
    reset_all_seeds(seed)
    v = vb.import_vabam(cfg)
    X01, _, meta = load_windows(cfg, mc["DataSource"])
    tr = (meta["split"] == "train").to_numpy()
    va = (meta["split"] == "val").to_numpy()
    TrData, ValData = X01[tr], X01[va]
    SigDim, DataSize = TrData.shape[1], TrData.shape[0]
    n_epochs = n_epochs or mc["NEpochs"]
    sampling = mc.get("Sampling", "fixed")
    pool_info = None
    if sampling == "random_crop":
        # Same validation windows as the fixed-stride model; training windows re-drawn every epoch.
        TrData, DataSize, n_tr_traces = make_random_crop_sequence(
            cfg, mc["DataSource"], mc.get("CropsPerTrace", 24), mc["BatSize"], seed)
    elif sampling == "pool":
        TrData, ValData, pool_info = make_pool_sequence(
            cfg, mc["DataSource"], mc["SourceWeights"], mc.get("EpochWindows", 32768), mc["BatSize"],
            seed, val_sources=mc.get("ValSources"))
        DataSize = int(sum(pool_info["epoch_windows_by_source"].values()))
    elif sampling != "fixed":
        raise ValueError("Unknown Sampling %r" % sampling)

    # Optional XLA compilation (global TF switch, official code untouched). Off unless the config sets XLA: true.
    tf.config.optimizer.set_jit(bool(mc.get("XLA", False)))
    timer = EpochTimer()
    t0 = time.time()
    with vb.working_dir(res_dir):
        os.makedirs(os.path.dirname(rel_ckpt), exist_ok=True)
        model = v.ModelCall(mc, SigDim, DataSize, Resume=False, Reparam=True, ModelSaveName=rel_ckpt,
                            ModelSummary=False)
        rel_loss = v.DCLCall(mc, rel_ckpt, ToSaveLoss=None, SaveWay="max", Resume=False,
                             Patience=mc.get("Patience", cfg["train"]["patience"]))
        if sampling == "fixed":
            hist = model.fit(TrData, batch_size=mc["BatSize"], epochs=n_epochs, shuffle=True,
                             validation_data=(ValData, None), callbacks=[rel_loss, timer], verbose=verbose)
        else:
            hist = model.fit(TrData, epochs=n_epochs, validation_data=(ValData, None),
                             callbacks=[rel_loss, timer], verbose=verbose)
    elapsed = time.time() - t0
    h = pd.DataFrame(hist.history)
    h.index.name = "epoch"
    info = {
        "vabam_root": str(v.root), "tensorflow": tf.__version__, "seed": int(seed), "sampling": sampling,
        "n_train_windows_per_epoch": int(DataSize), "n_val_windows": int(len(ValData)), "sig_dim": int(SigDim),
        "n_train_traces": int(meta.loc[tr, "trace_id"].nunique()),
        "n_train_dates": int(meta.loc[tr, "date"].nunique()),
        "pool": pool_info,
        "epochs_run": int(len(h)), "epochs_max": int(n_epochs), "elapsed_s": round(elapsed, 1),
        "sec_per_epoch": round(elapsed / max(len(h), 1), 2),
        # median epoch time without the first epoch (graph tracing / XLA compilation)
        "sec_per_epoch_steady": round(float(np.median(timer.times[1:] or timer.times or [np.nan])), 2),
        "batch_size": int(mc["BatSize"]), "xla": bool(mc.get("XLA", False)),
        "final": {k: float(h[k].iloc[-1]) for k in h.columns},
    }
    h["sec"] = (timer.times + [np.nan] * len(h))[:len(h)]
    return h, info


def _epoch_timer_base():
    import tensorflow as tf

    return tf.keras.callbacks.Callback


class EpochTimer:
    """Keras callback (created lazily so this module imports without TensorFlow) recording wall time per epoch."""

    def __new__(cls):
        base = _epoch_timer_base()

        class _T(base):
            def __init__(self):
                super().__init__()
                self.times = []

            def on_epoch_begin(self, epoch, logs=None):
                self._t = time.time()

            def on_epoch_end(self, epoch, logs=None):
                self.times.append(round(time.time() - self._t, 3))

        return _T()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override NEpochs (e.g. smoke test)")
    ap.add_argument("--verbose", type=int, default=2)
    ap.add_argument("--gpu", type=str, default=None, help="GPU id (sets CUDA_VISIBLE_DEVICES); '-1' = CPU only")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    setup_gpu(args.gpu)
    mc = vb.model_config(cfg, args.model)
    res = vb.results_dir(cfg)
    h, info = fit_model(cfg, mc, res, vb.checkpoint_relpath(cfg, args.model), args.epochs,
                        cfg["train"]["seed"], args.verbose)

    out = vb.assert_inside_spinoff(res / "train")
    out.mkdir(parents=True, exist_ok=True)
    h.to_csv(out / ("%s_history.csv" % args.model))
    run = {"model": args.model, "config": mc, **info, "checkpoint": str(vb.checkpoint_path(cfg, args.model))}
    with open(out / ("%s_run.json" % args.model), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, default=str)
    print(json.dumps({k: run[k] for k in ("model", "sampling", "n_train_windows_per_epoch", "n_val_windows",
                                          "epochs_run", "elapsed_s", "checkpoint")}, indent=2))


if __name__ == "__main__":
    main()
