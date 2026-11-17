"""
Parameter search engine (configs/sweep.yml). Used from notebooks/03_param_search.ipynb and from terminals.

Workers (one process per GPU; worker k of n runs every n-th run of the frozen plan):
    python run.py sweep --stage s1 --worker 0 --nworkers 4 --gpu 0
Notebook (no TensorFlow needed):
    from src.model import sweep
    sweep.freeze_plan("s1"); sweep.commands("s1"); sweep.status("s1"); sweep.collect("s1")

Layout (all under results/sweep/<stage>/):
    plan.json                       frozen run list (workers only run what is in it)
    runs/<run_id>/params.json       model overrides, seed, epochs
    runs/<run_id>/RUNNING           host / pid / start time while training
    runs/<run_id>/history.csv, run.json, metrics.json   when finished;  error.txt if it failed
    ckpt/<run_id>/model.hdf5        best checkpoint (official RelLossWeight)
    Logs/<run_id>/Logs_model.txt    RelLossWeight log, one line per epoch (used for live curves)
"""
import argparse
import itertools
import json
import os
import re
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import vabam_bridge as vb

_LOG_ITEM = re.compile(r"'(\w+)':\s*(?:np\.float\d+\()?\s*([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))")

SWEEP_CFG = vb.SPINOFF_ROOT / "configs" / "sweep.yml"
MAIN_CFG = vb.SPINOFF_ROOT / "configs" / "circadian_vabam.yml"


# ------------------------------------------------------------------ config / paths
def load_cfgs():
    return (yaml.safe_load(open(MAIN_CFG, encoding="utf-8")), yaml.safe_load(open(SWEEP_CFG, encoding="utf-8")))


def stage_dir(scfg, stage):
    return vb.assert_inside_spinoff(vb.SPINOFF_ROOT / scfg["results_dir"] / stage)


def run_dir(scfg, stage, run_id):
    return stage_dir(scfg, stage) / "runs" / run_id


# ------------------------------------------------------------------ planning
def _grid_runs(stage, st):
    factors = list(st["grid"].items())
    runs = []
    for combo in itertools.product(*[list(levels.items()) for _, levels in factors]):
        ov, labels = {}, []
        for (fname, _), (label, val) in zip(factors, combo):
            labels.append(label)
            ov.update(val if isinstance(val, dict) else {fname: val})
        for seed in st.get("seeds", [0]):
            name = "_".join(labels)
            runs.append({"run_id": "%s_%s_s%d" % (stage, name, seed), "config_name": name, "overrides": ov,
                         "seed": seed})
    return runs


def _from_runs(stage, st, scfg):
    src = st["from"]
    ranked = rank(collect(src["stages"], scfg=scfg), scfg=scfg, by_config=True)
    if ranked.empty:
        raise RuntimeError("No finished runs in %s; finish them before freezing %s" % (src["stages"], stage))
    top = ranked.head(src["top_k"])
    runs = []
    for _, r in top.iterrows():
        parent_ov = json.loads(r["overrides_json"])
        variants = st.get("variants") or {"": {}}
        for vname, vov in variants.items():
            name = r["config_name"] + ("_" + vname if vname else "")
            for seed in st.get("seeds", [0]):
                runs.append({"run_id": "%s_%s_s%d" % (stage, name, seed), "config_name": name,
                             "overrides": {**parent_ov, **vov}, "seed": seed, "parent": r["config_name"]})
    return runs


def make_plan(stage, scfg=None):
    scfg = scfg or load_cfgs()[1]
    st = scfg["stages"][stage]
    runs = _from_runs(stage, st, scfg) if "from" in st else _grid_runs(stage, st)
    for r in runs:
        r.update(stage=stage, epochs=st["epochs"], patience=st["patience"])
    return runs


def freeze_plan(stage, overwrite=False):
    """Write results/sweep/<stage>/plan.json (once). Workers on every PC read this file."""
    _, scfg = load_cfgs()
    p = stage_dir(scfg, stage) / "plan.json"
    if p.exists() and not overwrite:
        print("plan already frozen:", p)
        return pd.DataFrame(json.load(open(p, encoding="utf-8"))["runs"])
    runs = make_plan(stage, scfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": runs}, f, indent=2)
    print("frozen %d runs -> %s" % (len(runs), p))
    return pd.DataFrame(runs)


def load_plan(stage, scfg=None):
    scfg = scfg or load_cfgs()[1]
    p = stage_dir(scfg, stage) / "plan.json"
    if not p.exists():
        raise FileNotFoundError("Plan for %s not frozen yet: run sweep.freeze_plan('%s') (notebook) first" %
                                (stage, stage))
    return json.load(open(p, encoding="utf-8"))["runs"]


def commands(stage, python="python"):
    """Terminal commands, one per GPU over all hosts in configs/sweep.yml."""
    _, scfg = load_cfgs()
    slots = [(h, g) for h, gpus in scfg["hosts"].items() for g in gpus for _ in range(scfg.get("per_gpu", 1))]
    n = len(slots)
    lines = []
    for w, (h, g) in enumerate(slots):
        lines.append("[%s] %s \"%s\" sweep --stage %s --worker %d --nworkers %d --gpu %d" % (
            h, python, vb.SPINOFF_ROOT / "run.py", stage, w, n, g))
    print("\n".join(lines))
    return lines


# ------------------------------------------------------------------ model config for a run
def run_model_config(cfg, scfg, run):
    mc = vb.model_config(cfg, scfg["base_model"])
    mc.update(scfg.get("common_overrides", {}))
    mc.update(run["overrides"])
    mc.update(NEpochs=run["epochs"], Patience=run["patience"], ConfigName=run["run_id"])
    return mc


# ------------------------------------------------------------------ label-free evaluation
def _load_parts(cfg, mc, sig_dim, ckpt):
    import tensorflow as tf

    v = vb.import_vabam(cfg)
    _, parts = v.ModelCall(mc, sig_dim, 1000, LoadWeight=True, ReturnModelPart=True, Reparam=False,
                           ModelSaveName=str(ckpt), ModelSummary=False)
    return parts, tf.keras.Model(parts[0].input, parts[0].get_layer("Z_Log_Sigma").output)


def evaluate_run(cfg, scfg, mc, ckpt, eval_overrides=None):
    from scipy.stats import spearmanr

    from src.analysis.exp01_u2os_cry import encode_generate, generate
    from src.model.train import load_windows

    sel = {**scfg["selection"], **(eval_overrides or {})}
    rng = np.random.default_rng(0)
    X01, Xn, meta = load_windows(cfg, mc["DataSource"])
    idx = np.where(((meta["split"] == "val") & (meta["cell_line"] != "U2OS")).to_numpy())[0]
    if len(idx) > sel["n_eval_windows"]:
        idx = np.sort(rng.choice(idx, sel["n_eval_windows"], replace=False))
    parts, lv = _load_parts(cfg, mc, X01.shape[1], ckpt)
    x = X01[idx]
    theta, z, _, gen, _ = encode_generate(parts, lv, x, mc["FcLimit"])
    r2 = 1 - np.sum((gen - x) ** 2, 1) / np.maximum(np.sum((x - x.mean(1, keepdims=True)) ** 2, 1), 1e-12)

    sub = rng.choice(len(x), size=min(sel["n_sweep"], len(x)), replace=False)
    sw = np.array(sel["sweep_theta"])
    base = generate(parts, theta[sub] * mc["FcLimit"], z[sub]) - 0.5
    gens, amps = [], []
    for th in sw:
        fc = theta[sub] * mc["FcLimit"]
        fc[:, -1] = th * mc["FcLimit"]
        g = generate(parts, fc, z[sub]) - 0.5
        gens.append(g)
        amps.append(g.std(1))
    amps = np.array(amps)
    ok = base.std(1) > 1e-6
    ratio = amps[-1] / np.maximum(amps[0], 1e-12)
    ctrl = np.array([spearmanr(sw, amps[:, k])[0] if np.ptp(amps[:, k]) > 0 else 0.0 for k in range(len(sub))])
    shape_min = np.array([min(np.corrcoef(gens[j][k], base[k])[0, 1] for j in range(len(sw)))
                          if ok[k] else np.nan for k in range(len(sub))])
    cp = sel["control_pass"]
    passed = (ratio >= cp["amp_ratio_min"]) & (ctrl >= cp["ctrl_min"]) & (shape_min >= cp["shape_min"]) & ok
    return {
        "recon_r2": float(np.median(r2)), "recon_r2_q25": float(np.quantile(r2, 0.25)),
        "recon_mse": float(np.median(np.mean((gen - x) ** 2, 1))),
        "amp_ratio": float(np.median(ratio)), "thetaL_ctrl": float(np.nanmedian(ctrl)),
        "shape_keep": float(np.nanmedian(shape_min)), "control_pass": float(np.mean(passed)),
        "pass_amp": float(np.mean(ratio >= cp["amp_ratio_min"])), "pass_ctrl": float(np.mean(ctrl >= cp["ctrl_min"])),
        "pass_shape": float(np.nanmean(shape_min >= cp["shape_min"])),
        "theta_inferred_sd": float(theta[:, -1].std()), "z_active_dims": int(np.sum(np.var(z, 0) > 0.01)),
        "n_eval_windows": int(len(idx)), "n_eval_traces": int(meta.iloc[idx]["trace_id"].nunique()),
        "n_sweep_windows": int(len(sub)),
    }


# ------------------------------------------------------------------ worker
def _done(rd):
    return (rd / "metrics.json").exists()


def train_and_eval(cfg, scfg, run, verbose=2):
    from src.model.train import fit_model

    sd = stage_dir(scfg, run["stage"])
    rd = run_dir(scfg, run["stage"], run["run_id"])
    rd.mkdir(parents=True, exist_ok=True)
    mc = run_model_config(cfg, scfg, run)
    with open(rd / "params.json", "w", encoding="utf-8") as f:
        json.dump({"run": run, "model_config": mc}, f, indent=2, default=str)
    rel_ckpt = "./ckpt/%s/model.hdf5" % run["run_id"]
    h, info = fit_model(cfg, mc, sd, rel_ckpt, n_epochs=run["epochs"], seed=run["seed"], verbose=verbose)
    h.to_csv(rd / "history.csv")
    info.update(host=socket.gethostname(), run_id=run["run_id"])
    with open(rd / "run.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    st = scfg["stages"][run["stage"]]
    ev = st.get("eval")
    m = {} if ev is False else evaluate_run(cfg, scfg, mc, sd / rel_ckpt, ev)
    m.update(best_val_OrigRecLoss=float(h["val_OrigRecLoss"].min()),
             best_val_FeatRecLoss=float(h["val_FeatRecLoss"].min()),
             epochs_run=info["epochs_run"], sec_per_epoch=info["sec_per_epoch"],
             sec_per_epoch_steady=info["sec_per_epoch_steady"], batch_size=info["batch_size"], xla=info["xla"],
             windows_per_epoch=info["n_train_windows_per_epoch"], host=info["host"])
    with open(rd / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    return m


def worker(stage, w, n, gpu=None, verbose=2):
    from src.model.train import setup_gpu

    cfg, scfg = load_cfgs()
    plan = load_plan(stage, scfg)
    mine = [r for i, r in enumerate(plan) if i % n == w]
    tf = setup_gpu(gpu)
    print("[sweep] stage %s, worker %d/%d, %d runs assigned: %s" % (stage, w, n, len(mine),
                                                                  [r["run_id"] for r in mine]), flush=True)
    for r in mine:
        rd = run_dir(scfg, stage, r["run_id"])
        if _done(rd):
            print("[sweep] skip (done) %s" % r["run_id"], flush=True)
            continue
        rd.mkdir(parents=True, exist_ok=True)
        with open(rd / "RUNNING", "w", encoding="utf-8") as f:
            json.dump({"host": socket.gethostname(), "pid": os.getpid(), "gpu": gpu,
                       "start": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        print("[sweep] >>> %s  overrides=%s seed=%d" % (r["run_id"], r["overrides"], r["seed"]), flush=True)
        try:
            tf.keras.backend.clear_session()
            m = train_and_eval(cfg, scfg, r, verbose=verbose)
            if "recon_r2" in m:
                print("[sweep] <<< %s  recon_r2=%.3f control_pass=%.2f amp_ratio=%.2f shape_keep=%.3f" % (
                    r["run_id"], m["recon_r2"], m["control_pass"], m["amp_ratio"], m["shape_keep"]), flush=True)
            else:
                print("[sweep] <<< %s  %.1f s/epoch (steady), batch %d, xla %s" % (
                    r["run_id"], m["sec_per_epoch_steady"], m["batch_size"], m["xla"]), flush=True)
            if (rd / "error.txt").exists():
                (rd / "error.txt").unlink()
        except Exception:
            with open(rd / "error.txt", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            print("[sweep] !!! %s failed, see %s" % (r["run_id"], rd / "error.txt"), flush=True)
        finally:
            if (rd / "RUNNING").exists():
                (rd / "RUNNING").unlink()
    print("[sweep] worker %d finished stage %s" % (w, stage), flush=True)


# ------------------------------------------------------------------ status / collect / rank (no TensorFlow)
def _parse_log(path):
    rows = []
    if not path.exists():
        return pd.DataFrame()
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or "{" not in line:
            continue
        ep = line.split(" ", 1)[0]
        # RelLossWeight writes a dict literal, e.g. {'val_OrigRecLoss': np.float32(0.0082), ...} (or plain floats)
        d = {k: float(v) for k, v in _LOG_ITEM.findall(line[line.index("{"):])}
        if not ep.isdigit() or not d:
            continue
        rows.append({"epoch": int(ep), **d, "saved": " saved " in (" " + line + " ") and "not saved" not in line})
    return pd.DataFrame(rows)


def status(stage):
    _, scfg = load_cfgs()
    plan = load_plan(stage, scfg)
    sd = stage_dir(scfg, stage)
    rows = []
    for i, r in enumerate(plan):
        rd = sd / "runs" / r["run_id"]
        log = _parse_log(sd / "Logs" / r["run_id"] / "Logs_model.txt")
        state = ("done" if _done(rd) else "error" if (rd / "error.txt").exists()
                 else "running" if (rd / "RUNNING").exists() else "pending")
        info = json.load(open(rd / "RUNNING", encoding="utf-8")) if state == "running" else {}
        if state == "done":
            info = {"host": json.load(open(rd / "metrics.json", encoding="utf-8")).get("host", "")}
        rows.append({"#": i, "run_id": r["run_id"], "state": state, "host": info.get("host", ""),
                     "gpu": info.get("gpu", ""), "epochs": int(log["epoch"].max()) if len(log) else 0,
                     "max_epochs": r["epochs"],
                     "last_val_OrigRec": float(log["val_OrigRecLoss"].iloc[-1]) if len(log) else np.nan,
                     "best_val_OrigRec": float(log["val_OrigRecLoss"].min()) if len(log) else np.nan})
    df = pd.DataFrame(rows)
    print(df["state"].value_counts().to_dict())
    return df


def collect(stages, scfg=None):
    scfg = scfg or load_cfgs()[1]
    stages = [stages] if isinstance(stages, str) else list(stages)
    rows = []
    for stage in stages:
        runs_dir = stage_dir(scfg, stage) / "runs"
        if not runs_dir.exists():
            continue
        for rd in sorted(runs_dir.iterdir()):
            if not _done(rd):
                continue
            p = json.load(open(rd / "params.json", encoding="utf-8"))
            m = json.load(open(rd / "metrics.json", encoding="utf-8"))
            run = p["run"]
            rows.append({"stage": stage, "run_id": run["run_id"], "config_name": run["config_name"],
                         "seed": run["seed"], "overrides_json": json.dumps(run["overrides"], sort_keys=True),
                         **{k: v for k, v in run["overrides"].items() if not isinstance(v, dict)},
                         "data": "MULTI" if "circascope2021" in run["overrides"].get(
                             "SourceWeights", p["model_config"].get("SourceWeights", {})) else "POP",
                         **m})
    return pd.DataFrame(rows)


def bench_table(stage="bench"):
    """Throughput of the bench stage: seconds per epoch (steady state) and training windows per second,
    relative to the first setting in the plan."""
    _, scfg = load_cfgs()
    rows = []
    for r in load_plan(stage, scfg):
        rd = run_dir(scfg, stage, r["run_id"])
        if not _done(rd):
            rows.append({"setting": r["config_name"], "state": "error" if (rd / "error.txt").exists() else "pending"})
            continue
        m = json.load(open(rd / "metrics.json", encoding="utf-8"))
        h = pd.read_csv(rd / "history.csv")
        rows.append({"setting": r["config_name"], "state": "done", "host": m.get("host", ""),
                     "batch": m["batch_size"], "xla": m["xla"], "steps_per_epoch": -(-m["windows_per_epoch"] // m["batch_size"]),
                     "first_epoch_s": float(h["sec"].iloc[0]) if "sec" in h else np.nan,
                     "sec_per_epoch": m["sec_per_epoch_steady"],
                     "windows_per_s": round(m["windows_per_epoch"] / m["sec_per_epoch_steady"], 1),
                     "val_OrigRec_last": float(h["val_OrigRecLoss"].iloc[-1])})
    df = pd.DataFrame(rows)
    if "sec_per_epoch" in df and df["sec_per_epoch"].notna().any():
        ref = df["sec_per_epoch"].dropna().iloc[0]
        df["speedup_vs_first"] = (ref / df["sec_per_epoch"]).round(2)
    return df


def rank(df, scfg=None, by_config=True):
    """Rank-sum over the pre-specified label-free criteria (all higher = better). With by_config, seeds of the
    same configuration are averaged first (mean) and the seed count / sd are reported."""
    scfg = scfg or load_cfgs()[1]
    if df.empty:
        return df
    crit = scfg["selection"]["criteria"]
    if by_config:
        num = df.select_dtypes("number").columns.difference(["seed"])
        g = df.groupby(["config_name", "overrides_json"], as_index=False)
        agg = g[list(num)].mean()
        agg["n_seeds"] = g.size()["size"].to_numpy()
        for c in crit:
            agg[c + "_sd"] = g[c].std()[c].to_numpy()
        agg["stages"] = g["stage"].agg(lambda s: ",".join(sorted(set(s))))["stage"].to_numpy()
        agg["data"] = g["data"].first()["data"].to_numpy()
        df = agg
    r = pd.DataFrame({c: df[c].rank(ascending=False) for c in crit})
    df = df.assign(rank_sum=r.sum(1)).sort_values(["rank_sum", "recon_r2"], ascending=[True, False])
    return df.reset_index(drop=True)


def factors(df):
    """Swept parameters that take more than one value among the finished runs."""
    skip = {"seed", "host"}
    cand = ["data"] + [c for c in df.columns if c[:1].isupper() and c not in skip]
    return [c for c in cand if c in df and df[c].astype(str).nunique() > 1]


def best_yaml(ranked, k=1, prefix="SKZFC_CIRC_BEST"):
    """YAML text for the top-k configurations, to paste under vabam.Models in configs/circadian_vabam.yml
    (then: run.py train / select / external with that model name)."""
    cfg, scfg = load_cfgs()
    base = cfg["vabam"]["Models"][scfg["base_model"]]
    out = []
    for i, r in ranked.head(k).iterrows():
        mc = {**base, **scfg.get("common_overrides", {}), **json.loads(r["overrides_json"])}
        body = yaml.safe_dump({"%s%d" % (prefix, i + 1): mc},
                              sort_keys=False, default_flow_style=None)
        out.append("# %s  (rank_sum %.1f, recon_r2 %.3f, control_pass %.2f)\n%s" % (
            r["config_name"], r["rank_sum"], r["recon_r2"], r["control_pass"], body))
    txt = "\n".join(out)
    print(txt)
    return txt


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--freeze", action="store_true", help="freeze the plan of this stage")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--commands", action="store_true")
    ap.add_argument("--bench-table", action="store_true", help="print the throughput table of the bench stage")
    ap.add_argument("--verbose", type=int, default=2)
    a = ap.parse_args(argv)
    if a.freeze:
        print(freeze_plan(a.stage)[["run_id", "overrides", "seed", "epochs"]].to_string(index=False))
    elif a.status:
        print(status(a.stage).to_string(index=False))
    elif a.commands:
        commands(a.stage)
    elif a.bench_table:
        print(bench_table(a.stage).to_string(index=False))
    elif a.worker is not None:
        # stages with a fixed grid (no 'from') can be frozen automatically by the first worker
        scfg = load_cfgs()[1]
        if "from" not in scfg["stages"][a.stage] and not (stage_dir(scfg, a.stage) / "plan.json").exists():
            freeze_plan(a.stage)
        worker(a.stage, a.worker, a.nworkers, a.gpu, a.verbose)
        if a.stage == "bench" and a.nworkers == 1:
            print(bench_table(a.stage).to_string(index=False))
    else:
        ap.error("give --worker, --freeze, --status or --commands")


if __name__ == "__main__":
    main()
