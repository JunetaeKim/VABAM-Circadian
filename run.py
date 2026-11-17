"""
Command-line entry point (runs from any working directory).

    python run.py <step> [--config CFG] [--model NAME] [step options]

Steps
    preprocess   detrend, normalise and window the MSB 2025 recordings (configuration A)
    multi        build the multi-source training pool and external sets (configuration B)
    filters      frequency response of the VABAM filters for the chosen FcLimit
    train        train a VABAM model from configs/circadian_vabam.yml
    exp01        U-2 OS wild type vs CRY1 KO vs CRY1/2 dKO
    all          train, then exp01
    select       label-free model selection
    external     label-free evaluation on validation and external sets
    audit        duplicate audit, Nat Commun 2024 vs MSB 2025
    sweep        parameter search (configs/sweep.yml)

--model defaults to default_model in the config; remaining options are passed to the step.
Outputs are written under results/ and data/processed/ only.
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _call(module_main, prog, argv):
    old = sys.argv
    sys.argv = [prog] + argv
    try:
        module_main()
    finally:
        sys.argv = old


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["preprocess", "multi", "filters", "train", "exp01", "all", "select",
                                     "external", "audit", "sweep"])
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "circadian_vabam.yml"))
    ap.add_argument("--model", default=None)
    args, rest = ap.parse_known_args()

    import yaml

    cfg_path = os.path.abspath(args.config)
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    model = args.model or cfg.get("default_model", "SKZFC_CIRC_1_30_200")

    if args.step == "preprocess":
        from src.data import preprocess
        _call(preprocess.main, "preprocess", ["--config", cfg_path] + rest)
    elif args.step == "filters":
        from src.model import filter_response
        _call(filter_response.main, "filter_response", ["--config", cfg_path] + rest)
    elif args.step == "sweep":
        from src.model import sweep
        sweep.main(rest)
    elif args.step == "multi":
        from src.data import multisource
        _call(multisource.main, "multi", ["--config", cfg_path] + rest)
    elif args.step == "audit":
        from src.data import audit
        _call(audit.main, "audit", ["--config", cfg_path] + rest)
    elif args.step == "external":
        from src.analysis import external_eval
        models = [args.model] if args.model else []
        _call(external_eval.main, "external", ["--config", cfg_path] + (["--models"] + models if models else [])
              + rest)
    elif args.step == "select":
        from src.model import select_models
        _call(select_models.main, "select", ["--config", cfg_path] + ([] if args.model is None else
                                                                    ["--models", args.model]) + rest)
    elif args.step in ("train", "all"):
        from src.model import train
        _call(train.main, "train", ["--config", cfg_path, "--model", model] + rest)
    if args.step in ("exp01", "all"):
        from src.analysis import exp01_u2os_cry
        exp_rest = [a for a in rest if args.step == "exp01" or not a.startswith("--epochs")]
        if args.step == "all":  # train-only options are not passed on; start from a clean Keras session
            exp_rest = []
            import tensorflow as tf

            tf.keras.backend.clear_session()
        _call(exp01_u2os_cry.main, "exp01", ["--config", cfg_path, "--model", model] + exp_rest)


if __name__ == "__main__":
    main()
