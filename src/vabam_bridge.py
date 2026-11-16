"""
Bridge to the official VABAM implementation. The official code is imported, never modified,
and nothing is ever written into its folder (or anywhere outside this repository).

Write paths that VABAM itself would touch, and how they are redirected here:
  * Python bytecode cache (__pycache__ next to Models/*.py, Utilities/*.py)
        -> sys.dont_write_bytecode = True before any VABAM import
  * RelLossWeight (Utilities.Utilities) writes './Logs/<dir>/Logs_<model>.txt' relative to the
    current working directory and creates that folder
        -> training runs inside `working_dir(<repo>/results)`
  * LoadModelConfigs creates './Results/<SigType>/'
        -> not used; the model config dict is built from configs/*.yml here
  * RelLossWeight saves the model to SavePath
        -> SavePath is always under <repo>/results (checked by assert_inside_spinoff)
"""
import contextlib
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True                      # never create __pycache__ inside the official repo
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # TF >= 2.16 only: use tf-keras (Keras 2 API) as VABAM expects

SPINOFF_ROOT = Path(__file__).resolve().parents[1]


def assert_inside_spinoff(path) -> Path:
    """Raise if `path` resolves outside this repository. Use for every write."""
    p = Path(path).resolve()
    if p != SPINOFF_ROOT and SPINOFF_ROOT not in p.parents:
        raise PermissionError("Refusing to write outside %s: %s" % (SPINOFF_ROOT, p))
    return p


@contextlib.contextmanager
def working_dir(path):
    """chdir into a folder inside this repository (created if needed) and restore afterwards."""
    p = assert_inside_spinoff(path)
    p.mkdir(parents=True, exist_ok=True)
    old = os.getcwd()
    os.chdir(p)
    try:
        yield p
    finally:
        os.chdir(old)


def resolve_vabam_root(cfg: dict) -> Path:
    """Official repository: $VABAM_ROOT, else paths.vabam_root, else VABAM-official or Github/VABAM-official in a
    parent folder (up to three levels above this repository)."""
    cands = [os.environ.get("VABAM_ROOT"), cfg["paths"].get("vabam_root")]
    for anc in list(SPINOFF_ROOT.parents)[:3]:
        cands += [anc / "VABAM-official", anc / "Github" / "VABAM-official"]
    tried = []
    for c in [c for c in cands if c]:
        root = Path(c)
        root = root if root.is_absolute() else (SPINOFF_ROOT / root).resolve()
        tried.append(str(root))
        if (root / "Models" / "Caller.py").exists():
            break
    else:
        raise FileNotFoundError("Official VABAM code not found (expects Models/Caller.py). Clone "
                                "https://github.com/JunetaeKim/VABAM-official next to this repository or set "
                                "VABAM_ROOT. Tried: %s" % ", ".join(tried))
    if root == SPINOFF_ROOT or SPINOFF_ROOT in root.parents:
        raise ValueError("vabam_root must point to the official repository, not into this repository")
    return root


def import_vabam(cfg: dict):
    """Put the official repo first on sys.path and import the pieces we use. Returns a namespace."""
    root = resolve_vabam_root(cfg)
    for name in ("Models", "Utilities"):
        mod = sys.modules.get(name)
        if mod is not None and Path(getattr(mod, "__path__", [""])[0]).resolve().parent != root:
            raise ImportError("A different '%s' package is already imported from %s" % (name, mod.__path__))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from Models import Caller, MainModel  # noqa: E402  (official code)
    import types

    return types.SimpleNamespace(
        root=root,
        ModelCall=Caller.ModelCall,
        DCLCall=Caller.DCLCall,
        GenLowFilter=MainModel.GenLowFilter,
        GenHighFilter=MainModel.GenHighFilter,
    )


def model_config(cfg: dict, model_name: str) -> dict:
    """Same dict layout that VABAM's LoadModelConfigs returns: {'ConfigName', **Common_Param, **Model}."""
    v = cfg["vabam"]
    if model_name not in v["Models"]:
        raise KeyError("Model %r not in config; available: %s" % (model_name, list(v["Models"])))
    return {"ConfigName": model_name, **v["Common_Param"], **v["Models"][model_name]}


def results_dir(cfg: dict) -> Path:
    return assert_inside_spinoff(SPINOFF_ROOT / cfg["paths"]["results"])


def checkpoint_relpath(cfg: dict, model_name: str) -> str:
    """'./ckpt/<SigType>/<model>.hdf5', relative to results/.

    VABAM's RelLossWeight derives its log folder from SavePath.split('/')[2] and writes it relative to the
    working directory ('./Logs/<SigType>/Logs_<model>.txt'). Training therefore runs with cwd = results/ and
    this relative SavePath, which puts both the checkpoint and the log under results/.
    """
    sig = cfg["vabam"]["Common_Param"]["SigType"]
    return "./ckpt/%s/%s.hdf5" % (sig, model_name)


def checkpoint_path(cfg: dict, model_name: str) -> Path:
    """Absolute checkpoint path: <repo>/results/ckpt/<SigType>/<model>.hdf5."""
    return assert_inside_spinoff(results_dir(cfg) / checkpoint_relpath(cfg, model_name))


def build_model(cfg, model_name, sig_dim, data_size, reparam=True, load_weights=False, return_parts=False,
                summary=False):
    vb = import_vabam(cfg)
    mc = model_config(cfg, model_name)
    ck = checkpoint_path(cfg, model_name)
    out = vb.ModelCall(mc, sig_dim, data_size, Resume=False, LoadWeight=load_weights, ReturnModelPart=return_parts,
                       Reparam=reparam, ModelSaveName=str(ck), ModelSummary=summary)
    return out, mc, ck
