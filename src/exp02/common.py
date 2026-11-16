"""Shared paths for exp02. Reads and writes stay inside the repository."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repository root
EXT = ROOT / "data" / "raw" / "external"            # run_exp02.py fetch
PROC = ROOT / "data" / "processed"
RES = ROOT / "results" / "exp02"


def inside(p):
    p = Path(p).resolve()
    if p != ROOT and ROOT not in p.parents:
        raise PermissionError("Refusing to write outside %s: %s" % (ROOT, p))
    return p


def outdir(*parts):
    p = inside(RES.joinpath(*parts))
    p.mkdir(parents=True, exist_ok=True)
    return p
