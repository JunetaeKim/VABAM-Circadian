"""
Loaders for LumiCycle bioluminescence recordings (Ector et al. 2024 Nat Commun; 2025 Mol Syst Biol).

Every loader returns a ``TraceSet``:
    time   : np.ndarray (T,)       hours since the start of the recording
    values : pd.DataFrame (T, N)   one column per trace, column name = trace_id
    meta   : pd.DataFrame (N, ...) indexed by trace_id with columns
             source, date, cell_line, genotype, reporter, reset, replicate, label

Verified formats (opened and inspected on 2026-11-16)
-----------------------------------------------------
MSB 2025, ``raw_lumicycle_data.xlsx`` (Figshare item 28212131)
    sheets: 'raw_truncated' (5.0-142.67 h, 827 rows, dt = 10 min),
            'raw_full' (0-206 h, trace lengths differ; NaN padded),
            'legend'  (column-name convention)
    column 0 = 'Time' [h]; other columns = traces, e.g.
        20210525_HCC1143_Bmal1_Luc_1, 20211102_MDAMB468_Per2_1,
        20220705_U2OS_WT_BLH_1, 20220628_U2OS_CRY1_sKO_BLH_1,
        20220628_U2OS_CRY1_2_dKO_BLH_1, 20210525_CAL51_Bmal1_Luc_15uMDe,
        20210525_HCC38_Bmal1_Luc_serumshock
    BLH = Bmal1-Luc. U2OS knockouts carry the Bmal1 reporter only.
MSB 2025, ``processed_lumicycle_data*.xlsx``
    sheets 'detrended', 'normalized', 'envelope' (no time column; rows align
    with 'raw_truncated'), 'mra', 'maxpower'. Produced with pyBOAT,
    sinc detrending T_c = 48 h, envelope window 48 h.

Nat Commun 2024, ``circadian_time_series_data.xlsx`` (Figshare item 24297496; verified 2026-11-16)
    sheets 'raw', 'detrended', 'samples'. 135 of 164 traces are the same recordings as MSB 2025
    (time axis shifted by 6 h); 29 are new (GIMEN, SH-SY5Y). See load_natcomm2024 and
    src/audit_duplicates.py.
"""
import argparse
import re
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

SAMPLING_MIN = 10.0  # LumiCycle sampling interval [min]

_REPORTER_ALIASES = {"bmal1": "Bmal1", "blh": "Bmal1", "per2": "Per2", "plh": "Per2"}
_GENOTYPE_ALIASES = {
    "wt": "WT",
    "cry1_sko": "CRY1_sKO",
    "cry2_sko": "CRY2_sKO",
    "cry1_2_dko": "CRY1_2_dKO",
    "cry12_dko": "CRY1_2_dKO",
    "cry1/2_dko": "CRY1_2_dKO",
}
_NAME_RE = re.compile(
    r"^(?:(?P<date>\d{8})_)?(?P<body>.+?)_(?P<reporter>Bmal1|BMAL1|Per2|PER2|BLH|PLH)(?:_Luc)?(?:_(?P<tail>.+))?$"
)


@dataclass
class TraceSet:
    time: np.ndarray
    values: pd.DataFrame
    meta: pd.DataFrame

    @property
    def dt_h(self) -> float:
        return float(np.nanmedian(np.diff(self.time)))

    def select(self, cell_line=None, reporter=None, genotype=None, reset=None, ids=None) -> "TraceSet":
        """Subset by metadata. Each argument may be a single value or a list."""
        m = self.meta
        keep = pd.Series(True, index=m.index)
        for col, val in (("cell_line", cell_line), ("reporter", reporter), ("genotype", genotype), ("reset", reset)):
            if val is None:
                continue
            vals = [val] if isinstance(val, str) else list(val)
            keep &= m[col].isin(vals)
        if ids is not None:
            keep &= m.index.isin(list(ids))
        idx = m.index[keep]
        return TraceSet(self.time, self.values.loc[:, idx], m.loc[idx])

    def summary(self) -> pd.DataFrame:
        return (
            self.meta.groupby(["cell_line", "genotype", "reporter"]).size().rename("n_traces").reset_index()
        )

    def __len__(self):
        return self.values.shape[1]


def parse_trace_name(name: str, source: str = "") -> Dict[str, Union[str, int, None]]:
    """Parse a LumiCycle column name into metadata (see module docstring for examples)."""
    m = _NAME_RE.match(str(name).strip())
    if m is None:
        raise ValueError("Unrecognised trace name: %r" % name)
    body = m.group("body")
    parts = body.split("_")
    cell_line = parts[0]
    geno_raw = "_".join(parts[1:]) if len(parts) > 1 else "WT"
    genotype = _GENOTYPE_ALIASES.get(geno_raw.lower(), geno_raw)

    tail = m.group("tail") or ""
    replicate, reset = None, "Dex_1uM"
    for tok in [t for t in tail.split("_") if t]:
        if tok.isdigit():
            replicate = int(tok)
        elif tok.lower() in ("15umde", "15dex", "15umdex", "15de"):
            reset = "Dex_1.5uM"
        elif tok.lower() in ("1umde", "1dex", "1umdex", "1de"):
            reset = "Dex_1uM"
        elif tok.lower() == "serumshock":
            reset = "serum_shock"
        else:
            reset = tok
    reporter = _REPORTER_ALIASES[m.group("reporter").lower()]
    label = cell_line if genotype == "WT" else "%s_%s" % (cell_line, genotype)
    return {
        "source": source,
        "date": m.group("date"),
        "cell_line": cell_line,
        "genotype": genotype,
        "reporter": reporter,
        "reset": reset,
        "replicate": replicate,
        "label": label,
    }


def _build_meta(names: Iterable[str], source: str) -> pd.DataFrame:
    rows = []
    for n in names:
        d = parse_trace_name(n, source)
        d["trace_id"] = str(n)
        rows.append(d)
    return pd.DataFrame(rows).set_index("trace_id")


def load_msb2025_raw(path, sheet: str = "raw_truncated", drop_incomplete: bool = False) -> TraceSet:
    """Raw LumiCycle counts from the MSB 2025 dataset.

    sheet='raw_truncated' (default) is the authors' common window 5-142.67 h,
    identical in length for all 180 traces. sheet='raw_full' keeps 0-206 h with
    NaN padding (lengths differ between experiments).
    Traces with missing samples are kept by default (one MSB trace,
    20211102_MDAMB468_Per2_2, stops at 134.7 h); preprocess.py uses only the
    finite part of each trace. Set drop_incomplete=True to remove them.
    """
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if "Time" not in df.columns:
        raise ValueError("Expected a 'Time' column in sheet %r of %s" % (sheet, path))
    df = df[df["Time"].notna()].reset_index(drop=True)
    time = df["Time"].to_numpy(dtype=float)
    values = df.drop(columns="Time").astype(float)
    bad = values.columns[values.isna().any()]
    if drop_incomplete and len(bad):
        warnings.warn("Dropping %d traces with missing values: %s" % (len(bad), list(bad)))
        values = values.drop(columns=bad)
    meta = _build_meta(values.columns, "msb2025")
    values.columns = meta.index
    return TraceSet(time, values, meta)


def load_msb2025_processed(path, sheet: str = "detrended", time: Optional[np.ndarray] = None) -> TraceSet:
    """Authors' pyBOAT-processed signals (no time column; rows align with 'raw_truncated')."""
    values = pd.read_excel(path, sheet_name=sheet, engine="openpyxl").astype(float)
    values = values.loc[values.notna().any(axis=1)].reset_index(drop=True)
    if time is None:
        time = 5.0 + np.arange(len(values)) * SAMPLING_MIN / 60.0
    meta = _build_meta(values.columns, "msb2025_" + sheet)
    values.columns = meta.index
    return TraceSet(np.asarray(time, float), values, meta)


NATCOMM2024_TIME_OFFSET_H = 6.0
# Verified 2026-11-16 (src/audit_duplicates.py): 135 of 164 traces in circadian_time_series_data.xlsx are
# value-identical (max abs diff 0) to MSB 2025 raw_full traces once the NC time axis is shifted by +6.0 h,
# i.e. NC 'Time' = 0 is 6 h after the start of the LumiCycle recording. Adding the offset puts both sources on
# the same "hours since start of recording" axis.


def load_natcomm2024(path, sheet: str = "raw", keep_audit: Optional[str] = None) -> TraceSet:
    """circadian_time_series_data.xlsx (Nat Commun 2024, Figshare 24297496). Format verified 2026-11-16:
    sheets 'raw' (Time 0-120 h, 721 rows, 174 columns incl. 10 pandas-duplicated MCF7 headers '*.1'),
    'detrended', 'samples' (Short Name -> Experiment ID in the MSB naming convention).

    trace_id = Experiment ID (e.g. 20220614_GIMEN_Bmal1_Luc_1); 'Neuroblastoma_' prefixes are dropped.
    One Experiment ID occurs twice (SY5Y_BMAL1_1 and _2 -> 20220614_SY5Y_Bmal1_Luc_2, different values);
    duplicates get the suffix '__<short name>'.
    keep_audit: CSV from src/audit_duplicates.py; only traces whose status != 'same_recording' are kept.
    """
    xls = pd.ExcelFile(path, engine="openpyxl")
    df = pd.read_excel(xls, sheet_name=sheet)
    df = df[df["Time"].notna()].reset_index(drop=True)
    smp = pd.read_excel(xls, sheet_name="samples")
    smp.columns = ["short", "experiment"]
    smp = smp[smp["short"].isin(df.columns)]
    if keep_audit is not None:
        aud = pd.read_csv(keep_audit)
        keep = set(aud.loc[aud["status"] != "same_recording", "nc_short"])
        smp = smp[smp["short"].isin(keep)]
    exp = smp["experiment"].str.replace("Neuroblastoma_", "", regex=False)
    dup = exp.duplicated(keep=False)
    ids = np.where(dup, exp + "__" + smp["short"], exp)
    rows = []
    for tid, e, short in zip(ids, exp, smp["short"]):
        d = parse_trace_name(e, "natcomm2024")
        d.update(trace_id=tid, nc_short=short)
        rows.append(d)
    meta = pd.DataFrame(rows).set_index("trace_id")
    values = df[smp["short"].tolist()].astype(float)
    values.columns = meta.index
    time = df["Time"].to_numpy(float) + NATCOMM2024_TIME_OFFSET_H
    return TraceSet(time, values, meta)


def load_trace_sets(cfg: dict, root=".", sources: Optional[List[str]] = None) -> List[TraceSet]:
    """One TraceSet per source (sources may have different time grids). `sources` defaults to
    cfg['preprocess']['sources'] or, if absent, every source with enabled: true."""
    import os

    names = sources or cfg.get("preprocess", {}).get("sources") or [
        n for n, sp in cfg["data"]["sources"].items() if sp.get("enabled", True)]
    sets: List[TraceSet] = []
    for name in names:
        spec = cfg["data"]["sources"][name]
        path = os.path.join(root, spec["path"])
        if not os.path.exists(path):
            raise FileNotFoundError("Source %s: %s not found" % (name, path))
        if spec["kind"] == "msb2025_raw":
            sets.append(load_msb2025_raw(path, sheet=spec.get("sheet", "raw_truncated")))
        elif spec["kind"] == "natcomm2024":
            audit = spec.get("keep_audit")
            sets.append(load_natcomm2024(path, sheet=spec.get("sheet", "raw"),
                                         keep_audit=os.path.join(root, audit) if audit else None))
        else:
            raise ValueError("Unknown source kind %r" % spec["kind"])
    ids = pd.concat([s.meta for s in sets]).index
    if ids.duplicated().any():
        raise ValueError("Duplicate trace ids across sources: %s" % list(ids[ids.duplicated()][:5]))
    return sets


def load_traces(cfg: dict, root=".", sources: Optional[List[str]] = None) -> TraceSet:
    """Single TraceSet; only valid when all selected sources share one time grid (e.g. MSB alone)."""
    sets = load_trace_sets(cfg, root, sources)
    if len(sets) == 1:
        return sets[0]
    t0 = sets[0].time
    for s in sets[1:]:
        if len(s.time) != len(t0) or not np.allclose(s.time, t0, atol=1e-3):
            raise ValueError("Sources have different time grids; use load_trace_sets().")
    return TraceSet(t0, pd.concat([s.values for s in sets], axis=1), pd.concat([s.meta for s in sets]))


def _inspect(path: str):
    xls = pd.ExcelFile(path, engine="openpyxl")
    print("file:", path)
    for s in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=s, header=None)
        print("  sheet %-14s shape=%s first header cells=%s" % (s, df.shape, df.iloc[0, :4].tolist()))
    if "raw_truncated" in xls.sheet_names:
        ts = load_msb2025_raw(path)
        print("\nloaded %d traces, %d samples, t=%.2f..%.2f h, dt=%.4f h" % (
            len(ts), len(ts.time), ts.time[0], ts.time[-1], ts.dt_h))
        print(ts.summary().to_string(index=False))
        print("resetting:", ts.meta["reset"].value_counts().to_dict())
        nan_tr = ts.values.columns[ts.values.isna().any()]
        for c in nan_tr:
            print("incomplete trace %s: last finite t=%.2f h" % (c, ts.time[ts.values[c].notna().to_numpy()][-1]))
    elif "samples" in xls.sheet_names:
        ts = load_natcomm2024(path)
        print("\nloaded %d traces, %d samples, t=%.2f..%.2f h (after +%.0f h offset)" % (
            len(ts), len(ts.time), ts.time[0], ts.time[-1], NATCOMM2024_TIME_OFFSET_H))
        print(ts.summary().to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inspect a LumiCycle Excel file.")
    ap.add_argument("--inspect", required=True)
    _inspect(ap.parse_args().inspect)
