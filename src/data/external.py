"""External data for exp02: download (fetch) and harmonised tables (tables).

fetch   Download the public data used by exp02 into data/raw/external/ (about 1.4 GB; skips files already present).
tables  python run_exp02.py tables  -> data/processed/external_v1/   (CPU, under a minute, < 1 GB memory)
          drug_response.csv        long table: source, cell_model, replicate, drug, measure, readout, experiment, value
                                   tod_*   time-of-day response of breast models (authors' values, Nat Commun 2024)
                                   dr_*    dose-response parameters (authors' fits, Nat Commun 2024)
                                   gdsc_auc  GDSC drug AUC of the catalogued models (cBioPortal CCLE 2019)
          model_expression.csv.gz  genes x cell models, log2(TPM+1), DepMap default profile of each catalogued model
          models.csv               cell models with their ids and which tables cover them
          cohorts/<name>_clinical.csv   one row per tumour sample with the same columns in every cohort:
                                   os_months, os_event, a second endpoint where available (pfs/rfs/efs, dss), age_years,
                                   subtype, stage and the cohort's own covariates
          cohorts/<name>_expr.npz  X (genes x samples, float16, log scale as published), genes, samples; written for
                                   the cohorts whose rows are gene symbols (TCGA-BRCA, METABRIC, SCAN-B). SEQC (RefSeq
                                   transcripts) and NRC (array probe sets) need an annotation table first.
        Needs data/catalog/models.csv (python run.py audit --catalog). The neuroblastoma time-of-day files are raw
        growth curves; their response values are not derived here.

Sources
  * Authors' raw cisplatin dose-response imaging (Ector et al. MSB 2025, GitHub Granada-Lab, MIT licence)
  * cBioPortal datahub (GitHub): CCLE 2019 RNA-seq RPKM + GDSC drug AUC, TCGA-BRCA PanCancer Atlas, METABRIC
"""
import gzip
import re
import sys
import urllib.request
from pathlib import Path


# shared paths; reads and writes stay inside the repository
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

GH = "https://raw.githubusercontent.com/Granada-Lab/Circadian-Subtypes-Breast-Cancer-Cells/main/" \
     "Figure4_drug_sensitivity_parameter_calculation/raw_cisplatin_data/"
CIS_LINES = ["BT549", "CAL51", "HCC1143", "HCC1428", "HCC1806", "HCC1937", "HCC38", "HCC70", "MCF10A", "MCF7",
             "MDAMB231", "MDAMB436", "MDAMB468", "SUM149PT", "T47D"]
CBIO = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/"
CBIO_FILES = [
    "ccle_broad_2019/data_mrna_seq_rpkm.txt",
    "ccle_broad_2019/data_drug_treatment_auc.txt",
    "brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt",
    "brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
    "brca_metabric/data_mrna_illumina_microarray.txt",
    "brca_metabric/data_clinical_patient.txt",
]


def get(url, dest):
    dest = inside(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print("  have", dest.relative_to(EXT))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print("  get ", dest.relative_to(EXT), flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    tmp.replace(dest)


# ----------------------------------------------------------------------------- harmonised tables
RAWD = EXT.parent
TOD_COLS = {"maxmimum range": "tod_max_range", "mean": "tod_mean", "stdev": "tod_sd", "coeffvar": "tod_cv",
            "AUC/24": "tod_auc24"}


def _model(name):
    """'MDA-MB-231' -> ('MDAMB231', ''), 'HCC1937_2' -> ('HCC1937', '2')."""
    base, _, rep = str(name).strip().partition("_")
    return re.sub(r"[^A-Za-z0-9]", "", base), rep


def drug_response(models):
    import pandas as pd

    rows = []
    f = RAWD / "natcommun2024_tod/fig4_time_of_day/ToD_maxrange_sorted.xlsx"
    if f.exists():
        x = pd.ExcelFile(f)
        for sheet in x.sheet_names:
            drug, readout = sheet.rsplit("_", 1)
            d = x.parse(sheet)
            for _, r in d.iterrows():
                m, rep = _model(r["cellline"])
                for col, meas in TOD_COLS.items():
                    if col in d and pd.notna(r[col]):
                        rows.append(dict(source="natcomm2024", cell_model=m, replicate=rep, drug=drug, measure=meas,
                                         readout=readout, experiment="", value=float(r[col]), file=f.name))
    for f in sorted((RAWD / "natcommun2024_tod/fig3_drug_sensitivity").glob("*DR_Parameters_*.xlsx")):
        k = re.match(r"(?:\d+_)?(\d{4,8})_DR_Parameters_(.+)$", f.stem)
        m, rep = _model(k.group(2))
        x = pd.ExcelFile(f)
        for sheet in x.sheet_names:
            s = re.match(r"parameters_+(CellNr|Conf)$", sheet)
            if not s:
                continue
            d = x.parse(sheet).set_index("valuenames")
            for drug in d.columns:
                for name, v in pd.to_numeric(d[drug], errors="coerce").dropna().items():
                    rows.append(dict(source="natcomm2024", cell_model=m, replicate=rep, drug=str(drug).strip(),
                                     measure="dr_" + str(name).replace("GRAOC", "AOC"), readout=s.group(1),
                                     experiment=k.group(1), value=float(v), file=f.name))
    f = EXT / "cbioportal/ccle_broad_2019/data_drug_treatment_auc.txt"
    if f.exists():
        ids = {i.split(";")[0]: m for m, i in zip(models.cell_model, models.ccle_id) if i}
        head = open(f, encoding="utf-8").readline().rstrip("\n").split("\t")
        d = pd.read_csv(f, sep="\t", usecols=["ENTITY_STABLE_ID"] + [c for c in head if c in ids])
        d = d.melt(id_vars="ENTITY_STABLE_ID", var_name="ccle_id", value_name="value").dropna()
        for r in d.itertuples(index=False):
            rows.append(dict(source="gdsc_ccle2019", cell_model=ids[r.ccle_id], replicate="", drug=r.ENTITY_STABLE_ID,
                             measure="gdsc_auc", readout="", experiment="", value=float(r.value), file=f.name))
    out = pd.DataFrame(rows)
    return out.drop_duplicates(["source", "cell_model", "replicate", "drug", "measure", "readout", "experiment", "value"])


def model_expression(models):
    """DepMap expression of the catalogued models: genes x models (default profile of each model)."""
    import numpy as np
    import pandas as pd

    d = EXT / "depmap"
    fs = sorted(d.glob("OmicsExpression*.csv"))
    ids = {i: m for m, i in zip(models.cell_model, models.depmap_id) if i}
    if not fs or not ids:
        return None
    cols = {}
    with open(fs[0], encoding="utf-8") as h:
        head = h.readline().rstrip("\n").split(",")
        n_meta = next(k for k, c in enumerate(head) if re.search(r"\(\d+\)$", c))
        i_model, i_def = head.index("ModelID"), head.index("IsDefaultEntryForModel")
        for line in h:                                      # one model profile per line; keep only what is needed
            meta = line.split(",", n_meta)
            if meta[i_model] in ids and meta[i_def] == "Yes":
                cols[ids[meta[i_model]]] = np.array(meta[n_meta].rstrip("\n").split(","), dtype=np.float32)
    genes = head[n_meta:]
    out = pd.DataFrame(cols, index=[g.split(" (")[0] for g in genes])
    out.insert(0, "entrez", [re.search(r"\((\d+)\)$", g).group(1) for g in genes])
    out.index.name = "gene"
    return out


def tables():
    import pandas as pd

    cat = ROOT / "data" / "catalog" / "models.csv"
    if not cat.exists():
        raise SystemExit("missing %s - run: python run.py audit --catalog" % cat)
    m = pd.read_csv(cat, keep_default_na=False)
    m = m[(m.genotype_variant.astype(str) == "False")]
    m = (m.groupby("cell_model").agg(cancer_type=("cancer_type", "first"), species=("species", "first"),
                                     lineage_group=("lineage_group", "first"), ccle_id=("ccle_id", "max"),
                                     depmap_id=("depmap_id", "max"), n_usable_records=("n_usable", "sum"))
         .reset_index())
    out = inside(PROC / "external_v1")
    out.mkdir(parents=True, exist_ok=True)
    dr = drug_response(m)
    dr.to_csv(out / "drug_response.csv", index=False)
    ex = model_expression(m)
    if ex is not None:
        ex.to_csv(out / "model_expression.csv.gz", float_format="%.4f")
    m["has_expression"] = m.cell_model.isin(ex.columns) if ex is not None else False
    for meas in ("tod_max_range", "dr_GR50", "gdsc_auc"):
        n = dr[dr.measure == meas].groupby("cell_model").drug.nunique()
        m["n_drugs_" + meas] = m.cell_model.map(n).fillna(0).astype(int)
    m.to_csv(out / "models.csv", index=False)
    print(m.to_string(index=False))
    print(dr.groupby(["source", "measure"]).agg(models=("cell_model", "nunique"), drugs=("drug", "nunique"),
                                                rows=("value", "size")).to_string())
    return dr, ex, m


# ----------------------------------------------------------------------------- patient cohorts
COH = EXT / "cohorts"
CBP = EXT / "cbioportal"
D2M = 30.4375                                        # days per month


def _geo_samples(path):
    """Sample table of a GEO series matrix: accession, title and every 'key: value' characteristic."""
    import pandas as pd

    acc, title, rows = [], [], []
    with gzip.open(path, "rt", errors="replace") as h:
        for line in h:
            if line.startswith("!series_matrix_table_begin"):
                break
            v = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")]
            if v[0] == "!Sample_geo_accession":
                acc = v[1:]
            elif v[0] == "!Sample_title":
                title = v[1:]
            elif v[0] == "!Sample_characteristics_ch1":
                rows.append(v[1:])
    rec = [dict(gsm=a, title=t) for a, t in zip(acc, title)]
    for r in rows:                                    # fields are not aligned across samples: read key by key
        for i, x in enumerate(r):
            if ": " in x:
                k, val = x.split(": ", 1)
                rec[i][k.strip()] = val.strip()
    return pd.DataFrame(rec)


def cohort_clinical():
    """{cohort: DataFrame}. Times in months, events 0/1, age in years; cohort-specific covariates keep their meaning."""
    import pandas as pd

    num = lambda s: pd.to_numeric(s, errors="coerce")
    ev = lambda s: s.astype(str).str.startswith("1").astype(float).where(s.notna())
    out = {}
    f = CBP / "brca_tcga_pan_can_atlas_2018"
    if (f / "data_clinical_patient.txt").exists():
        c = pd.read_csv(f / "data_clinical_patient.txt", sep="\t", comment="#")
        smp = [x for x in open(f / "data_mrna_seq_v2_rsem.txt", encoding="utf-8").readline().rstrip("\n").split("\t")[2:]
               if x[13:15] == "01"]
        d = pd.DataFrame({"sample": smp, "patient": [x[:12] for x in smp]}).merge(c, left_on="patient",
                                                                                  right_on="PATIENT_ID", how="left")
        out["TCGA_BRCA"] = pd.DataFrame(dict(
            sample=d["sample"], patient=d.patient, cancer="breast", os_months=num(d.OS_MONTHS), os_event=ev(d.OS_STATUS),
            pfs_months=num(d.PFS_MONTHS), pfs_event=ev(d.PFS_STATUS), dss_months=num(d.DSS_MONTHS),
            dss_event=ev(d.DSS_STATUS), age_years=num(d.AGE), subtype=d.SUBTYPE.astype(str).str.replace("BRCA_", ""),
            stage=d.AJCC_PATHOLOGIC_TUMOR_STAGE))
    f = CBP / "brca_metabric"
    if (f / "data_clinical_patient.txt").exists():
        c = pd.read_csv(f / "data_clinical_patient.txt", sep="\t", comment="#")
        smp = open(f / "data_mrna_illumina_microarray.txt", encoding="utf-8").readline().rstrip("\n").split("\t")[2:]
        d = pd.DataFrame({"sample": smp}).merge(c, left_on="sample", right_on="PATIENT_ID", how="left")
        out["METABRIC"] = pd.DataFrame(dict(
            sample=d["sample"], patient=d["sample"], cancer="breast", os_months=num(d.OS_MONTHS), os_event=ev(d.OS_STATUS),
            rfs_months=num(d.RFS_MONTHS), rfs_event=ev(d.RFS_STATUS), dss_months=num(d.OS_MONTHS),
            dss_event=(d.VITAL_STATUS == "Died of Disease").astype(float).where(d.VITAL_STATUS.notna()),
            age_years=num(d.AGE_AT_DIAGNOSIS), subtype=d.CLAUDIN_SUBTYPE, er=d.ER_IHC, her2=d.HER2_SNP6,
            chemotherapy=d.CHEMOTHERAPY, hormone_therapy=d.HORMONE_THERAPY, npi=num(d.NPI)))
    fs = sorted((COH / "GSE96058_SCANB_breast").glob("*series_matrix.txt.gz"))
    if fs:
        d = pd.concat([_geo_samples(p).assign(platform=p.name.split("-")[1].split("_")[0]) for p in fs], ignore_index=True)
        out["SCANB"] = pd.DataFrame(dict(
            sample=d.title, patient=d.title.str.replace("repl", "", regex=False), cancer="breast",
            is_replicate=d.title.str.endswith("repl"), os_months=num(d["overall survival days"]) / D2M,
            os_event=num(d["overall survival event"]), age_years=num(d["age at diagnosis"]), subtype=d["pam50 subtype"],
            er=d["er status"], pgr=d["pgr status"], her2=d["her2 status"], ki67=d["ki67 status"], grade=d["nhg"],
            node_status=d["lymph node status"], tumor_size_mm=num(d["tumor size"]),
            endocrine_treated=d["endocrine treated"], chemo_treated=d["chemo treated"], platform=d.platform, gsm=d.gsm))
    f = COH / "GSE62564_SEQC_neuroblastoma/GSE62564_series_matrix.txt.gz"
    if f.exists():
        d = _geo_samples(f)
        out["SEQC_NB"] = pd.DataFrame(dict(
            sample=d.title.str.split(" ").str[0], patient=d.title.str.split(" ").str[0], cancer="neuroblastoma",
            os_months=num(d["os day"]) / D2M, os_event=num(d["os bin"]), efs_months=num(d["efs day"]) / D2M,
            efs_event=num(d["efs bin"]), age_years=num(d["age"]) / 365.25, sex=d["Sex"],
            high_risk=(d["high risk"] == "HR") if "high risk" in d else None, split=d["training/validation"], gsm=d.gsm))
    f = COH / "GSE85047_NRC_neuroblastoma/GSE85047_series_matrix.txt.gz"
    if f.exists():
        d = _geo_samples(f)
        yes = lambda s: s.astype(str).str.lower().map({"yes": 1.0, "no": 0.0, "1": 1.0, "0": 0.0})
        out["NRC_NB"] = pd.DataFrame(dict(
            sample=d.gsm, patient=d.gsm, cancer="neuroblastoma",
            os_months=num(d["overall_survival_time(days)"]) / D2M, os_event=yes(d["event_overall"]),
            pfs_months=num(d["profression_free_survival_time(days)"]) / D2M, pfs_event=yes(d["event_progression_free"]),
            age_years=num(d["age_at_diagnosis(days)"]) / 365.25, stage=d["inss"], mycn_amp=d["mycn_amplification"],
            title=d.title))
    return out


def cohort_expression(name, out):
    """Gene-symbol matrices as published (log scale), float16, written to <out>/<name>_expr.npz."""
    import numpy as np
    import pandas as pd

    if name in ("TCGA_BRCA", "METABRIC"):
        p = CBP / ("brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt" if name == "TCGA_BRCA"
                   else "brca_metabric/data_mrna_illumina_microarray.txt")
        head = open(p, encoding="utf-8").readline().rstrip("\n").split("\t")
        X = pd.read_csv(p, sep="\t", dtype={c: np.float32 for c in head[2:]})
        ent = X["Entrez_Gene_Id"]
        X = X.dropna(subset=["Hugo_Symbol"])
        entrez = X.groupby("Hugo_Symbol")["Entrez_Gene_Id"].first()
        X = X.drop(columns="Entrez_Gene_Id").groupby("Hugo_Symbol").mean()
        if name == "TCGA_BRCA":
            X = np.log2(X.clip(lower=0) + 1)
        genes, smp, M = X.index.to_numpy(str), X.columns.to_numpy(str), X.to_numpy(np.float32)
        extra = dict(entrez=entrez.reindex(X.index).fillna(-1).to_numpy(np.int64),
                     scale="log2(RSEM+1)" if name == "TCGA_BRCA" else "log2 intensity")
    elif name == "SCANB":
        p = next((COH / "GSE96058_SCANB_breast").glob("GSE96058_gene_expression*.csv.gz"))
        genes, rows = [], []
        with gzip.open(p, "rt") as h:                    # row by row: the file is 0.6 GB compressed
            smp = np.array([x.strip('"') for x in h.readline().rstrip("\n").split(",")[1:]])
            for line in h:
                g, _, rest = line.partition(",")
                genes.append(g.strip('"'))
                rows.append(np.array(rest.split(","), dtype=np.float32).astype(np.float16))
        genes, M = np.array(genes), np.vstack(rows)
        extra = dict(scale="log2(FPKM+0.1)")
    else:
        return None
    np.savez_compressed(out / (name + "_expr.npz"), X=M.astype(np.float16), genes=genes, samples=smp, **extra)
    return M.shape


def cohorts(expression=("TCGA_BRCA", "METABRIC", "SCANB")):
    import pandas as pd

    out = inside(PROC / "external_v1" / "cohorts")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, d in cohort_clinical().items():
        d.to_csv(out / (name + "_clinical.csv"), index=False)
        ok = d.os_months.notna() & d.os_event.notna() & (d.os_months > 0)
        if "is_replicate" in d:
            ok &= ~d.is_replicate
        shape = cohort_expression(name, out) if name in expression else None
        rows.append(dict(cohort=name, cancer=d.cancer.iloc[0], samples=len(d), patients=d.patient.nunique(),
                         with_os=int(ok.sum()), os_events=int(d.os_event[ok].sum()),
                         median_followup_months=round(float(d.os_months[ok].median()), 1),
                         second_endpoints=",".join(c[:-7] for c in d.columns if c.endswith("_months") and c != "os_months"),
                         expr_genes=shape[0] if shape else None, expr_file=(name + "_expr.npz") if shape else ""))
        print(rows[-1], flush=True)
    s = pd.DataFrame(rows)
    s.to_csv(out / "cohorts.csv", index=False)
    return s


def main():
    print("Downloading to", EXT)
    for l in CIS_LINES:
        get(GH + "2021_DR_%s.xlsx" % l, EXT / "msb2025_cisplatin" / ("2021_DR_%s.xlsx" % l))
    for f in CBIO_FILES:
        get(CBIO + f, EXT / "cbioportal" / f)
    (EXT / "SOURCES.md").write_text(
        "# External data for exp02\n\n"
        "- msb2025_cisplatin/: raw cisplatin dose-response imaging, Ector et al. Mol Syst Biol 2025;21:315, "
        "github.com/Granada-Lab/Circadian-Subtypes-Breast-Cancer-Cells (MIT)\n"
        "- cbioportal/: github.com/cBioPortal/datahub (ccle_broad_2019, brca_tcga_pan_can_atlas_2018, "
        "brca_metabric)\n", encoding="utf-8")
    print("done")


if __name__ == "__main__":
    sys.exit(main())
