"""Download the public data used by exp02 into data/raw/external/ (about 1.4 GB; skips files already present).

Sources
  * Authors' raw cisplatin dose-response imaging (Ector et al. MSB 2025, GitHub Granada-Lab, MIT licence)
  * cBioPortal datahub (GitHub): CCLE 2019 RNA-seq RPKM + GDSC drug AUC, TCGA-BRCA PanCancer Atlas, METABRIC
"""
import sys
import urllib.request
from pathlib import Path

from .common import EXT, inside

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
