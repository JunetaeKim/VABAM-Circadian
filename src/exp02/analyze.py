"""Downstream evaluation of one learned representation (VABAM z, or a baseline VAE z).

    python run_exp02.py analyze --rep VABAM --codes results/exp01_SKZFC_CIRC_1_30_200/codes_all_windows.npz
    python run_exp02.py analyze --rep BaseVAE_CIRC_30_s0          (codes taken from results/exp02/codes/)

Input: an .npz with key 'z' (n_windows x d), rows aligned with data/processed/windows_<tag>_meta.csv.
Output: results/exp02/<rep>/  (feature tables, cluster/drug/bridge/survival tables, figures, summary.json)

Feature recipe (identical for every representation; fixed before any drug/expression/survival data were seen):
  PCA of z over all windows (no labels), then per trace
    rep_logR     log mean radius in PC1-PC2                 rep_Rdecay  slope of log radius over window start
    rep_Tring    rotation period of the PC1-PC2 angle [h]   rep_jitter  SD of the angular step [rad]
    rep_logDisp  log RMS spread of windows around the trace mean (all dims)
    rep_PC1m..3m trace mean of PC1-3
  classical baseline: cosinor (16-36 h) amplitude/decay/period/period SD/R2, autocorrelation rhythmicity index
Cell line = median over replicate traces per reporter; reporter-averaged = mean of within-reporter z-scores.
"""
import argparse, json, re, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

from .common import ROOT, EXT, PROC, outdir

warnings.filterwarnings("ignore")
DT = 1 / 6.0
REP = ["rep_logR", "rep_Rdecay", "rep_Tring", "rep_jitter", "rep_logDisp", "rep_PC1m", "rep_PC2m", "rep_PC3m"]
RING = ["rep_logR", "rep_Rdecay", "rep_Tring", "rep_jitter"]
CL = ["cl_logAmp", "cl_ampDecay", "cl_period", "cl_periodSD", "cl_cosR2", "cl_ACF_RI"]
BREAST = ["BT549", "CAL51", "HCC1143", "HCC1428", "HCC1806", "HCC1937", "HCC38", "HCC70", "MCF10A", "MCF7",
          "MDAMB231", "MDAMB436", "MDAMB468", "SUM149PT", "T47D"]
# circadian phenotypes as coloured in the authors' code (Pipeline_paper2_revised_final.py, Fig 3C)
PHENO = {**{k: "Functional" for k in ["MCF7", "MCF10A", "MDAMB468", "T47D", "HCC1937", "HCC1143"]},
         **{k: "Unstable" for k in ["HCC1806", "MDAMB231", "CAL51"]},
         **{k: "Weak" for k in ["HCC1428", "MDAMB436", "BT549", "HCC70"]},
         **{k: "Dysfunctional" for k in ["HCC38", "SUM149PT"]}}
CLOCK = ["ARNTL", "CLOCK", "NPAS2", "CRY1", "CRY2", "PER1", "PER2", "PER3", "NR1D1", "NR1D2", "RORA", "RORC",
         "DBP", "TEF", "HLF", "NFIL3", "BHLHE40", "BHLHE41", "CSNK1D", "CSNK1E", "TIMELESS", "CIART"]


# ----------------------------------------------------------------------------------------------- features
def classical_window_features(Xn):
    t = np.arange(Xn.shape[1]) * DT
    Xc = Xn - Xn.mean(1, keepdims=True)
    sst = (Xc ** 2).sum(1)
    best = np.zeros((len(Xn), 3)); bestr2 = np.full(len(Xn), -1.0)
    for T in np.arange(16, 36.01, 0.1):
        C = np.c_[np.cos(2 * np.pi * t / T), np.sin(2 * np.pi * t / T)]
        beta = np.linalg.lstsq(C, Xc.T, rcond=None)[0]
        r2 = 1 - ((Xc.T - C @ beta) ** 2).sum(0) / sst
        u = r2 > bestr2
        bestr2[u] = r2[u]; best[u, 0] = T; best[u, 1] = np.hypot(beta[0], beta[1])[u]
    best[:, 2] = bestr2
    return best


def acf_ri(x):
    x = x[np.isfinite(x)]; x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:] / (np.sum(x ** 2) + 1e-12)
    return float(ac[int(16 / DT):int(36 / DT)].max())


def build_features(Z, tag):
    M = pd.read_csv(PROC / ("windows_%s_meta.csv" % tag))
    Xn = np.load(PROC / ("windows_%s.npz" % tag))["Xn"]
    TR = np.load(PROC / ("traces_%s.npz" % tag), allow_pickle=True)
    assert len(Z) == len(M), "codes (%d) and windows meta (%d) differ" % (len(Z), len(M))
    Zc = Z - Z.mean(0)
    _, s, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Zc @ Vt[:3].T
    M["PC1"], M["PC2"], M["PC3"] = P[:, 0], P[:, 1], P[:, 2]
    M["cos_period"], M["cos_amp"], M["cos_r2"] = classical_window_features(Xn).T
    full = dict(zip(TR["trace_id"], TR["Xn_full"])); lens = dict(zip(TR["trace_id"], TR["lengths"]))
    rows = []
    for tid, g in M.groupby("trace_id"):
        g = g.sort_values("t_start_h"); th = g.t_start_h.values
        r = np.hypot(g.PC1, g.PC2).values
        a = np.unwrap(np.arctan2(g.PC2, g.PC1).values)
        w = np.polyfit(th, a, 1)[0]
        zi = Z[g.index.values]
        la = np.log(g.cos_amp.values + 1e-6)
        rows.append(dict(
            trace_id=tid, cell_line=g.cell_line.iloc[0], genotype=g.genotype.iloc[0], reporter=g.reporter.iloc[0],
            date=str(g.date.iloc[0]), split=g.split.iloc[0],
            rep_logR=np.log(r.mean()), rep_Rdecay=np.polyfit(th, np.log(r + 1e-6), 1)[0],
            rep_Tring=min(2 * np.pi / abs(w), 200.0) if w != 0 else 200.0, rep_jitter=np.std(np.diff(a)),
            rep_logDisp=np.log(np.sqrt(((zi - zi.mean(0)) ** 2).sum(1).mean())),
            rep_PC1m=g.PC1.mean(), rep_PC2m=g.PC2.mean(), rep_PC3m=g.PC3.mean(),
            cl_logAmp=np.median(la), cl_ampDecay=np.polyfit(th, la, 1)[0], cl_period=np.median(g.cos_period),
            cl_periodSD=np.std(g.cos_period), cl_cosR2=np.median(g.cos_r2), cl_ACF_RI=acf_ri(full[tid][:lens[tid]])))
    F = pd.DataFrame(rows)
    F["line"] = np.where(F.cell_line == "U2OS", "U2OS_" + F.genotype, F.cell_line)
    L = F.groupby(["line", "reporter"])[REP + CL].median().unstack("reporter")
    L.columns = ["%s_%s" % c for c in L.columns]
    A = {}
    for f in REP + CL:
        v = [(L.loc[BREAST, "%s_%s" % (f, r)] - L.loc[BREAST, "%s_%s" % (f, r)].mean()) /
             L.loc[BREAST, "%s_%s" % (f, r)].std() for r in ("Bmal1", "Per2")]
        A[f] = pd.concat(v, axis=1).mean(1)
    window_check = float(pd.Series(np.hypot(M.PC1, M.PC2)).corr(M.cos_amp, method="spearman"))
    return F, L, pd.DataFrame(A), (s ** 2 / (s ** 2).sum())[:5], window_check


# ----------------------------------------------------------------------------------------------- clustering
def r2_groups(X, y):
    sst = ((X - X.mean(0)) ** 2).sum()
    return 1 - sum(((X[y == k] - X[y == k].mean(0)) ** 2).sum() for k in set(y)) / sst


def clustering(F, L, rng):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    lines = list(PHENO); y = np.array([PHENO[l] for l in lines])
    cols = lambda fs: ["%s_%s" % (f, r) for f in fs for r in ("Bmal1", "Per2")]
    out = []
    for name, c in {"rep": cols(REP), "rep_ring": cols(RING), "classical": cols(CL)}.items():
        X = StandardScaler().fit_transform(L.loc[lines, c].values)
        r2 = r2_groups(X, y)
        null = np.array([r2_groups(X, rng.permutation(y)) for _ in range(5000)])
        lab = AgglomerativeClustering(4, linkage="ward").fit_predict(X)
        ari = adjusted_rand_score(y, lab)
        nari = np.array([adjusted_rand_score(rng.permutation(y), lab) for _ in range(2000)])
        out.append(dict(features=name, phenotype_R2=r2, p_R2=(np.sum(null >= r2) + 1) / 5001, ARI_ward=ari,
                        p_ARI=(np.sum(nari >= ari) + 1) / 2001))
    # trace -> cell line on a held-out recording date
    B = F[F.cell_line != "U2OS"]
    nd = B.groupby("line").date.nunique(); B = B[B.line.isin(nd[nd >= 2].index)]
    ident = []
    for name, c in {"rep": REP, "classical": CL}.items():
        for rep in ("Bmal1", "Per2"):
            D = B[B.reporter == rep]; X, yy, dd = D[c].values, D.line.values, D.date.values
            ok_all = []
            for d in np.unique(dd):
                te = dd == d; ok = np.isin(yy[te], yy[~te])
                if ok.sum() == 0:
                    continue
                m = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
                m.fit(X[~te], yy[~te]); ok_all += list(m.predict(X[te][ok]) == yy[te][ok])
            ident.append(dict(features=name, reporter=rep, n_test=len(ok_all), acc=np.mean(ok_all),
                              chance=1 / D.line.nunique()))
    return pd.DataFrame(out), pd.DataFrame(ident)


# ----------------------------------------------------------------------------------------------- drugs
def cisplatin_gr():
    """GR metrics (Hafner 2016) from the authors' raw imaging, following DR_analysis_pipeline.m."""
    treat = {"HCC1428": 20, "MCF10A": 20, "T47D": 19}
    out = []
    for l in BREAST:
        f = EXT / "msb2025_cisplatin" / ("2021_DR_%s.xlsx" % l)
        sheets = pd.ExcelFile(f).sheet_names
        chan = "CellNr" if any(s.startswith("CellNr") for s in sheets) else "Conf"
        for sh in [s for s in sheets if s.startswith(chan)]:
            raw = pd.read_excel(f, sheet_name=sh, header=None)
            hr = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "Date Time"][0]
            hdr = raw.iloc[hr].astype(str).tolist()
            num = raw.iloc[hr + 1:].apply(pd.to_numeric, errors="coerce").dropna(how="all", subset=[1])
            el = num[1].values; t0 = treat.get(l, 1) - 1; t1 = t0 + int(round(96 / (el[4] - el[3])))
            doses = {}
            for j, h in enumerate(hdr):
                if j < 2 or "Std Err" in h:
                    continue
                m = re.search(r"([\d.]+)\s*(?:µM|uM|mg/ml)\s*\(", h)
                doses.setdefault(float(m.group(1)) if m else 0.0, j)

            def ep(j):
                y = pd.Series(num[j].values.astype(float)).rolling(5, center=True, min_periods=1).median().values
                seg = y[t0:t1 + 1]; return seg[0], np.median(seg[-4:])
            c0, c1 = ep(doses[0.0]); kc = np.log2(c1 / c0)
            for d, j in doses.items():
                if 0 < d <= 70:
                    x0, x1 = ep(j)
                    out.append(dict(line=l, sheet=sh, dose=d, GR=2 ** (np.log2(x1 / x0) / kc) - 1 if kc > 0 else np.nan))
    D = pd.DataFrame(out)
    S = D.groupby(["line", "sheet"]).apply(lambda g: pd.Series({
        "GR_AOC": np.mean(1 - g.GR.clip(-1, 1.5)), "GR_70": g.loc[g.dose.idxmax(), "GR"]}),
        include_groups=False).reset_index()
    return S.groupby("line")[["GR_AOC", "GR_70"]].mean()


def loo_ridge(X, y):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    pred = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        sc = StandardScaler().fit(X[tr])
        pred[i] = Ridge(alpha=5.0).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[i:i + 1]))[0]
    return spearmanr(pred, y)[0]


def drugs(A, rng, n_perm):
    G = cisplatin_gr()
    cis = []
    for f in A.columns:
        r, p = spearmanr(A.loc[G.index, f], G.GR_AOC); cis.append(dict(feature=f, rho=r, p=p, n=len(G)))
    for name, c in {"rep": REP, "classical": CL}.items():
        cis.append(dict(feature="LOO_ridge_" + name, rho=loo_ridge(A.loc[G.index, c].values, G.GR_AOC.values),
                        p=np.nan, n=len(G)))
    D = pd.read_csv(EXT / "cbioportal/ccle_broad_2019/data_drug_treatment_auc.txt", sep="\t").set_index("ENTITY_STABLE_ID")
    cols = {c.replace("_BREAST", ""): c for c in D.columns if c.endswith("_BREAST")}
    ov = [l for l in BREAST if l in cols]
    Y = D[[cols[l] for l in ov]].apply(pd.to_numeric, errors="coerce").T; Y.index = ov
    Y = Y.loc[:, Y.notna().sum() >= 10]

    def scores(X):
        return np.array([loo_ridge(X.loc[Y[d].dropna().index].values, Y[d].dropna().values) for d in Y.columns])
    info, per = [], {}
    for name, c in {"rep": REP, "rep_ring": RING, "classical": CL}.items():
        X = A[c]; s = scores(X); per[name] = s
        null = []
        for _ in range(n_perm):
            Xp = X.copy(); Xp.index = rng.permutation(X.index); null.append(np.sum(scores(Xp) > 0.5))
        null = np.array(null); k = int(np.sum(s > 0.5))
        info.append(dict(features=name, n_lines=len(ov), n_drugs=Y.shape[1], n_drugs_LOO_rho_gt_0_5=k,
                         null_mean=null.mean(), null_95pct=np.percentile(null, 95),
                         p=(np.sum(null >= k) + 1) / (n_perm + 1), median_LOO_rho=float(np.median(s))))
    perdrug = pd.DataFrame({"drug": D.loc[Y.columns, "NAME"].values, **{"LOO_rho_" + k: v for k, v in per.items()}})
    return G, pd.DataFrame(cis), pd.DataFrame(info), perdrug


# ----------------------------------------------------------------------------------------------- bridge
def read_expr(path, keep=None):
    head = pd.read_csv(path, sep="\t", nrows=0).columns
    use = [c for c in head if c == "Hugo_Symbol" or (keep is None and c != "Entrez_Gene_Id") or
           (keep is not None and c in keep)]
    X = pd.read_csv(path, sep="\t", usecols=use, dtype={c: np.float32 for c in use if c != "Hugo_Symbol"})
    return X.dropna(subset=["Hugo_Symbol"]).groupby("Hugo_Symbol").mean()


_COH = {}


def cohorts(genes_ccle):
    """Load once per process: CCLE, TCGA, METABRIC (z-scored genes) and clinical tables."""
    if _COH:
        return _COH
    cb = EXT / "cbioportal"
    C = read_expr(cb / "ccle_broad_2019/data_mrna_seq_rpkm.txt", keep=["%s_BREAST" % l for l in BREAST])
    C = np.log2(C.rename(columns=lambda c: c.replace("_BREAST", "")) + 1)
    T = np.log2(read_expr(cb / "brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt").clip(lower=0) + 1)
    Mx = read_expr(cb / "brca_metabric/data_mrna_illumina_microarray.txt")
    genes = C.index[(C.mean(1) >= 1) & (C.std(1) >= 0.3)].intersection(T.index).intersection(Mx.index)
    z = lambda X: X.loc[genes].sub(X.loc[genes].mean(1), axis=0).div(X.loc[genes].std(1).replace(0, 1), axis=0)
    Tz, Mz = z(T), z(Mx)
    tp = pd.read_csv(cb / "brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt", sep="\t", comment="#")
    tc = pd.DataFrame({"sample": Tz.columns}); tc["PATIENT_ID"] = tc["sample"].str[:12]
    tc = tc[tc["sample"].str[13:15] == "01"].merge(tp, on="PATIENT_ID")
    mp = pd.read_csv(cb / "brca_metabric/data_clinical_patient.txt", sep="\t", comment="#")
    mc = pd.DataFrame({"sample": Mz.columns}).merge(mp, left_on="sample", right_on="PATIENT_ID")
    ev = lambda s: s.astype(str).str.startswith("1").astype(int)
    _COH.update(C=C, genes=genes, cohorts={
        "TCGA_OS": (Tz, tc.assign(time=tc.OS_MONTHS, event=ev(tc.OS_STATUS), age=tc.AGE, sub=tc.SUBTYPE)),
        "TCGA_PFI": (Tz, tc.assign(time=tc.PFS_MONTHS, event=ev(tc.PFS_STATUS), age=tc.AGE, sub=tc.SUBTYPE)),
        "METABRIC_OS": (Mz, mc.assign(time=mc.OS_MONTHS, event=ev(mc.OS_STATUS), age=mc.AGE_AT_DIAGNOSIS,
                                      sub=mc.CLAUDIN_SUBTYPE)),
        "METABRIC_DSS": (Mz, mc.assign(time=mc.OS_MONTHS, event=(mc.VITAL_STATUS == "Died of Disease").astype(int),
                                       age=mc.AGE_AT_DIAGNOSIS, sub=mc.CLAUDIN_SUBTYPE)),
        "METABRIC_RFS": (Mz, mc.assign(time=mc.RFS_MONTHS, event=ev(mc.RFS_STATUS), age=mc.AGE_AT_DIAGNOSIS,
                                       sub=mc.CLAUDIN_SUBTYPE))})
    return _COH


def cox(Z, clin, up, down, adjust):
    from lifelines import CoxPHFitter
    d = clin[["sample", "time", "event", "age", "sub"]].copy()
    d["score"] = (Z.loc[up].mean(0) - Z.loc[down].mean(0)).reindex(d["sample"]).values
    d = d.dropna(subset=["time", "event", "score", "age"]); d = d[d.time > 0]
    d["score"] = (d.score - d.score.mean()) / d.score.std()
    if adjust:
        d = d.dropna(subset=["sub"]); d = d[~d["sub"].isin(["NC", "Normal", "BRCA_Normal"])]
        X = pd.concat([d[["time", "event", "score", "age"]],
                       pd.get_dummies(d["sub"], prefix="s", drop_first=True, dtype=float)], axis=1)
    else:
        X = d[["time", "event", "score"]]
    s = CoxPHFitter().fit(X, "time", "event").summary.loc["score"]
    return float(s["exp(coef)"]), float(s["p"]), float(s["z"]), int(len(X)), int(X.event.sum())


def bridge(A, rng, n_random, feats=("rep_logR", "rep_jitter", "cl_logAmp", "cl_periodSD")):
    H = cohorts(None); C, genes = H["C"], H["genes"]
    lines = [l for l in A.index if l in C.columns]
    cg = [dict(gene=g, feature=f, rho=spearmanr(A.loc[lines, f], C.loc[g, lines])[0],
               p=spearmanr(A.loc[lines, f], C.loc[g, lines])[1]) for g in CLOCK if g in C.index for f in feats]
    Cr = np.apply_along_axis(rankdata, 1, C.loc[genes, lines].values)
    sig = {}
    for f in feats:
        fr = rankdata(A.loc[lines, f].values)
        s = pd.Series([np.corrcoef(fr, r)[0, 1] for r in Cr], index=genes).sort_values()
        sig[f] = (list(s.index[-50:]), list(s.index[:50]))
    rand = [(list(rng.choice(np.array(genes), 50, replace=False)), list(rng.choice(np.array(genes), 50, replace=False)))
            for _ in range(n_random)]
    rows = []
    for cn, (Z, clin) in H["cohorts"].items():
        nz = np.abs([cox(Z, clin, u, d, True)[2] for u, d in rand])
        for f in feats:
            hr, p, *_ = cox(Z, clin, *sig[f], False)
            hra, pa, za, n, e = cox(Z, clin, *sig[f], True)
            rows.append(dict(cohort=cn, feature=f, n=n, events=e, HR_unadj=hr, p_unadj=p, HR_adj=hra, p_adj=pa,
                             z_adj=za, pct_vs_random=float(np.mean(nz < abs(za)))))
    sigdf = pd.DataFrame({"%s_%s" % (f, d): sig[f][i] for f in feats for i, d in enumerate(["up", "down"])})
    return pd.DataFrame(cg), sigdf, pd.DataFrame(rows), len(lines)


# ----------------------------------------------------------------------------------------------- main
def run(rep, codes, tag="msb_rel_L400_S18", n_perm=200, n_random=300, skip=()):
    rng = np.random.default_rng(0)
    out = outdir(rep)
    Z = np.load(codes)["z"].astype(float)
    F, L, A, expl, wc = build_features(Z, tag)
    F.to_csv(out / "trace_features.csv", index=False); L.to_csv(out / "line_features.csv")
    A.to_csv(out / "line_features_repavg.csv")
    summ = {"rep": rep, "codes": str(codes), "latent_dim": int(Z.shape[1]), "zPCA_explained": expl.round(4).tolist(),
            "window_radius_vs_cosinor_amp_spearman": round(wc, 3)}
    print("[%s] features: z dim %d, radius~amplitude rho %.2f" % (rep, Z.shape[1], wc), flush=True)
    C, I = clustering(F, L, rng)
    C.to_csv(out / "cluster_phenotype.csv", index=False); I.to_csv(out / "trace_line_identification.csv", index=False)
    summ["cluster"] = C.round(4).to_dict("records"); summ["identification"] = I.round(4).to_dict("records")
    print(C.round(3).to_string(index=False), flush=True)
    if "drugs" not in skip:
        G, cis, info, perdrug = drugs(A, rng, n_perm)
        G.to_csv(out / "cisplatin_gr.csv"); cis.to_csv(out / "drug_cisplatin.csv", index=False)
        info.to_csv(out / "drug_gdsc_information.csv", index=False); perdrug.to_csv(out / "drug_gdsc_per_drug.csv", index=False)
        summ["cisplatin"] = cis.round(4).to_dict("records"); summ["gdsc"] = info.round(4).to_dict("records")
        print(info.round(3).to_string(index=False), flush=True)
    if "bridge" not in skip:
        cg, sig, cx, nl = bridge(A, rng, n_random)
        cg.to_csv(out / "bridge_clockgenes.csv", index=False); sig.to_csv(out / "bridge_signatures.csv", index=False)
        cx.to_csv(out / "bridge_survival_cox.csv", index=False)
        summ["bridge_lines"] = nl; summ["survival"] = cx.round(4).to_dict("records")
        print(cx.round(3).to_string(index=False), flush=True)
    json.dump(summ, open(out / "summary.json", "w", encoding="utf-8"), indent=1, default=float)
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, help="name of the representation (output folder)")
    ap.add_argument("--codes", default=None, help="npz with key z; default results/exp02/codes/<rep>.npz")
    ap.add_argument("--tag", default="msb_rel_L400_S18")
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--n_random", type=int, default=300)
    ap.add_argument("--skip", nargs="*", default=[], choices=["drugs", "bridge"])
    a = ap.parse_args(argv)
    from pathlib import Path
    codes = Path(a.codes) if a.codes else ROOT / "results" / "exp02" / "codes" / ("%s.npz" % a.rep)
    codes = codes if codes.is_absolute() else ROOT / codes
    run(a.rep, codes, a.tag, a.n_perm, a.n_random, a.skip)


if __name__ == "__main__":
    main()
