#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
import pacmap
from sklearn.neighbors import LocalOutlierFactor

import matplotlib
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join("output/figures/class_cello/")

matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica"]
matplotlib.rcParams["figure.max_open_warning"] = 20000

reseed = 42
random.seed(reseed)
np.random.seed(reseed)

sc.settings.verbosity = 2  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.figdir = PLOTS_DIR
sc.settings.set_figure_params(
    dpi=120, dpi_save=600, vector_friendly=True, format="pdf", transparent=True
)
sc.settings.autoshow = False
sc.settings.autosave = True


# Load data:

bioproject = snakemake.params["bioprj"]
project = snakemake.params["prj"]
cb_fpr = snakemake.params["res"]

## IN:
data_path = snakemake.input[0]

## OUT:
h5ad_out_path = snakemake.output["h5ad_annotations_all"]
cello_table_out = snakemake.output["tables_annotations_all"]
h5ad_astro_out_path = snakemake.output["h5ad_annotations_astrocytes"]
astro_out_path = snakemake.output["tables_annotations_astrocytes"]
resource_loc = f"models/cello/"


adata = sc.read_h5ad(data_path)
samples = pd.read_table("samples.tsv").set_index("Run", drop=False)
adata.uns["name"] = project
adata.obs["bioproject"] = bioproject
adata.obs["project"] = project
adata.obs["model"] = [samples["Model"][i] for i in adata.obs["orig.ident"]]
adata.obs["tech"] = [samples["Tech"][i] for i in adata.obs["orig.ident"]]
adata.obs["region"] = [samples["Region"][i] for i in adata.obs["orig.ident"]]
adata.obs["sex"] = [samples["Sex"][i] for i in adata.obs["orig.ident"]]
adata.obs["stage"] = [samples["AgeGroup"][i] for i in adata.obs["orig.ident"]]
adata.obs["libname"] = [samples["LibraryName"][i] for i in adata.obs["orig.ident"]]
adata.obs["expbtch"] = [samples["ExpBatch"][i] for i in adata.obs["orig.ident"]]
adata.obs["condit"] = [samples["Condition"][i] for i in adata.obs["orig.ident"]]

adata


embedding = pacmap.PaCMAP(
    n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, apply_pca=False
)
adata.obsm["X_pacmap"] = embedding.fit_transform(adata.obsm["X_pca"], init="pca")


adata.raw = adata
astro_markers = [
    "Abcd1",
    "Abcd2",
    "Abcd3",
    "Acaa1",
    "Acaa2",
    "Acox1",
    "Agt",
    "Aldh1a1",
    "Aldh1l1",
    "Aldoc",
    "Apoe",
    "Aqp4",
    "Caf4",
    "Ckb",
    "Cnr1",
    "Cnr2",
    "Cst3",
    "Dagla",
    "Daglb",
    "Decr2",
    "Dnm1",
    "Drp1",
    "Ech1",
    "Egfr",
    "Eno1",
    "Faah",
    "Fgfr3",
    "Fis1",
    "Fos",
    "Fth1",
    "Ftl1",
    "Gfap",
    "Gja1",
    "Gli1",
    "Glul",
    "Hacd2",
    "Hadhb",
    "Hepacam",
    "Hif1",
    "Htra1",
    "Lcat",
    "Lpcat3",
    "Lxn",
    "Mdv1",
    "Mfn1",
    "Mfn2",
    "Mgll",
    "Mief1",
    "Napepld",
    "Ndrg2",
    "Nfia",
    "Opa1",
    "Otp",
    "Pex1",
    "Pex10",
    "Pex12",
    "Pex13",
    "Pex14",
    "Pex16",
    "Pex2",
    "Pex26",
    "Pex3",
    "Pex6",
    "Pla2g7",
    "Plcb1",
    "Pygb",
    "S100a6",
    "S100b",
    "Scd2",
    "Sgcd",
    "Slc1a2",
    "Slc1a3",
    "Slc38a1",
    "Slc6a11",
    "Slit1",
    "Slit2",
    "Sox9",
    "Tafa1",
    "Tkt",
    "Trpv1",
]
npr = [
    "Adcyap1r1",
    "Avpr1a",
    "Cckar",
    "Cckbr",
    "Cntfr",
    "Crhr1",
    "Crhr2",
    "Esr1",
    "Galr1",
    "Galr2",
    "Galr3",
    "Ghr",
    "Ghrhr",
    "Gpr83",
    "Grpr",
    "Hcrtr1",
    "Hcrtr2",
    "Kiss1r",
    "Mc1r",
    "Mc3r",
    "Mc4r",
    "Mchr1",
    "Nmbr",
    "Nmur1",
    "Nmur2",
    "Npffr1",
    "Npffr2",
    "Npy1r",
    "Npy2r",
    "Npy5r",
    "Ntrk2",
    "Ntsr2",
    "Oprd1",
    "Oprk1",
    "Oprl1",
    "Oprm1",
    "Oxtr",
    "Prlhr",
    "Prlr",
    "Prokr2",
    "Qrfpr",
    "Sstr1",
    "Sstr2",
    "Sstr3",
    "Tacr1",
    "Tacr3",
    "Trhr",
    "Vipr1",
    "Vipr2",
]

npep = [
    "Adcyap1",
    "Agrp",
    "Avp",
    "Bdnf",
    "Cartpt",
    "Cck",
    "Cntf",
    "Crh",
    "Gal",
    "Ghrh",
    "Grp",
    "Hcrt",
    "Kiss1",
    "Nmb",
    "Nms",
    "Nmu",
    "Npvf",
    "Npw",
    "Npy",
    "Nts",
    "Oxt",
    "Pdyn",
    "Penk",
    "Pmch",
    "Pnoc",
    "Pomc",
    "Qrfp",
    "Reln",
    "Rxfp1",
    "Sst",
    "Tac1",
    "Tac2",
    "Trh",
]

astro_markers = [g for g in astro_markers if g in adata.var_names]
astro_markers
npr = [g for g in npr if g in adata.var_names]
npr
npep = [g for g in npep if g in adata.var_names]
npep
save_mrtree = f"-mrtree-{project}-full-dataset_fpr_{cb_fpr}.pdf"


sc.pl.embedding(
    adata,
    basis="X_pacmap",
    color="k_tree",
    title="PaCMAP: mrtree-derived clusters in {sample}".format(
        sample=adata.uns["name"]
    ),
    save=save_mrtree,
)
sc.pl.embedding(
    adata, basis="X_pacmap", color=astro_markers, save="_markers" + save_mrtree
)
sc.pl.embedding(adata, basis="X_pacmap", color=npr, save="_npr" + save_mrtree)
sc.pl.embedding(adata, basis="X_pacmap", color=npep, save="_npep" + save_mrtree)

# Query Omnipath and get PanglaoDB
markers = dc.get_resource("PanglaoDB")
markers


# Filter by canonical_marker and human
markers = markers[
    (markers["mouse"] == "True") & (markers["canonical_marker"] == "True")
]
markers


# Remove duplicated entries
markers = markers[~markers.duplicated(["cell_type", "genesymbol"])]
markers["genesymbol_mm"] = [i.capitalize() for i in markers["genesymbol"]]
markers


# Enrichment with Over Representation Analysis
dc.run_ora(
    mat=adata,
    net=markers,
    source="cell_type",
    target="genesymbol_mm",
    min_n=3,
    verbose=False,
    use_raw=True,
)


# Object for visualizing the ORA-results
acts = dc.get_acts(adata, obsm_key="ora_estimate")


sc.pl.embedding(
    acts,
    basis="X_pacmap",
    color=[
        "Astrocytes",
        "Ependymal cells",
        "Tanycytes",
        "Oligodendrocyte progenitor cells",
    ],
    title="PaCMAP: Enrichment with Over Representation Analysis for {feature} in {sample}".format(
        feature="ora_celltype", sample=adata.uns["name"]
    ),
    save=f"_ora-glia_{project}-full-dataset_fpr_{cb_fpr}.pdf",
)


# Annotaiton
mean_enr = dc.summarize_acts(acts, groupby="k_tree", min_std=1)
annotation_dict = dc.assign_groups(mean_enr)
annotation_dict


[k for k, v in annotation_dict.items() if v == "Astrocytes"]


# Add cell type column based on annotation
adata.obs["ora_celltype"] = [
    annotation_dict[str(clust)] for clust in adata.obs["k_tree"]
]
save_ora_celltype = f"_ora_celltype_{project}-full-dataset_fpr_{cb_fpr}.pdf"

## TODO: increase n_neighbors and contamination filter
outlier_scores = LocalOutlierFactor(n_neighbors=25, contamination=0.05).fit_predict(
    adata.obsm["X_umap"]
)
adata = adata[outlier_scores != -1]

sc.pl.embedding(
    adata,
    basis="X_pacmap",
    color="ora_celltype",
    title="PaCMAP: {feature}".format(feature="ora_celltype"),
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="tab20",
    save=save_ora_celltype,
)


sc.pl.umap(
    adata,
    color="ora_celltype",
    title="UMAP: {feature}".format(feature="ora_celltype"),
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="tab20",
    save=save_ora_celltype,
)


pos_astro = adata[
    (
        (
            (
                np.sum(
                    np.array(
                        [
                            (adata[:, "Adcyap1r1"].to_df() > 1).stack().values,
                            (adata[:, "Agt"].to_df() > 1).stack().values,
                            (adata[:, "Aldh1l1"].to_df() > 1).stack().values,
                            (adata[:, "Fgfr3"].to_df() > 1).stack().values,
                            (adata[:, "Glul"].to_df() > 1).stack().values,
                            (adata[:, "Gli1"].to_df() > 1).stack().values,
                            (adata[:, "Lcat"].to_df() > 1).stack().values,
                            (adata[:, "Lxn"].to_df() > 1).stack().values,
                            (adata[:, "Aqp4"].to_df() > 10).stack().values,
                            (adata[:, "Gja1"].to_df() > 10).stack().values,
                            (adata[:, "Gfap"].to_df() > 10).stack().values,
                            (adata[:, "Hacd2"].to_df() > 10).stack().values,
                            (adata[:, "Hepacam"].to_df() > 10).stack().values,
                            (adata[:, "Htra1"].to_df() > 10).stack().values,
                            (adata[:, "Ndrg2"].to_df() > 10).stack().values,
                            (adata[:, "Ntsr2"].to_df() > 10).stack().values,
                            (adata[:, "Ntrk2"].to_df() > 10).stack().values,
                            (adata[:, "Pla2g7"].to_df() > 10).stack().values,
                            (adata[:, "Slc1a3"].to_df() > 10).stack().values,
                            (adata[:, "Slc6a11"].to_df() > 10).stack().values,
                            (adata[:, "Slc1a2"].to_df() > 100).stack().values,
                            (adata[:, "Apoe"].to_df() > 100).stack().values,
                        ]
                    ),
                    axis=0,
                )
                >= 7
            )
            | (adata.obs["ora_celltype"].isin(["Astrocytes"]))
        )
        & (
            np.sum(
                np.array(
                    [
                        (adata[:, "Rbfox3"].to_df() > 5).stack().values,
                        (adata[:, "Dlx5"].to_df() > 1).stack().values,
                        (adata[:, "Elavl4"].to_df() > 5).stack().values,
                        (adata[:, "Stmn2"].to_df() > 5).stack().values,
                        (adata[:, "Snap25"].to_df() > 5).stack().values,
                        (adata[:, "Th"].to_df() > 1).stack().values,
                        (adata[:, "Slc17a6"].to_df() > 5).stack().values,
                        (adata[:, "Gad1"].to_df() > 5).stack().values,
                        (adata[:, "Gad2"].to_df() > 5).stack().values,
                        (adata[:, "Npy"].to_df() > 5).stack().values,
                        (adata[:, "Agrp"].to_df() > 5).stack().values,
                        (adata[:, "Crh"].to_df() > 5).stack().values,
                        (adata[:, "Trh"].to_df() > 5).stack().values,
                        (adata[:, "Avp"].to_df() > 10).stack().values,
                        (adata[:, "Pomc"].to_df() > 10).stack().values,
                        (adata[:, "Hcrt"].to_df() > 10).stack().values,
                        (adata[:, "Oxt"].to_df() > 10).stack().values,
                        (adata[:, "Cxcl14"].to_df() > 1).stack().values,
                        (adata[:, "Cxcl1"].to_df() > 1).stack().values,
                        (adata[:, "Cxcl2"].to_df() > 1).stack().values,
                        (adata[:, "Foxj1"].to_df() > 1).stack().values,
                        (adata[:, "Vim"].to_df() > 1).stack().values,
                        (adata[:, "Nes"].to_df() > 1).stack().values,
                        (adata[:, "Enkur"].to_df() > 1).stack().values,
                        (adata[:, "Foxj1"].to_df() > 1).stack().values,
                        (adata[:, "Kif6"].to_df() > 1).stack().values,
                        (adata[:, "Kif9"].to_df() > 1).stack().values,
                        (adata[:, "Hydin"].to_df() > 1).stack().values,
                        (adata[:, "Mog"].to_df() > 1).stack().values,
                        (adata[:, "Plp1"].to_df() > 1).stack().values,
                        (adata[:, "Cnp"].to_df() > 1).stack().values,
                        (adata[:, "Mag"].to_df() > 1).stack().values,
                        (adata[:, "Opalin"].to_df() > 1).stack().values,
                        (adata[:, "Sox10"].to_df() > 1).stack().values,
                        (adata[:, "Olig1"].to_df() > 5).stack().values,
                        (adata[:, "Olig2"].to_df() > 5).stack().values,
                        (adata[:, "Aif1"].to_df() > 1).stack().values,
                        (adata[:, "Itgam"].to_df() > 1).stack().values,
                        (adata[:, "Ptprc"].to_df() > 1).stack().values,
                        (adata[:, "Fcrls"].to_df() > 1).stack().values,
                        (adata[:, "Pdgfra"].to_df() > 5).stack().values,
                        (adata[:, "Pdgfrb"].to_df() > 10).stack().values,
                        (adata[:, "Gpr17"].to_df() > 1).stack().values,
                        (adata[:, "Ugt8a"].to_df() > 1).stack().values,
                        (adata[:, "Sema3c"].to_df() > 1).stack().values,
                        (adata[:, "Sema4d"].to_df() > 1).stack().values,
                        (adata[:, "Sema4f"].to_df() > 1).stack().values,
                        (adata[:, "Gpr37"].to_df() > 10).stack().values,
                        (adata[:, "Cspg4"].to_df() > 1).stack().values,
                        (adata[:, "Lingo1"].to_df() > 10).stack().values,
                        (adata[:, "Rgs5"].to_df() > 10).stack().values,
                        (adata[:, "Des"].to_df() > 1).stack().values,
                        (adata[:, "Acta2"].to_df() > 1).stack().values,
                        (adata[:, "Cd248"].to_df() > 1).stack().values,
                        (adata[:, "Myh11"].to_df() > 1).stack().values,
                        (adata[:, "Cdh5"].to_df() > 10).stack().values,
                        (adata[:, "Fgf10"].to_df() > 1).stack().values,
                        (adata[:, "Rax"].to_df() > 1).stack().values,
                        (adata[:, "Tbx3"].to_df() > 1).stack().values,
                    ]
                ),
                axis=0,
            )
            < 3
        )
    ),
    :,
]

sc.pl.embedding(
    pos_astro,
    basis="X_pacmap",
    color="ora_celltype",
    title="PaCMAP: {feature}".format(feature="ora_celltype"),
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="tab20",
    save="-pos_astro" + save_ora_celltype,
)


sc.pl.umap(
    pos_astro,
    color="ora_celltype",
    title="UMAP: {feature}".format(feature="ora_celltype"),
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="tab20",
    save="-pos_astro" + save_ora_celltype,
)

# Save results:
if not os.path.exists(os.path.dirname(h5ad_out_path)):
    os.mkdir(os.path.dirname(h5ad_out_path))
adata.write(h5ad_out_path)
astrodata = pos_astro
# split train and test sets first
target_train = min([1000 / astrodata.n_obs, 0.9])
inds1 = np.random.choice(
    [True, False], size=(astrodata.n_obs,), p=[target_train, 1 - target_train]
)
inds2 = np.ones(astrodata.n_obs).astype(bool)
inds2[inds1] = False
astrodata.obs["train"] = pd.array(inds1, dtype=bool)
astrodata.obs["test"] = pd.array(inds2, dtype=bool)
astrodata.write(h5ad_astro_out_path)


if not os.path.exists(os.path.dirname(astro_out_path)):
    os.mkdir(os.path.dirname(astro_out_path))
pd.DataFrame(astrodata.obs).to_csv(astro_out_path, sep="\t", header=True)


if not os.path.exists(os.path.dirname(cello_table_out)):
    os.mkdir(os.path.dirname(cello_table_out))
pd.DataFrame(adata.obs).to_csv(cello_table_out, sep="\t", header=True)
