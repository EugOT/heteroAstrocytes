#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx
import operator as op
import anndata as ad
import scanpy as sc
from cytoolz import compose

from pyscenic.export import add_scenic_metadata
from pyscenic.utils import load_motifs
from pyscenic.transform import df2regulons
from pyscenic.aucell import aucell


# Set seed for reproducibility
reseed = 42
random.seed(reseed)
np.random.seed(reseed)


# Set maximum number of jobs
n_cores = snakemake.threads
sc.settings.njobs = n_cores
sc.settings.verbosity = 2  # verbosity: errors (0), warnings (1), info (2), hints (3)

# Load data:

bioproject = snakemake.params["bioprj"]
project = snakemake.params["prj"]

## IN:
data_path = snakemake.input["h5ad"]
motifs_path = snakemake.input["motifs"]

## OUT:
regulons_fname = snakemake.output["regulons"]
auc_mtx_fname = snakemake.output["auc_mtx"]
h5ad_scenic = snakemake.output["h5ad_scenic"]
h5ad_regulons = snakemake.output["h5ad_regulons"]
gephi = snakemake.output["gephi"]

BASE_FOLDER = os.path.dirname(motifs_path)

adata = sc.read_h5ad(data_path)


# Funcions:
## Functions for the creation of regulons.
def derive_regulons(code, folder):
    # Load enriched motifs.
    motifs = load_motifs(os.path.join(folder, "{}.motifs.csv".format(code)))
    motifs.columns = motifs.columns.droplevel(0)

    def contains(*elems):
        def f(context):
            return any(elem in context for elem in elems)

        return f

    # For the creation of regulons we only keep the 10-species databases and the activating modules. We also remove the
    # enriched motifs for the modules that were created using the method 'weight>50.0%' (because these modules are not part
    # of the default settings of modules_from_adjacencies anymore.
    motifs = motifs[
        np.fromiter(
            map(compose(op.not_, contains("weight>50.0%")), motifs.Context),
            dtype=bool,
        )
        & np.fromiter(
            map(
                contains(
                    "mm9-tss-centered-10kb-10species.mc9nr.genes_vs_motifs.rankings",
                    "mm9-500bp-upstream-10species.mc9nr.genes_vs_motifs.rankings",
                    "mm9-tss-centered-5kb-10species.mc9nr.genes_vs_motifs.rankings",
                ),
                motifs.Context,
            ),
            dtype=bool,
        )
        & np.fromiter(map(contains("activating"), motifs.Context), dtype=bool)
    ]

    # We build regulons only using enriched motifs with a NES of 3.0 or higher; we take only directly annotated TFs or TF annotated
    # for an orthologous gene into account; and we only keep regulons with at least 3 genes.
    regulons = list(
        filter(
            lambda r: len(r) >= 3,
            df2regulons(
                motifs[
                    (motifs["NES"] >= 3.0)
                    & (
                        (motifs["Annotation"] == "gene is directly annotated")
                        | (
                            motifs["Annotation"].str.startswith(
                                "gene is orthologous to"
                            )
                            & motifs["Annotation"].str.endswith(
                                "which is directly annotated for motif"
                            )
                        )
                    )
                ]
            ),
        )
    )

    # Rename regulons, i.e. remove suffix.
    regulons = list(map(lambda r: r.rename(r.transcription_factor), regulons))

    # Pickle these regulons.
    with open(os.path.join(folder, "{}.regulons.dat".format(code)), "wb") as f:
        pickle.dump(regulons, f)


## Functions for the export of regulons as file for Gephi.
def export_regulons(regulons, fname):
    """

    Export regulons as GraphML.

    :param regulons: The sequence of regulons to export.
    :param fname: The name of the file to create.
    """
    graph = nx.DiGraph()
    for regulon in regulons:
        src_name = regulon.transcription_factor
        graph.add_node(src_name, group="transcription_factor")
        edge_type = "activating" if "activating" in regulon.context else "inhibiting"
        node_type = (
            "activated_target"
            if "activating" in regulon.context
            else "inhibited_target"
        )
        for dst_name, edge_strength in regulon.gene2weight.items():
            graph.add_node(dst_name, group=node_type)
            graph.add_edge(
                src_name, dst_name, weight=edge_strength, interaction=edge_type
            )
    nx.readwrite.write_graphml(graph, fname)


# Create regulons.
derive_regulons(f"{bioproject}-astrocytes_dataset-0.001", BASE_FOLDER)


with open(regulons_fname, "rb") as f:
    regulons = pickle.load(f)


# Export regulons as GraphML.
export_regulons(regulons, fname=gephi)


# Derive regulons impact per cell matrix using AUCell.
auc_mtx = aucell(adata.to_df(), regulons, num_workers=n_cores)


auc_mtx.to_csv(auc_mtx_fname)

# Add regulons to adata object.
add_scenic_metadata(adata, auc_mtx, regulons)


# Create a visualization of cells embedding in the regulons space.
# embedding = pacmap.PaCMAP(n_components=2, MN_ratio=0.5, FP_ratio=2.0, apply_pca=True)
# adata.obsm["X_pacmap"] = embedding.fit_transform(adata.obsm["X_aucell"], init="pca")


adata.write_h5ad(h5ad_scenic)

# Create anndata object with regulons as a main X matrix.
aucell_adata = ad.AnnData(auc_mtx)
aucell_adata.obs = adata.obs

adata.write_h5ad(h5ad_regulons)
