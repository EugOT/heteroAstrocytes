import anndata
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pyroe import load_fry


matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica"]
matplotlib.rcParams["figure.max_open_warning"] = 20000

#define count matrix format
custom_format = {'X' : ['S', 'A'],
                 'spliced' : ['S'],
                 'unspliced' : ['U'],
                 'ambiguous' : ['A']}

# load count matrix
adata = load_fry(snakemake.params["path"],
                 output_format = custom_format)

adata.uns["name"] = snakemake.params["sample_run_name"]

adata.write(snakemake.output["h5ad"])

# Knee plot
expected_num_cells = snakemake.params["expected_num_cells"]
knee = np.sort((np.array(adata.X.sum(axis=1))).flatten())[::-1]

fig, ax = plt.subplots(figsize=(10, 7))

ax.loglog(knee, range(len(knee)), linewidth=5, color="g")
ax.axvline(x=knee[expected_num_cells], linewidth=3, color="k")
ax.axhline(y=expected_num_cells, linewidth=3, color="k")

ax.set_xlabel("UMI Counts")
ax.set_ylabel("Set of Barcodes")
ax.set_title(adata.uns["name"] + ": " + str(expected_num_cells))

plt.grid(True, which="both")
plt.savefig(os.path.join(snakemake.output["knee"]), dpi=600, transparent=True)
