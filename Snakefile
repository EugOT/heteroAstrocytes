"""Tretiakov et al., 2023"""
import pandas as pd
from os import listdir, rename, getcwd
from os.path import join, basename, dirname, abspath
from pathlib import Path
from snakemake.utils import validate, min_version, Paramspace

##### set minimum snakemake version #####
min_version("7.20.0")

##### load config and sample sheets #####
configfile: "config.yaml"
# samples = pd.read_table(config["samples"]).set_index("Run", drop=False)
paramspace = Paramspace(
    pd.read_csv("params_pairs.tsv", sep="\t"))
# resolutions = [0.001, 0.01, 0.05, 0.1]
resolutions = [0.001, 0.01]
resolution2 = [0.001]
fpr = [f"{i}" for i in resolutions] + ["nc"]
substr_level = [100, 250, 500]
substr_level2= [100, 250]
signature = [f"{i}" for i in substr_level] + ["full"]
neighbors = [5, 10, 25, 50]
neighbor2 = [5]
connectivity = ["full_tree", "min_tree"]
metrics = ["correlation"]
whole_hypothalamus = ['PRJNA779749', 'PRJNA548917', 'PRJNA547712', 'PRJNA438862']
subregions = ['PRJNA847050', 'PRJNA815819', 'PRJNA798401', 'PRJNA723345',
              'PRJNA722418', 'PRJNA705596', 'PRJNA679294', 'PRJNA611624',
              'PRJNA604055', 'PRJNA548532', 'PRJNA515063',
              'PRJNA453138']
subregions2 = ['PRJNA779749', 'PRJNA847050', 'PRJNA815819', 'PRJNA798401',
              'PRJNA723345', 'PRJNA722418', 'PRJNA705596', 'PRJNA679294',
              'PRJNA611624', 'PRJNA604055', 'PRJNA548532', 'PRJNA515063',
              'PRJNA453138']

########################
##### subworkflows #####
########################
subworkflow PRJNA847050:
    workdir:
        "PRJNA847050"
    snakefile:
        "PRJNA847050/Snakefile"
    configfile:
        "PRJNA847050/config.yaml"

subworkflow PRJNA815819:
    workdir:
        "PRJNA815819"
    snakefile:
        "PRJNA815819/Snakefile"
    configfile:
        "PRJNA815819/config.yaml"

subworkflow PRJNA798401:
    workdir:
        "PRJNA798401"
    snakefile:
        "PRJNA798401/Snakefile"
    configfile:
        "PRJNA798401/config.yaml"

subworkflow PRJNA779749:
    workdir:
        "PRJNA779749"
    snakefile:
        "PRJNA779749/Snakefile"
    configfile:
        "PRJNA779749/config.yaml"

subworkflow PRJNA723345:
    workdir:
        "PRJNA723345"
    snakefile:
        "PRJNA723345/Snakefile"
    configfile:
        "PRJNA723345/config.yaml"

subworkflow PRJNA722418:
    workdir:
        "PRJNA722418"
    snakefile:
        "PRJNA722418/Snakefile"
    configfile:
        "PRJNA722418/config.yaml"

subworkflow PRJNA705596:
    workdir:
        "PRJNA705596"
    snakefile:
        "PRJNA705596/Snakefile"
    configfile:
        "PRJNA705596/config.yaml"

subworkflow PRJNA679294:
    workdir:
        "PRJNA679294"
    snakefile:
        "PRJNA679294/Snakefile"
    configfile:
        "PRJNA679294/config.yaml"

subworkflow PRJNA611624:
    workdir:
        "PRJNA611624"
    snakefile:
        "PRJNA611624/Snakefile"
    configfile:
        "PRJNA611624/config.yaml"

subworkflow PRJNA604055:
    workdir:
        "PRJNA604055"
    snakefile:
        "PRJNA604055/Snakefile"
    configfile:
        "PRJNA604055/config.yaml"

subworkflow PRJNA548917:
    workdir:
        "PRJNA548917"
    snakefile:
        "PRJNA548917/Snakefile"
    configfile:
        "PRJNA548917/config.yaml"

subworkflow PRJNA548532:
    workdir:
        "PRJNA548532"
    snakefile:
        "PRJNA548532/Snakefile"
    configfile:
        "PRJNA548532/config.yaml"

subworkflow PRJNA547712:
    workdir:
        "PRJNA547712"
    snakefile:
        "PRJNA547712/Snakefile"
    configfile:
        "PRJNA547712/config.yaml"

# subworkflow PRJNA540713:
    # workdir:
    #     "PRJNA540713"
    # snakefile:
    #     "PRJNA540713/Snakefile"
    # configfile:
    #     "PRJNA540713/config.yaml"

subworkflow PRJNA515063:
    workdir:
        "PRJNA515063"
    snakefile:
        "PRJNA515063/Snakefile"
    configfile:
        "PRJNA515063/config.yaml"

# subworkflow PRJNA490830:
    # workdir:
    #     "PRJNA490830"
    # snakefile:
    #     "PRJNA490830/Snakefile"
    # configfile:
    #     "PRJNA490830/config.yaml"

# subworkflow PRJNA481681:
    # workdir:
    #     "PRJNA481681"
    # snakefile:
    #     "PRJNA481681/Snakefile"
    # configfile:
    #     "PRJNA481681/config.yaml"

subworkflow PRJNA453138:
    workdir:
        "PRJNA453138"
    snakefile:
        "PRJNA453138/Snakefile"
    configfile:
        "PRJNA453138/config.yaml"

subworkflow PRJNA438862:
    workdir:
        "PRJNA438862"
    snakefile:
        "PRJNA438862/Snakefile"
    configfile:
        "PRJNA438862/config.yaml"

# subworkflow PRJNA360829:
    # workdir:
    #     "PRJNA360829"
    # snakefile:
    #     "PRJNA360829/Snakefile"
    # configfile:
    #     "PRJNA360829/config.yaml"

# subworkflow PRJNA345147:
    # workdir:
    #     "PRJNA345147"
    # snakefile:
    #     "PRJNA345147/Snakefile"
    # configfile:
    #     "PRJNA345147/config.yaml"


########################
##### target rules #####
########################
shell.executable("/bin/bash")

rule all:
    input:
        "GRCm39/index/piscem_idx.refinfo",
        "GRCm39/index/simpleaf_index.json",
        "GRCm39/index/t2g_3col.tsv",
        # expand(["output/tables/deg_results/logreg-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.tsv", "output/tables/deg_results/mast-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.tsv", "output/figures/markers_subregions/rank_genes_groups_clusters-logreg-{subregions}-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf", "output/figures/markers_subregions/rank_genes_groups_clusters-wilcoxon-{subregions}-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf"], substr_level=substr_level2, subregions=subregions, fpr=resolution2, connectivity_model=connectivity, metric=metrics, knn=neighbor2),
        # expand("data/resolved_subregions/PRJNA779749-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pickle", substr_level=substr_level2, fpr=resolution2, connectivity_model=connectivity, metric=metrics, knn=neighbors),
        expand("output/tables/shared_anchor_genes/astrocytes_genes-{whole_hypothalamus}-{fpr}.tsv", whole_hypothalamus=whole_hypothalamus, fpr=fpr),
        expand("output/tables/shared_signature/astrocytes_genes-aggregated-{fpr}.tsv", fpr=fpr),
        expand(["data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_{subregions}-{substr_level}-astrocytes_datasets_{fpr}.h5ad"], subregions=subregions, substr_level=substr_level, fpr=fpr),
        expand(["data/paired_integrations-full/{params}.h5ad", "data/paired_integrations-full/{params}.tsv"], params=paramspace.instance_patterns),
        PRJNA847050(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA815819(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA798401(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA779749(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA723345(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA722418(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA705596(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA679294(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA611624(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA604055(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA548917(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA548532(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA547712(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA515063(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA453138(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA438862(expand("output/tables/01A-eda-whole_dataset-fpr_{res}/parameters.json", res=resolutions)),
        PRJNA847050("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA815819("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA798401("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA779749("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA723345("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA722418("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA705596("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA679294("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA611624("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA604055("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA548917("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA548532("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA547712("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA515063("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA453138("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA438862("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        PRJNA847050(expand("data/class_cello/PRJNA847050-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA815819(expand("data/class_cello/PRJNA815819-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA798401(expand("data/class_cello/PRJNA798401-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA779749(expand("data/class_cello/PRJNA779749-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA723345(expand("data/class_cello/PRJNA723345-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA722418(expand("data/class_cello/PRJNA722418-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA705596(expand("data/class_cello/PRJNA705596-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA679294(expand("data/class_cello/PRJNA679294-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA611624(expand("data/class_cello/PRJNA611624-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA604055(expand("data/class_cello/PRJNA604055-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA548917(expand("data/class_cello/PRJNA548917-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA548532(expand("data/class_cello/PRJNA548532-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA547712(expand("data/class_cello/PRJNA547712-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA515063(expand("data/class_cello/PRJNA515063-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA453138(expand("data/class_cello/PRJNA453138-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA438862(expand("data/class_cello/PRJNA438862-astrocytes_dataset-{res}-initial_selection.h5ad", res=resolutions)),
        PRJNA847050("data/class_cello/PRJNA847050-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA815819("data/class_cello/PRJNA815819-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA798401("data/class_cello/PRJNA798401-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA779749("data/class_cello/PRJNA779749-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA723345("data/class_cello/PRJNA723345-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA722418("data/class_cello/PRJNA722418-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA705596("data/class_cello/PRJNA705596-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA679294("data/class_cello/PRJNA679294-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA611624("data/class_cello/PRJNA611624-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA604055("data/class_cello/PRJNA604055-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA548917("data/class_cello/PRJNA548917-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA548532("data/class_cello/PRJNA548532-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA547712("data/class_cello/PRJNA547712-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA515063("data/class_cello/PRJNA515063-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA453138("data/class_cello/PRJNA453138-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA438862("data/class_cello/PRJNA438862-astrocytes_dataset-nc-initial_selection.h5ad"),
        PRJNA847050("data/PRJNA847050-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA815819("data/PRJNA815819-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA798401("data/PRJNA798401-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA779749("data/PRJNA779749-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA723345("data/PRJNA723345-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA722418("data/PRJNA722418-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA705596("data/PRJNA705596-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA679294("data/PRJNA679294-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA611624("data/PRJNA611624-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA604055("data/PRJNA604055-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA548917("data/PRJNA548917-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA548532("data/PRJNA548532-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA547712("data/PRJNA547712-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA515063("data/PRJNA515063-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA453138("data/PRJNA453138-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA438862("data/PRJNA438862-astrocytes_dataset-0.001-regulons.h5ad"),
        PRJNA847050("output/tables/PRJNA847050-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA815819("output/tables/PRJNA815819-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA798401("output/tables/PRJNA798401-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA779749("output/tables/PRJNA779749-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA723345("output/tables/PRJNA723345-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA722418("output/tables/PRJNA722418-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA705596("output/tables/PRJNA705596-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA679294("output/tables/PRJNA679294-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA611624("output/tables/PRJNA611624-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA604055("output/tables/PRJNA604055-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA548917("output/tables/PRJNA548917-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA548532("output/tables/PRJNA548532-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA547712("output/tables/PRJNA547712-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA515063("output/tables/PRJNA515063-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA453138("output/tables/PRJNA453138-astrocytes_dataset-0.001-graph-regulons.graphml"),
        PRJNA438862("output/tables/PRJNA438862-astrocytes_dataset-0.001-graph-regulons.graphml"),



        # PRJNA540713("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        # PRJNA490830("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        # PRJNA481681("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        # PRJNA360829("output/tables/01-eda-whole_dataset-nc/parameters.json"),
        # PRJNA345147("output/tables/01-eda-whole_dataset-nc/parameters.json"),


##### load rules #####

rule build_velocity_spliceu_index:
    input:
        fasta="GRCm39/fasta/genome.fa",
        gtf="GRCm39/genes/genes.gtf"
    output:
        refinfo="GRCm39/index/piscem_idx.refinfo",
        simpleaf="GRCm39/index/simpleaf_index.json",
        t2g="GRCm39/index/t2g_3col.tsv"
    params:
        index="GRCm39"
    threads: 96
    container:
        "docker://etretiakov/usefulaf:0.9.0"
    shell:
        ("simpleaf index \
        --use-piscem \
        --threads {threads} \
        --output {params.index} \
        --ref-type spliceu \
        --fasta {input.fasta} \
        --gtf {input.gtf}")


rule paired_integrations_full:
    output:
        h5ad_pair=f"data/paired_integrations-full/{paramspace.wildcard_pattern}.h5ad",
        anchor_genes=f"data/paired_integrations-full/{paramspace.wildcard_pattern}.tsv"
    params:
        fullpairintegr=paramspace.instance
    container:
        "docker://etretiakov/workbench-session-complete:jammy-2022.12.09-custom-11.2"
    threads: 16
    resources:
        mem_mb=40000
    script:
        "code/get_full_pair_mtx.py"


rule derive_signature_across_subregions:
    input:
        anchor_genes=expand("data/paired_integrations-full/whole_hypothalamus~{{whole_hypothalamus}}/subregions~{subregions}/fpr~{{fpr}}.tsv", whole_hypothalamus=whole_hypothalamus, subregions=subregions, fpr=fpr)
    output:
        shared_anchor_genes="output/tables/shared_anchor_genes/astrocytes_genes-{whole_hypothalamus}-{fpr}.tsv"
    container:
        "docker://etretiakov/workbench-session-complete:jammy-2022.12.09-custom-11.2"
    threads: 2
    resources:
        mem_mb=8000
    script:
        "code/get_shared_signature.R"


rule aggregate_signature:
    input:
        shared_anchor_genes=expand("output/tables/shared_anchor_genes/astrocytes_genes-{whole_hypothalamus}-{{fpr}}.tsv", whole_hypothalamus=whole_hypothalamus, fpr=fpr)
    output:
        aggregated_shared_genes="output/tables/shared_signature/astrocytes_genes-aggregated-{fpr}.tsv"
    params:
        res="{fpr}"
    container:
        "docker://etretiakov/workbench-session-complete:jammy-2022.12.09-custom-11.2"
    threads: 2
    resources:
        mem_mb=8000
    script:
        "code/get_aggregated_shared_signature.R"


rule paired_integrations_vanished_signature:
    input:
        h5ad_pair="data/paired_integrations-full/whole_hypothalamus~PRJNA779749/subregions~{subregions}/fpr~{fpr}.h5ad",
        aggregated_shared_genes="output/tables/shared_signature/astrocytes_genes-aggregated-{fpr}.tsv"
    output:
        h5ad_pair=expand("data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_{{subregions}}-{substr_level}-astrocytes_datasets_{{fpr}}.h5ad", substr_level=substr_level, subregions=subregions, fpr=fpr),
        tables_genes_corr="output/tables/paired_integrations-wo_signature/paired_correlation_table-PRJNA779749_and_{subregions}-astrocytes_datasets_{fpr}.tsv",
        tables_genes_corr_p_value="output/tables/paired_integrations-wo_signature/paired_correlation_p_value_table-PRJNA779749_and_{subregions}-astrocytes_datasets_{fpr}.tsv",
        tables_genes_irrelevant=expand("output/tables/paired_integrations-wo_signature/paired_list_of_signature_corr-PRJNA779749_and_{{subregions}}-{substr_level}-astrocytes_datasets_{{fpr}}.tsv", substr_level=substr_level, subregions=subregions, fpr=fpr)
    params:
        sample1="PRJNA779749",
        sample2="{subregions}",
        res="{fpr}"
    container:
        "docker://etretiakov/workbench-session-complete:jammy-2022.12.09-custom-11.3"
    threads: 16
    resources:
        mem_mb=40000
    script:
        "code/get_substr_pair_mtx.py"


# rule regionally_specific_clusters:
#     input:
#         h5ad_pairs=expand("data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_{subregions}-{{substr_level}}-astrocytes_datasets_{{fpr}}.h5ad", substr_level=substr_level, subregions=subregions, fpr=fpr)
#     output:
#         # de_picklefile="data/resolved_subregions/PRJNA779749-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pickle",
#         picklefile=expand("models/svm/{subregions}-astrocytes_dataset-msp_{{connectivity_model}}-metric_{{metric}}-k_{{knn}}-sign_{{substr_level}}-amb_{{fpr}}.pickle", substr_level=substr_level, subregions=subregions, fpr=fpr, connectivity_model=connectivity, metric=metrics),
#         sing_h5ads=expand("data/resolved_subregions/{subregions}-astrocytes_dataset-msp_{{connectivity_model}}-metric_{{metric}}-k_{{knn}}-sign_{{substr_level}}-amb_{{fpr}}.h5ad", substr_level=substr_level, subregions=subregions, fpr=fpr, connectivity_model=connectivity, metric=metrics),
#         sing_loom=expand("data/resolved_subregions/{subregions}-astrocytes_dataset-msp_{{connectivity_model}}-metric_{{metric}}-k_{{knn}}-sign_{{substr_level}}-amb_{{fpr}}.loom", substr_level=substr_level, subregions=subregions, fpr=fpr, connectivity_model=connectivity, metric=metrics),
#         umap_tsv=expand("data/resolved_subregions/{subregions}-astrocytes_dataset-msp_{{connectivity_model}}-metric_{{metric}}-k_{{knn}}-sign_{{substr_level}}-amb_{{fpr}}-umap.tsv", substr_level=substr_level, subregions=subregions, fpr=fpr, connectivity_model=connectivity, metric=metrics),
#         pacmap_tsv=expand("data/resolved_subregions/{subregions}-astrocytes_dataset-msp_{{connectivity_model}}-metric_{{metric}}-k_{{knn}}-sign_{{substr_level}}-amb_{{fpr}}-pacmap.tsv", substr_level=substr_level, subregions=subregions, fpr=fpr, connectivity_model=connectivity, metric=metrics),
#         plot_densmap_sup=expand("output/figures/resolved_subregions/astrocytes_X1_{subregions}-msp_{{connectivity_model}}-metric_{{metric}}-k_{{knn}}-sign_{{substr_level}}-amb_{{fpr}}-densmap_supervised.pdf", substr_level=substr_level, subregions=subregions, fpr=fpr, connectivity_model=connectivity, metric=metrics)
#     params:
#         k="{knn}",
#         substr_sign="{substr_level}",
#         res="{fpr}",
#         metric="{metric}",
#         connectivity_model="{connectivity_model}"
#     container:
#         "docker://etretiakov/workbench-session-complete:jammy-2022.12.09-custom-11.3"
#     threads: 24
#     resources:
#         mem_mb=64000
#     script:
#         "code/cluster_and_annotate_multiplex_graph.py"


# rule markers_logreg:
#     input:
#         h5ad="data/resolved_subregions/{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.h5ad"
#     output:
#         # top20_figures_logreg_violin="output/figures/markers_subregions/rank_genes_groups_clusters_0-logreg-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf",
#         top25_figures_logreg="output/figures/markers_subregions/rank_genes_groups_clusters-logreg-{subregions}-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf",
#         # top5_figures_logreg_heatmap="output/figures/markers_subregions/rank_genes_groups_clusters_2-logreg-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf",
#         # top10_figures_logreg_dotplot="output/figures/markers_subregions/rank_genes_groups_clusters_3-logreg-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf",
#         # top10_figures_logreg_tracksplot="output/figures/markers_subregions/rank_genes_groups_clusters_4-logreg-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf"
#     params:
#         k="{knn}",
#         substr_sign="{substr_level}",
#         res="{fpr}",
#         metric="{metric}",
#         connectivity_model="{connectivity_model}",
#         subregions="{subregions}",
#     container:
#         "docker://etretiakov/workbench-session-complete:jammy-2022.12.09-custom-11.3"
#     threads: 8
#     resources:
#         mem_mb=32000
#     script:
#         "code/subregions_markers_logreg.py"


# rule markers_wilcoxon:
#     input:
#         h5ad="data/resolved_subregions/{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.h5ad"
#     output:
#         # top20_figures_wilcoxon_violin="output/figures/markers_subregions/rank_genes_groups_clusters_0-wilcoxon-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf",
#         top25_figures_wilcoxon="output/figures/markers_subregions/rank_genes_groups_clusters-wilcoxon-{subregions}-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf",
#         # top5_figures_wilcoxon_heatmap="output/figures/markers_subregions/rank_genes_groups_clusters_2-wilcoxon-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf",
#         # top10_figures_wilcoxon_dotplot="output/figures/markers_subregions/rank_genes_groups_clusters_3-wilcoxon-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf",
#         # top10_figures_wilcoxon_tracksplot="output/figures/markers_subregions/rank_genes_groups_clusters_4-wilcoxon-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf"
#     params:
#         k="{knn}",
#         substr_sign="{substr_level}",
#         res="{fpr}",
#         metric="{metric}",
#         connectivity_model="{connectivity_model}",
#         subregions="{subregions}",
#     container:
#         "docker://etretiakov/workbench-session-complete:jammy-2022.12.09-custom-11.3"
#     threads: 8
#     resources:
#         mem_mb=32000
#     script:
#         "code/subregions_markers_wilcoxon.py"


# rule deg_analysis:
#     input:
#         loom="data/resolved_subregions/{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.loom",
#         umap_tsv="data/resolved_subregions/{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}-umap.tsv",
#         pacmap_tsv="data/resolved_subregions/{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}-pacmap.tsv"
#     output:
#         deg_table_logreg="output/tables/deg_results/logreg-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.tsv",
#         deg_table_mast="output/tables/deg_results/mast-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.tsv",
#         # deg_top10_figures_logreg="output/figures/deg_results/logreg-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf",
#         # deg_top10_figures_mast="output/figures/deg_results/mast-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf",
#         deg_top10_figures_logreg_dotplot="output/figures/deg_results/logreg_dotplot-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf",
#         deg_top10_figures_mast_dotplot="output/figures/deg_results/mast_dotplot-{subregions}-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{knn}-sign_{substr_level}-amb_{fpr}.pdf",
#     container:
#         "docker://etretiakov/workbench-session-complete:jammy-2022.12.09-custom-11.3"
#     threads: 32
#     resources:
#         mem_mb=256000
#     script:
#         "code/subregions_deg.R"
