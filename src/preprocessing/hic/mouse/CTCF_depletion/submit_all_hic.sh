#!/bin/bash

celltype="mESC"
enzyme="HindIII"

hic_path="/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/revision_proc_data/mouse/CTCF_depletion/Hi-C"
raw_path="/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/revision_raw_data/mouse/CTCF_depletion/Hi-C"


# Untreated group
condition="untreated"

save_path=${hic_path}/${condition}
rep1_fastq_folder=$raw_path/$condition/rep1
rep2_fastq_folder=$raw_path/$condition/rep2
sbatch --export=save_path="${save_path}",condition=$condition,celltype=$celltype,enzyme=$enzyme,rep1_fastq_folder=$rep1_fastq_folder,rep2_fastq_folder=$rep2_fastq_folder --job-name="${condition} proc" submit_pipeline.sh


# Auxin Group (No dash or underscore)
condition="auxin2days"

save_path=${hic_path}/${condition}
rep1_fastq_folder=$raw_path/$condition/rep1
rep2_fastq_folder=$raw_path/$condition/rep2
sbatch --export=save_path="${save_path}",condition=$condition,celltype=$celltype,enzyme=$enzyme,rep1_fastq_folder=$rep1_fastq_folder,rep2_fastq_folder=$rep2_fastq_folder --job-name="${condition} proc" submit_pipeline.sh


# Washoff Group (No dash or underscore)
condition="washoff2days"

save_path=${hic_path}/${condition}
rep1_fastq_folder=$raw_path/$condition/rep1
rep2_fastq_folder=$raw_path/$condition/rep2
sbatch --export=save_path="${save_path}",condition=$condition,celltype=$celltype,enzyme=$enzyme,rep1_fastq_folder=$rep1_fastq_folder,rep2_fastq_folder=$rep2_fastq_folder --job-name="${condition} proc" submit_pipeline.sh

