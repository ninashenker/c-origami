#!/bin/bash

save_path="/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/revision_raw_data/mouse/CTCF_depletion/Hi-C"

# Define condition and rep

# Untreated
condition="Hi-C_untreated"
rep="rep1"
srx_id="SRX2873529"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
bash ../../fastq_utils/download_srx.sh $srx_id $fastq_path

rep="rep2"
srx_id="SRX2873530"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
bash ../../fastq_utils/download_srx.sh $srx_id $fastq_path


# Auxin
condition="Hi-C_auxin-2days"
rep="rep1"
srx_id="SRX2873531"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
bash ../../fastq_utils/download_srx.sh $srx_id $fastq_path

rep="rep2"
srx_id="SRX2873532"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
bash ../../fastq_utils/download_srx.sh $srx_id $fastq_path

# Washoff
condition="Hi-C_washoff-2days"
rep="rep1"
srx_id="SRX2873533"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
bash ../../fastq_utils/download_srx.sh $srx_id $fastq_path

rep="rep2"
srx_id="SRX2873534"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
bash ../../fastq_utils/download_srx.sh $srx_id $fastq_path
