#!/bin/bash

save_path="/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/revision_raw_data/mouse/CTCF_depletion/ChIP-seq"

# Define condition and rep

# Untreated
condition="CTCF_ChIP-seq_CTCF-AID_untreated"
rep="rep1"
srx_id="SRX2790845"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
bash ../../fastq_utils/download_srx.sh $srx_id $fastq_path

rep="rep2"
srx_id="SRX2790848"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
bash ../../fastq_utils/download_srx.sh $srx_id $fastq_path


# Auxsin
condition="CTCF_ChIP-seq_CTCF-AID_auxin2days"
rep="rep1"
srx_id="SRX2790846"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
bash ../../fastq_utils/download_srx.sh $srx_id $fastq_path

rep="rep2"
srx_id="SRX2790849"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
bash ../../fastq_utils/download_srx.sh $srx_id $fastq_path

# Washoff
condition="CTCF_ChIP-seq_CTCF-AID_washoff2days"
rep="rep1"
srx_id="SRX2790847"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
bash ../../fastq_utils/download_srx.sh $srx_id $fastq_path

rep="rep2"
srx_id="SRX2790850"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
bash ../../fastq_utils/download_srx.sh $srx_id $fastq_path

# Input 
condition="Input_for_CTCF_ChIP-seq_CTCF-AID_auxin2days"
rep="rep1"
srx_id="SRX2790851"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
bash ../../fastq_utils/download_srx.sh $srx_id $fastq_path

rep="rep2"
srx_id="SRX2790852"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
bash ../../fastq_utils/download_srx.sh $srx_id $fastq_path
