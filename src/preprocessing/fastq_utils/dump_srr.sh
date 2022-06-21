#!/bin/bash
#SBATCH -c4
#SBATCH -p cpu_medium,cpu_long
#SBATCH --mem=8gb # Job memory request
#SBATCH --time=48:00:00 # Time limit hrs:min:sec
#SBATCH --output=/gpfs/data/tsirigoslab/home/jt3545/hic_compare_project/jobreports/srr_dump_%j.log # Standard output and error log

module load sratoolkit/2.10.9
module load pigz

fasterq-dump --split-files $SRR -O $SAVE -t "${SAVE}" 
pigz -p8 "${SAVE}/${SRR}_1.fastq"
pigz -p8 "${SAVE}/${SRR}_2.fastq"
pigz -p8 "${SAVE}/${SRR}.fastq"
