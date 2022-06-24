#!/bin/bash
#SBATCH -J Download_H3K27ac
#SBATCH --output=/gpfs/scratch/jt3545/logs/H3K27ac_download_%j.log 
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH -c8

save_path="/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/revision_raw_data/human/imr90/ChIP-seq"

# Define condition and rep

# IP
condition="H3K27ac_ChIP-seq_IP"

rep="rep1"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
cd $fastq_path

file_name=ENCFF855STX
curl -L https://www.encodeproject.org/files/${file_name}/@@download/${file_name}.fastq.gz --output ${file_name}.fastq.gz
file_name=ENCFF272HWA
curl -L https://www.encodeproject.org/files/${file_name}/@@download/${file_name}.fastq.gz --output ${file_name}.fastq.gz
file_name=ENCFF956KYS
curl -L https://www.encodeproject.org/files/${file_name}/@@download/${file_name}.fastq.gz --output ${file_name}.fastq.gz
file_name=ENCFF740CFC
curl -L https://www.encodeproject.org/files/${file_name}/@@download/${file_name}.fastq.gz --output ${file_name}.fastq.gz


# input
condition="H3K27ac_ChIP-seq_input"

rep="rep1"
fastq_path="${save_path}/${condition}/${rep}"
mkdir -p $fastq_path
cd $fastq_path

file_name=ENCFF838NYU
curl -L https://www.encodeproject.org/files/${file_name}/@@download/${file_name}.fastq.gz --output ${file_name}.fastq.gz
