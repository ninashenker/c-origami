#!/bin/bash
#SBATCH -J ChIP-seq_pipeline
#SBATCH --output=/gpfs/scratch/jt3545/logs/ChIP-seq_pipeline_%j.log 
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH -c8

module load seqtk/1.3
module load git
module load samtools/1.9

data_path="/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/revision_raw_data/mouse/CTCF_depletion/ChIP-seq"
raw_sample_path="${data_path}/${name}"
save_path="/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/revision_proc_data/mouse/CTCF_depletion/ChIP-seq"
sample_path="${save_path}/${name}"

mkdir -p $sample_path
cd $sample_path

echo "Downloading"
## Download
# Create dir
mkdir -p fastq
cd fastq

ln -s ${raw_sample_path}/rep1/* ./
ln -s ${raw_sample_path}/rep2/* ./

cd ../

echo "All files downloaded, merging"
# Merge fastq files
cat ./fastq/* > ./fastq/reads.fastq.gz

echo "Files merged, subsampling"
# subsample fastq files
zcat ./fastq/reads.fastq.gz | echo "IP reads: $((`wc -l`/4))"
seqtk sample -s 2021 ./fastq/reads.fastq.gz 30000000 | gzip > ./fastq/sub_reads_R1.fastq.gz
zcat ./fastq/sub_reads_R1.fastq.gz | echo "sub IP reads: $((`wc -l`/4))"

echo "Files merged, running pipeline"
# run sns pipeline
mkdir -p sns
cd sns
git clone --depth 1 https://github.com/igordot/sns
sns/generate-settings mm10
sns/gather-fastqs ../fastq
sns/run chip

echo "Pipeline job submitted"
