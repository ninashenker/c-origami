#!/bin/bash
#SBATCH -J hic_pipeline
#SBATCH --output=/gpfs/scratch/jt3545/logs/hic_pipeline_%j.log 
#SBATCH --mem=5G
#SBATCH --time=4:00:00
#SBATCH -c2
#SBATCH -p cpu_short

mkdir -p $save_path
cp -r /gpfs/data/tsirigoslab/home/jt3545/hic_prediction/hic-bench-prep/template/hic-bench $save_path

# Move data to new directory

cd ${save_path}/hic-bench/pipelines/hicseq-standard/inputs/

rep1_folder=./fastq/${celltype}-${condition}-${enzyme}-rep1
mkdir -p $rep1_folder
ln -s ${rep1_fastq_folder}/* ${rep1_folder}/
rep2_folder=./fastq/${celltype}-${condition}-${enzyme}-rep2
mkdir -p $rep2_folder
ln -s ${rep2_fastq_folder}/* ${rep2_folder}/

rename _ _R ${rep1_folder}/*
rename _ _R ${rep2_folder}/*

./create-sample-sheet.tcsh mm10

cd ../
sbatch submit_main_run.sh
