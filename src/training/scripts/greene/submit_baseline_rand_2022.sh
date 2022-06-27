#!/bin/bash
#BSUB -J baseline_C.Origami
#SBATCH -c24
#SBATCH --gres=gpu:4
#SBATCH --mem=100gb # Job memory request
#SBATCH --time=2-00:00:00 # Time limit hrs:min:sec
#SBATCH --output=/home/jt3545/job_reports/corigami_baseline_%j.log # Standard output and error log

source ~/.bashrc
conda activate /scratch/jt3545/conda/corigami

python main.py run.name="c-origami-baseline-greene" \
               run.save_root="/home/jt3545/projects" \
               dataset.data_root="/home/jt3545/projects/C.Origami/data" \
               trainer.num_gpu=4 \
               run.seed=2022

