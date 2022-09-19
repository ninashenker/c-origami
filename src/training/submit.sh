#!/bin/bash
#BSUB -J baseline_C.Origami
#SBATCH -c24
#SBATCH -p gpu4_medium,gpu4_long,gpu8_medium,gpu8_long
#SBATCH --gres=gpu:4
#SBATCH --mem=100gb # Job memory request
#SBATCH --time=2-00:00:00 # Time limit hrs:min:sec
#SBATCH --output=logs/baseline_%J.log # Standard output and error log

#conda activate pip_package

python main.py run.name="c-origami-baseline"
