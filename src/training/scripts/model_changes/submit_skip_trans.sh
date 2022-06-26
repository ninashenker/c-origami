#!/bin/bash
#BSUB -J skip_trans_C.Origami
#SBATCH -c24
#SBATCH -p gpu4_medium,gpu4_long,gpu8_medium,gpu8_long
#SBATCH --gres=gpu:4
#SBATCH --mem=100gb # Job memory request
#SBATCH --time=3-00:00:00 # Time limit hrs:min:sec
#SBATCH --output=/gpfs/scratch/jt3545/logs/corigami_dilated_%j.log # Standard output and error log

source ~/.bashrc
conda activate /gpfs/data/tsirigoslab/home/jt3545/hic_prediction/conda/corigami

python main.py run.name="c-origami-skip-trans" model="ConvSkipTransModel" logger.wandb.mode="off"
