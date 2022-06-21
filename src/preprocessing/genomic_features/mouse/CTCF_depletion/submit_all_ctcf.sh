#!/bin/bash

name="CTCF_ChIP-seq_CTCF-AID_untreated"
sbatch --export=name="${name}" --job-name="${name} proc" submit_pipeline.sh

name="CTCF_ChIP-seq_CTCF-AID_auxin2days"
sbatch --export=name="${name}" --job-name="${name} proc" submit_pipeline.sh

name="CTCF_ChIP-seq_CTCF-AID_washoff2days"
sbatch --export=name="${name}" --job-name="${name} proc" submit_pipeline.sh

name="Input_for_CTCF_ChIP-seq_CTCF-AID_auxin2days"
sbatch --export=name="${name}" --job-name="${name} proc" submit_pipeline.sh
