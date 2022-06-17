#!/bin/bash

ASSEM=hg38
#DIR=./downloads
DIR=/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/C.Origami/data/hg38/dna_sequence

# Create dir
mkdir -p $DIR

# Download DNA data
a1=$(seq 1 22)
a2=("X" "Y")
total=(${a1[@]} ${a2[@]})
for DNA in ${total[@]};
do
	curl "http://hgdownload.soe.ucsc.edu/goldenPath/${ASSEM}/chromosomes/chr${DNA}.fa.gz" -o "$DIR/chr${DNA}.fa.gz"
done

# Get DNA length
python get_chr_lengths.py $ASSEM
