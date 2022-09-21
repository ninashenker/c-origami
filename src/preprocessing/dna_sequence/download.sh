#!/bin/bash -e

if [ "$#" -ne 2 ]; then
    echo 'Usage: download.sh <ASSEMBLY> <OUTPUT-DIR>'
    echo '       ASSEMBLY must be one of {hg19, hg38, mm9, mm10}'
    exit 1
fi

ASSEM=$1
DIR=$2

# Download DNA data
if [ "$ASSEM" = "hg38" ] || [ "$ASSEM" = "hg19" ]
then
    somatic=$(seq 1 22)
elif [ "$ASSEM" = "mm10" ] || [ "$ASSEM" = "mm9" ]
then
    somatic=$(seq 1 19)
else
    echo  'ASSEMBLY must be one of {hg19, hg38, mm9, mm10}'
    exit 1
fi

if [ ! -d "$DIR" ];
then
    mkdir -p "$DIR"
fi

for DNA in ${somatic[@]} X Y;
do
	curl "http://hgdownload.soe.ucsc.edu/goldenPath/${ASSEM}/chromosomes/chr${DNA}.fa.gz" -o "$DIR/chr${DNA}.fa.gz"
done

# Get DNA length
python get_chr_lengths.py "$ASSEM" "$DIR"
