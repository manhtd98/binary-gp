#!/bin/bash

#SBATCH --partition=parallel


datasets=( 'birds' 'emotions' 'enron' 'genbase' 'medical' 'yeast' 'scene' 'rcv1subset1' 'tmc2007_500' )

for dataset in "${datasets[@]}"
do
	echo $dataset
	sbatch Job.sh $dataset
done