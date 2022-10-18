#!/bin/sh

#SBATCH --job-name=GPML
#SBATCH --output=out_array_%A_%a.out
#SBATCH --error=out_array_%A_%a.err
#SBATCH -a 1-10
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --partition=parallel

homeDir='/nfs/scratch/nguyenba/Works/MultiTreeGPML'
cd $homeDir/Commands/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py37

# 1-dataset
outDir=$homeDir/Results/$1/
mkdir -p $outDir

python main.py $1 $outDir ${SLURM_ARRAY_TASK_ID}

#move the error and output file to tmp folder
mkdir -p $homeDir/Out/
mv *.out $homeDir/Out/
mkdir -p $homeDir/Err/
mv *.err $homeDir/Err/

