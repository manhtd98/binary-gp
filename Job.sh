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
pip install -r requirements.txt
python run_gp.py 

#move the error and output file to tmp folder
mkdir -p $homeDir/Out/
mv *.out $homeDir/Out/
mkdir -p $homeDir/Err/
mv *.err $homeDir/Err/

