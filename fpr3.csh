#!/bin/csh
#$ -cwd
#$ -V -S /bin/bash
#$ -N p3
#$ -q cpu-b.q@*
#$ -pe smp 16
#$ -o eo/fpr/crr32o
#$ -e eo/fpr/crr32e
export OMP_NUM_THREADS=1

micromamba activate env310
python3 experiment.py --size $1 --thr $2 --signal $3 --alpha $4  --workers $5 --iter $6 --seed $7