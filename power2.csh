#!/bin/csh
#$ -cwd
#$ -V -S /bin/bash
#$ -N pow2
#$ -q cpu-d.q@*
#$ -pe smp 16
#$ -o eo/power/corr2o
#$ -e eo/power/corr2e
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

micromamba activate env310
python3 experiment.py --size $1 --thr $2 --signal $3 --alpha $4  --workers $5 --iter $6 --seed $7