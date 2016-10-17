#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1

module load python
module load cudatoolkit/7.5
module load cudann

cd /home/holdenl/tensorflow/dictionary_learning

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python dl_convergence.py
