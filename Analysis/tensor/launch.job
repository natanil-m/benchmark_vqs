#!/bin/bash

#SBATCH --mem=250g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1    # match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x8  # or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbpj-delta-gpu #msoltaninia
#SBATCH --job-name=myjobtest
#SBATCH --time=01:00:00      # hh:mm:ss for the job
##SBATCH --output=/u/msoltaninia/Gitrep/VQS_QISKIT/HPC/Outputs/%j.out

### GPU options ###
##SBATCH --gpus-per-node=1
##SBATCH --gpu-bind=none     # or closest
##SBATCH --mail-user=ms69@alfred.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options


module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load anaconda3_gpu  # ... or any appropriate modules
#module load openblas
module list  # job documentation and metadata
echo "job is starting on `hostname`"
source activate global_finder310

!pip install tensorcircuit[jax]

#pip install tensorcircuit[tensorflow]

srun python3 runner.py
#srun python3 /u/msoltaninia/Gitrep/VQS_QISKIT/HPC/test.py
#mpirun -np 4 python3 test.py

echo "done"