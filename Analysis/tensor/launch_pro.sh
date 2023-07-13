#!/bin/bash

random_number=$((RANDOM))


sbatch <<EOT
#!/bin/bash


##SBATCH -e "errFile"$1".txt"
#SBATCH --mem=250g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=2    # match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x8  # or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=#GPU
#SBATCH --job-name=myjobtest
#SBATCH --time=03:00:00      # hh:mm:ss for the job
#SBATCH --output="/u/msoltaninia/TensorVQS/new_out_g2/$2/$1(qubits)-$2.$random_number2100.out"
##SBATCH --output=${output_name}






### GPU options ###
#SBATCH --gpus-per-node=2

##SBATCH --gpu-bind=none     # or closest
##SBATCH --mail-user=ms69@alfred.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options



module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load anaconda3_gpu  # ... or any appropriate modules
module list  # job documentation and metadata
echo "job is starting on `hostname`"
source activate global_finder310
# source activate root 

# pip install matplotlib
# pip install kahypar
# pip install scipy
# pip install pennylane-sparse
# conda install -c conda-forge cuquantum-python
# conda install -c "conda-forge/label/broken" cuquantum-python
# pip install pennylane-lightning[gpu]

# conda remove -c anaconda cudatoolkit -y
# conda clean --all
# conda install -c anaconda cudatoolkit
# pip install pennylane[torch,cirq,qiskit]
#pip install qiskit
#pip install qiskit-aer-gpu
#pip install qiskit-aqua

# conda install -c conda-forge custatevec
# conda install -c conda-forge cuquantum
# export CUQUANTUM_ROOT=$CONDA_PREFIX

export TF_GPU_ALLOCATOR=cuda_malloc_async
srun python3 t1.py $1 $2
echo "done"
EOT