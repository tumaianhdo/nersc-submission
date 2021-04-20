#!/bin/bash -l
# Loaded Modules
module purge
module load cgpu
module load gcc
module load cuda/10.2.89
module load openmpi
module load python
module load pytorch
export PATH="${PYTHONUSERBASE}/bin:${PATH}"
module load PrgEnv-gnu
module load craype-haswell

# Dynamic linking
export CRAYPE_LINK_TYPE=dynamic

old_dir=`pwd`
cd $PEGASUS_SCRATCH_DIR

echo "$(which python)" >> output
srun -G ${PEGASUS_NUM_WORKERS} -n ${PEGASUS_NUM_WORKERS} --cpus-per-gpu=2 --gpus-per-task=1 --gpu-bind=map_gpu:0,1 --hint=nomultithread python test.py >> output

cd $old_dir