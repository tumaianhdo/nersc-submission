#!/bin/bash -l
# Loaded Modules
module purge
module load cgpu
module load gcc
module load cuda/10.2.89
module load openmpi
module load python
module load pytorch
export PYTHONUSERBASE="${USER_HOME}/.local/cori/pytorch1.7.1-gpu"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"
module load PrgEnv-gnu
module load craype-haswell

# Dynamic linking
export CRAYPE_LINK_TYPE=dynamic

old_dir=`pwd`
cd $PEGASUS_SCRATCH_DIR

GPU_STRING="0"
for (( i=1; i<${PEGASUS_NUM_GPUS}; i++ )); do
    GPU_STRING+=",${i}"
done
srun -G ${PEGASUS_NUM_GPUS} -n ${PEGASUS_NUM_GPUS} --gpus-per-task=1 --gpu-bind=map_gpu:${GPU_STRING} --hint=nomultithread python $@ > hpo.log

cd $old_dir