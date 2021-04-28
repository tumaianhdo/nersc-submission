#!/bin/bash -l
module purge
module load PrgEnv-gnu
module load craype-haswell
module load python
module load pytorch
export PYTHONUSERBASE="${USER_HOME}/.local/cori/pytorch1.7.1"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"

# Dynamic linking
export CRAYPE_LINK_TYPE=dynamic

old_dir=`pwd`
cd $PEGASUS_SCRATCH_DIR

srun -n 1 python $@

cd $old_dir