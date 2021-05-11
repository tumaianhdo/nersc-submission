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
# cd $PEGASUS_HOME/bin/Pegasus-kickstast

start=$SECONDS
srun -n 1 python $@
end=$SECONDS
echo "Duration: $((end-start)) seconds."

cd $old_dir