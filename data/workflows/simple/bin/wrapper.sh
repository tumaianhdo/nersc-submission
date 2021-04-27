#!/bin/bash -l
module purge
module load PrgEnv-gnu
module load craype-haswell
module load python
module load pytorch
export PATH="${PYTHONUSERBASE}/bin:${PATH}"

# Dynamic linking
export CRAYPE_LINK_TYPE=dynamic

old_dir=`pwd`
cd $PEGASUS_SCRATCH_DIR

echo $(which python)
python -c "import cv2"
srun -n $PEGASUS_CORES python -c "import cv2"
srun -n $PEGASUS_CORES python $@ > output

cd $old_dir