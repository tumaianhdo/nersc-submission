#!/bin/bash -l
module purge
module load PrgEnv-gnu
module load craype-haswell
module load python
module load pytorch
export PYTHONUSERBASE="${USER_HOME}/.local/cori/pytorch1.7.1"
echo "PYTHONUSERBASE = ${PYTHONUSERBASE}"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"
echo "PATH = $PATH"

# Dynamic linking
export CRAYPE_LINK_TYPE=dynamic

old_dir=`pwd`
cd $PEGASUS_SCRATCH_DIR

module li
echo $(which python)
pip freeze 
pip freeze | grep opencv
pip show opencv-python
python -c "import cv2"
srun -n $PEGASUS_CORES python -c "import cv2"
srun -n $PEGASUS_CORES python $@ > output

cd $old_dir