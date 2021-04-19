#!/bin/bash -l

old_dir=`pwd`
cd $PEGASUS_SCRATCH_DIR

srun -G 2 -n $PEGASUS_CORES --gpus-per-task=1 --gpu-bind=map_gpu:0,1 --hint=nomultithread echo "Hello World!!!" > output

cd $old_dir