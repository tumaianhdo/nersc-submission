#!/bin/bash -l

old_dir=`pwd`
cd $PEGASUS_SCRATCH_DIR

srun -n $PEGASUS_CORES echo "Hello World!!!" > output

cd $old_dir