#!/bin/bash -l

old_dir=`pwd`
cd $PEGASUS_SCRATCH_DIR

module load namd
srun -n $PEGASUS_CORES namd2 $@

cd $old_dir
