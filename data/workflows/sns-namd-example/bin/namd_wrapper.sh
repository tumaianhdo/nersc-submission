#!/bin/bash -l

module load namd

cd $PEGASUS_SCRATCH

srun -n $PEGASUS_CORES namd2 $@
