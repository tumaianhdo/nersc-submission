#!/usr/bin/env bash

DIR=$(cd $(dirname $0) && pwd)

if [ $# -ne 1 ]; then
    echo "Usage: $0 DAXFILE"
    exit 1
fi

DAXFILE=$1

pegasus-plan --conf pegasus.properties \
    --dir submit \
    --sites nersc \
    --staging-site nersc \
    --output-site nersc \
    --cleanup leaf \
    --force \
    $DAXFILE
