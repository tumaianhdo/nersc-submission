#!/usr/bin/env bash

DIR=$(cd $(dirname $0) && pwd)

if [ $# -ne 1 ]; then
    echo "Usage: $0 WORKFLOW_FILE"
    exit 1
fi

WORKFLOW_FILE=$1

pegasus-plan --conf pegasus.properties \
    --dir submit \
    --sites cori \
    --staging-site cori \
    --output-site cori \
    --cleanup leaf \
    --force \
    $WORKFLOW_FILE
