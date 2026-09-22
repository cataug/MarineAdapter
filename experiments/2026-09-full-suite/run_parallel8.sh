#!/bin/bash

cd /home/tahiti/MARINE_DATASETS

N=8

echo "========================================"
echo " STARTING $N PARALLEL GPU SHARDS"
echo "========================================"

for shard in $(seq 0 $((N-1))); do
    (
        export MARINE_NUM_SHARDS=$N
        export MARINE_SHARD=$shard
        export MARINE_SKIP_PREP=1

        python3 MARINE_EXPERIMENTS_V2/run_full_suite.py \
          --stages all \
          --epochs 10 \
          --batch-size 128 \
          --workers 2 \
          --retries 1 \
          2>&1 | sed -u "s/^/[S$shard] /"
    ) &

    pids[$shard]=$!
done

for shard in $(seq 0 $((N-1))); do
    wait ${pids[$shard]}
    echo "[S$shard] finished rc=$?"
done

echo
echo "========================================"
echo " ALL SHARDS FINISHED"
echo "========================================"

python3 - <<'PY'
import os
os.environ.pop("MARINE_SHARD", None)
os.environ.pop("MARINE_NUM_SHARDS", None)

import sys
sys.path.insert(0, "/home/tahiti/MARINE_DATASETS/MARINE_EXPERIMENTS_V2")
import run_full_suite as s

s.collect_results()
print("FINAL RESULTS AGGREGATED")
PY
