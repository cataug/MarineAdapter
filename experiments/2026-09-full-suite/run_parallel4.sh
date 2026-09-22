#!/bin/bash

cd /home/tahiti/MARINE_DATASETS

echo "================================================"
echo " STARTING 4 GPU SHARDS"
echo "================================================"

for shard in 0 1 2 3; do
    (
        export MARINE_NUM_SHARDS=4
        export MARINE_SHARD=$shard
        export MARINE_SKIP_PREP=1

        python3 MARINE_EXPERIMENTS_V2/run_full_suite.py \
          --stages all \
          --epochs 10 \
          --batch-size 128 \
          --workers 4 \
          --retries 1 \
          2>&1 | sed -u "s/^/[SHARD $shard] /"
    ) &

    pids[$shard]=$!
done

failed=0

for shard in 0 1 2 3; do
    wait ${pids[$shard]}
    rc=$?

    echo "[SHARD $shard] finished rc=$rc"

    if [ "$rc" -ne 0 ]; then
        failed=1
    fi
done

echo
echo "================================================"
echo " ALL SHARDS FINISHED"
echo "================================================"

python3 - <<'PY'
import sys
sys.path.insert(0, "/home/tahiti/MARINE_DATASETS/MARINE_EXPERIMENTS_V2")

import run_full_suite as s
s.collect_results()

print("FINAL AGGREGATION COMPLETE")
PY

echo
find MARINE_EXPERIMENTS_V2/FULL_SUITE \
  -name result.json | wc -l

echo
echo "Final:"
echo "MARINE_EXPERIMENTS_V2/FULL_SUITE/all_results_full.csv"

if [ "$failed" -ne 0 ]; then
    echo "WARNING: one or more shards exited non-zero."
fi
