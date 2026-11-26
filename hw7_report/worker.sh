#!/bin/bash
cd /ephemeral/hw7 && source env.sh && mkdir -p logs
while [ ! -f stop ]; do
  task=$(flock queue.lock sh -c 'head -n1 queue.txt; sed -i 1d queue.txt')
  if [ -z "$task" ]; then sleep 20; continue; fi
  echo "$(date +%T) gpu$1 start $task" >> worker.log
  CUDA_VISIBLE_DEVICES=$1 python runner.py "$task" > "logs/${task//:/_}.log" 2>&1
  rc=$?
  echo "$(date +%T) gpu$1 done $task rc=$rc" >> worker.log
done
