#!/usr/bin/env bash
# bash worker.sh <gpu>: takes jobs ("<config> latency" or "<config> load <rpm> ...") from queue.txt until it is empty
cd "$(dirname "$0")"
gpu=$1
while true; do
    job=$(flock queue.lock bash -c 'head -1 queue.txt; sed -i 1d queue.txt')
    [ -z "$job" ] && break
    echo "$(date +%T) gpu$gpu start $job"
    GPU=$gpu bash run_bench.sh $job && echo "$(date +%T) gpu$gpu done $job" || echo "$(date +%T) gpu$gpu FAILED $job"
done
