#!/usr/bin/env bash
# GPU=0 bash run_bench.sh <config> latency           - benchmark.sh at 120 RPM -> results/latency/<config>
# GPU=0 bash run_bench.sh <config> load 60 120 ...   - load_test.sh per RPM     -> results/load/<config>/<rpm>
set -euo pipefail
cd "$(dirname "$0")"
config=$1
mode=${2:-latency}
gpu=${GPU:-0}
port=$((8000 + gpu))
export CUDA_VISIBLE_DEVICES=$gpu URL=http://localhost:$port VLLM_CACHE_ROOT=/ephemeral/cache/vllm-gpu$gpu/$config
mkdir -p logs
log="logs/$config.$mode.gpu$gpu.server.log"
bash "configs/$config.sh" --port "$port" > "$log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null; wait $server 2>/dev/null || true; until [ "$(nvidia-smi -i $gpu --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 2; done' EXIT
until curl -fs "$URL/health" > /dev/null; do
    kill -0 $server
    sleep 5
done
if [ "$mode" = latency ]; then
    bash benchmark.sh "results/latency/$config"
else
    for rpm in "${@:3}"; do
        [ -d "results/load/$config/$rpm" ] || bash load_test.sh "$rpm" "results/load/$config/$rpm"
    done
fi
