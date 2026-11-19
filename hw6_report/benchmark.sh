#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

url="${URL:-http://localhost:8000}"
data="data/requests.jsonl"
output="${1:-results/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$(dirname "$output")"
mkdir "$output"

# Очищаем кеш
trap 'curl -fsS -X POST "$url/reset_prefix_cache" >/dev/null' EXIT
curl -fsS -X POST "$url/reset_prefix_cache" >/dev/null

aiperf profile --url "$url" --model Qwen/Qwen3-8B \
  --endpoint-type chat --custom-dataset-type raw_payload \
  --input-file "$data" --dataset-sampling-strategy sequential \
  --warmup-request-count 40 --warmup-concurrency 8 \
  --request-count "$(wc -l < "$data")" \
  --request-rate 2 --arrival-pattern constant \
  --streaming --request-timeout-seconds 120 \
  --export-outputs-json --output-artifact-dir "$output/profile"
