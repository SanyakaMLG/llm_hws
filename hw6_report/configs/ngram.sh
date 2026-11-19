#!/usr/bin/env bash
set -euo pipefail

# Включаем API очистки кеша
export VLLM_SERVER_DEV_MODE=1

exec vllm serve Qwen/Qwen3-8B --speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4, "prompt_lookup_min": 2}' "$@"
