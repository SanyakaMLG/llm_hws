#!/usr/bin/env bash
set -euo pipefail

# Включаем API очистки кеша
export VLLM_SERVER_DEV_MODE=1

exec vllm serve Qwen/Qwen3-8B --speculative-config '{"method": "eagle3", "model": "/ephemeral/models/qwen3-8b-eagle3-tengyunw", "num_speculative_tokens": 3}' "$@"
