#!/usr/bin/env bash
set -euo pipefail

# Включаем API очистки кеша
export VLLM_SERVER_DEV_MODE=1

exec vllm serve Qwen/Qwen3-8B --quantization fp8 --speculative-config '{"method": "dflash", "model": "/ephemeral/models/qwen3-8b-dflash", "num_speculative_tokens": 11}' --attention-backend flash_attn --max-num-batched-tokens 32768 --max-num-seqs 256 --gpu-memory-utilization 0.85 "$@"
