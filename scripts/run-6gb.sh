#!/usr/bin/env bash
# Serve a 35B-A3B MoE inside a 6 GB VRAM budget at 128K context.
# Override anything with environment variables.
#
#   MODEL=./model.gguf LLAMA_SERVER=./llama-server ./scripts/run-6gb.sh
#
set -euo pipefail

MODEL="${MODEL:-./model.gguf}"
MMPROJ="${MMPROJ:-}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX="${CTX:-131072}"
NCMOE="${NCMOE:-40}"
THREADS="${THREADS:-4}"
THREADS_BATCH="${THREADS_BATCH:-6}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-2048}"
NMAX="${NMAX:-1}"
KV="${KV:-q8_0}"
LOAD_MODE="${LOAD_MODE:-none}"
CACHE_RAM="${CACHE_RAM:-2048}"
CTX_CHECKPOINTS="${CTX_CHECKPOINTS:-4}"

args=(
  -m "$MODEL" --alias moe
  --host "$HOST" --port "$PORT"
  --ctx-size "$CTX"
  --gpu-layers 99 --n-cpu-moe "$NCMOE"
  -ot token_embd=CPU
  --spec-type draft-mtp --spec-draft-n-max "$NMAX" --spec-draft-n-ubatch 64
  --cache-type-k "$KV" --cache-type-v "$KV"
  --threads "$THREADS" --threads-batch "$THREADS_BATCH"
  --batch-size "$BATCH" --ubatch-size "$UBATCH"
  --flash-attn on --parallel 1 --fit off
  --cache-ram "$CACHE_RAM" --ctx-checkpoints "$CTX_CHECKPOINTS"
  --load-mode "$LOAD_MODE" --jinja --no-context-shift
  --reasoning-format deepseek --reasoning on
)
if [ -n "$MMPROJ" ]; then
  args+=(--mmproj "$MMPROJ" --no-mmproj-offload)
fi

printf '%q ' "$LLAMA_SERVER" "${args[@]}"; echo
if [ "${DRY_RUN:-0}" = "1" ]; then exit 0; fi
exec "$LLAMA_SERVER" "${args[@]}"
