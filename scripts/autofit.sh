#!/usr/bin/env bash
# Try a list of --n-cpu-moe values and keep the first one that loads and stays
# under MAX_PROC_MIB of process VRAM. Nothing is installed or persisted.
#
#   MODEL=./model.gguf LLAMA_SERVER=./llama-server \
#   MAX_PROC_MIB=5644 NCMOE_CANDIDATES="40 41" ./scripts/autofit.sh
#
# Assumes an NVIDIA card visible to nvidia-smi and curl.
set -uo pipefail

MODEL="${MODEL:-./model.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
PORT="${PORT:-18080}"
CTX="${CTX:-131072}"
MAX_PROC_MIB="${MAX_PROC_MIB:-5644}"
NCMOE_CANDIDATES="${NCMOE_CANDIDATES:-40 41}"
THREADS="${THREADS:-4}"
THREADS_BATCH="${THREADS_BATCH:-6}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-2048}"
KV="${KV:-q8_0}"
LOAD_MODE="${LOAD_MODE:-none}"

die() { echo "error: $*" >&2; exit 1; }
[ -f "$MODEL" ] || die "model not found: $MODEL"
command -v "$LLAMA_SERVER" >/dev/null 2>&1 || [ -x "$LLAMA_SERVER" ] || die "llama-server not found: $LLAMA_SERVER"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
command -v curl >/dev/null 2>&1 || die "curl not found"
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && die "port ${PORT} already serving"

args_for() {
  local ncmoe=$1
  printf '%s\n' \
    -m "$MODEL" --alias moe --host 127.0.0.1 --port "$PORT" \
    --ctx-size "$CTX" --gpu-layers 99 --n-cpu-moe "$ncmoe" -ot token_embd=CPU \
    --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-n-ubatch 64 \
    --cache-type-k "$KV" --cache-type-v "$KV" \
    --threads "$THREADS" --threads-batch "$THREADS_BATCH" \
    --batch-size "$BATCH" --ubatch-size "$UBATCH" \
    --flash-attn on --parallel 1 --fit off \
    --load-mode "$LOAD_MODE" --jinja --no-context-shift
}

try_one() {
  local ncmoe=$1 log=/tmp/moe-autofit-$ncmoe.log
  mapfile -t A < <(args_for "$ncmoe")
  echo "== trying --n-cpu-moe $ncmoe (limit ${MAX_PROC_MIB} MiB) =="
  "$LLAMA_SERVER" "${A[@]}" >"$log" 2>&1 &
  local pid=$! started=$SECONDS ok=0
  while [ $((SECONDS - started)) -lt 300 ]; do
    kill -0 "$pid" 2>/dev/null || break
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 1
  done
  if [ "$ok" = 1 ]; then
    local used
    used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits |
           awk -F', ' -v p="$pid" '$1==p{print $2; exit}')
    used=${used:-0}
    echo "   loaded in $((SECONDS - started)) s, process VRAM ${used} MiB"
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    if [ "$used" -le "$MAX_PROC_MIB" ]; then
      WINNER=$ncmoe
      return 0
    fi
    echo "   over the limit; trying a safer placement"
    return 1
  fi
  echo "   did not load; last log lines:"
  tail -5 "$log" | sed 's/^/     /'
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  return 1
}

WINNER=""
for n in $NCMOE_CANDIDATES; do
  if try_one "$n"; then break; fi
done

if [ -z "$WINNER" ]; then
  echo
  echo "No placement fit under ${MAX_PROC_MIB} MiB."
  echo "Try a smaller CTX (e.g. 65536), a lower KV quantization, or a smaller model."
  exit 1
fi

echo
echo "winner: --n-cpu-moe $WINNER"
echo "command:"
printf '  %q ' "$LLAMA_SERVER"; mapfile -t A < <(args_for "$WINNER"); printf '%q ' "${A[@]}"; echo
