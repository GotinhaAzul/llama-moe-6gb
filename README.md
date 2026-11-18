# 35B-A3B MoE at 128K context in 6 GB of VRAM

A llama.cpp recipe and benchmark harness for running a large sparse MoE model inside a **6 GB**
VRAM budget while keeping a **131,072-token context**, with no extra quantization of weights or KV.

Measured on a single desktop (RTX 5060 Ti 16 GB, Ryzen 5 5500, 2×16 GB DDR4-3200):

| Metric | Result |
|---|---:|
| Model | 35B-A3B MoE, Q4_K_XL, 21.2 GiB, MTP head |
| Context | **131,072** |
| Peak VRAM | **5,562 MiB** (582 MiB headroom under 6 GB) |
| Decode, code (median of 6 prompts) | **43.8 tok/s** |
| Decode, prose (3 prompts) | **39.6 tok/s** |
| Decode, sampled (temp 0.6) | **44.6 tok/s** |
| Decode at 32K depth | **35.3 tok/s** |
| Decode at 131K depth | **29.6 tok/s** |
| Prefill 2K / 8K / 32K | **676 / 717 / 707 tok/s** |
| Prefill at 130,551 tokens | **616 tok/s** |

No weights are included. Bring your own GGUF.

## The recipe

```bash
llama-server \
  -m model.gguf \
  --ctx-size 131072 \
  --gpu-layers 99 \
  --n-cpu-moe 40 \
  -ot token_embd=CPU \
  --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-n-ubatch 64 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --threads 4 --threads-batch 6 \
  --batch-size 2048 --ubatch-size 2048 \
  --flash-attn on --parallel 1 --fit off \
  --load-mode none --jinja --no-context-shift
```

Add `--mmproj projector.gguf --no-mmproj-offload` only if the model has vision.

Requires a recent llama.cpp build. These measurements use **b12478**; `--load-mode`,
`--spec-type draft-mtp` and `-ot` need a build from that era.

## Why it fits

The GGUF is 21,686 MiB. Three properties of the architecture make a 6 GB budget possible:

1. **Hybrid attention.** Only 11 of the 41 blocks are full attention; the other 30 are
   linear-attention/SSM. At 128K the q8_0 KV cache is **1,496 MiB**, not the 12+ GiB a 41-layer
   full-attention model would need.
2. **Sparse MoE.** 256 experts, top-8. Each token reads roughly 3% of the expert weights, so
   40 of the 41 expert layers can live in host RAM and cost about 1 GiB of reads per verify step.
3. **Built-in MTP.** The model has an MTP head (`nextn_predict_layers=1`). Draft depth 1 yields
   about 1.9 tokens per verify step; depth 2 does not pay for itself because the union of experts
   grows almost linearly with the number of rows.

The placement that fits:

| Item | MiB |
|---|---:|
| GPU model buffer (dense + head + shared experts + MTP layer) | 2,410 |
| KV q8_0 @ 128K | 1,496 |
| Compute buffers, recurrent state, CUDA context, draft context | ~1,656 |
| **Peak** | **5,562** |

`-ot token_embd=CPU` moves the token embedding (515 MiB) to the CPU. It is a one-row lookup per
token, so the cost is negligible, and it buys enough headroom to keep the last expert layer on GPU.

## Tuning knobs

| Flag | Meaning | How to adapt |
|---|---|---|
| `--n-cpu-moe 40` | expert layers kept on CPU | total blocks − 1 keeps the MTP block on GPU; lower it until the card rejects the load |
| `-ot token_embd=CPU` | embedding on CPU | keep; frees ~0.5 GiB on most models |
| `--spec-type draft-mtp` | use the MTP head | only if the GGUF has one |
| `--spec-draft-n-max 1` | draft depth | 1 when experts are on CPU; 2–4 when they are on GPU |
| `--ctx-size 131072` | context length | KV scales linearly with it |
| `--cache-type-k/v q8_0` | KV quantization | q8_0 is the sweet spot; f16 if VRAM allows |
| `--load-mode none` | read weights into RAM | `mmap` avoids a second copy but was 8% slower in prefill here |
| `--threads 4` | CPU expert threads | physical cores − 2, leaving room for the main thread |

`n-cpu-moe 39` was faster (48 tok/s) but used 6,048 MiB and left less than 500 MiB of headroom;
`n-cpu-moe 41` used 5,220 MiB but lost 10–17% of throughput. `40` is the best fit under a hard
6 GB budget with 500 MiB of headroom.

## Reproduce

```bash
python3 bench/bench.py --config configs/example.json --full
python3 bench/gate.py --config configs/example.json   # long-context needles + coding tasks
```

`bench.py` starts its own server on a private port, measures prefill at 2K/8K/32K, decode on six
code and three prose prompts at greedy and profile sampling, decode at 32K, an optional full
context fill, and tracks peak process VRAM, RSS and swap. `--max-proc` aborts a variant that does
not fit.

Edit `configs/example.json` to point at your `llama-server` binary and GGUF, then adjust `args`.

## Caveats

- These numbers are from one machine and one model. Different RAM bandwidth, PCIe width or CPU
  will move the decode/prefill numbers; the VRAM budget is the portable part.
- The quality gate (needle retrieval at 65K/127K plus a coding suite) was interrupted in the
  original run. The 131K fill was verified end-to-end, but retrieval accuracy at full context was
  not certified. Run `bench/gate.py` before relying on 128K for critical work.
- Moving experts between CPU and GPU changes numerics slightly (different GEMM paths), so greedy
  output is not bit-identical to an all-GPU run.

## License

MIT. See `LICENSE`.
