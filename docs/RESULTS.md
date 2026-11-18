# Results

Hardware: RTX 5060 Ti 16 GB (448 GB/s, PCIe 3.0 x8), Ryzen 5 5500 (6C/12T, Zen 3),
2×16 GB DDR4-3200 dual-channel, NVMe. Model: 35B-A3B MoE, Q4_K_XL, 21,686 MiB, MTP head,
41 blocks, 256 experts (top-8), 11 full-attention blocks.

All numbers below are medians unless noted. "Code" is six distinct coding prompts, "prose" three
explanatory prompts, 256–512 tokens each, thinking off, server chat template.

## Baseline vs 6 GB

| Metric | All experts on GPU (16 GB card) | 6 GB recipe | Δ |
|---|---:|---:|---:|
| Code | 73.8 tok/s | **43.8** | −41% |
| Prose | 58.6 | **39.6** | −32% |
| Sampled (temp 0.6, 8 samples) | ~67 | **44.6** | −33% |
| Decode at 32K | ~64 | **35.3** | −45% |
| Decode at 131K | — | **29.6** | — |
| Prefill 2K / 8K / 32K | 1134 / 1208 / — | **676 / 717 / 707** | −40% |
| Prefill at 130,551 tokens | — | **616** | — |
| Peak VRAM | 14,530 MiB | **5,562 MiB** | −8,968 MiB |
| Process RSS | 13.3 GB | 23.5 GB | +10.2 GB |
| Swap | 176 MiB | 99 MiB | ~0 |

The full-context test filled **130,551 tokens** and generated 128 tokens at 29.6 tok/s, with the
VRAM peak unchanged at 5,562 MiB.

## Variants and verdicts

| Variant | Result | Verdict |
|---|---|---|
| `n-cpu-moe 39` | 6,048 MiB, 48.0 tok/s code | reject: less than 500 MiB headroom |
| `n-cpu-moe 38` | 6,546 MiB, 40.5 tok/s code | reject: does not fit, slower |
| `n-cpu-moe 41` | 5,220 MiB, 42.1 tok/s code | reject: 10–17% slower for 330 MiB |
| MTP depth 0 | 33.7 tok/s | reject: −42% vs depth 1 |
| MTP depth 2 | 48.6 code / 39.9 prose, 6,112 MiB | reject: worse prose, more VRAM |
| `load-mode mmap` | prefill 525 / 648 | reject: −8% prefill |
| `poll 50` | 40.9 tok/s code | reject: −15% |
| `ubatch 4096` | 6,148 MiB | reject: over budget |
| threads 6/8, fixed affinity | 34–48 tok/s code | reject: code regresses; 4/6 wins |
| `ctx-checkpoints 0` | 5,562 MiB, PP 703 | neutral on VRAM; kept 4 for prompt reuse |

## VRAM breakdown (`-lv 5`)

| Item | MiB |
|---|---:|
| `CUDA0 model buffer` | 2,410 |
| KV q8_0 @ 128K | 1,496 |
| Compute + RS state + CUDA context + MTP draft | ~1,656 |
| **Peak** | **5,562** |

Moving `token_embd` to CPU saves 515 MiB. The remaining floor is mostly fixed: one more expert
layer costs ~498 MiB, which is why `n-cpu-moe 39` cannot fit under 5,644 MiB (6,144 − 500).

## Measurement method

- One server per variant on a private port; no other GPU job running.
- Six decode samples per prompt set; median reported.
- Controls on both sides of each candidate; accept only if the candidate beats both and regresses
  nothing by more than ~3%.
- Peak process VRAM sampled every 0.2 s from `nvidia-smi`; RSS and swap from `/proc/<pid>/status`.
- Thermal guard at 75 °C; power profile `performance`.
- Prefill measured at 2K, 8K, 32K; full fill at 131K on the final config.
