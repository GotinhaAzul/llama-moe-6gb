#!/usr/bin/env python3
"""Benchmark a llama.cpp server inside a VRAM budget.

    python3 bench/bench.py --config configs/example.json [--quick] [--full]

The config JSON is a single variant:

    {
      "name": "qwen35-moe-6gb",
      "server": "llama-server",
      "model": "model.gguf",
      "port": 18100,
      "context": 131072,
      "min_free_mib": 550,
      "max_proc_mib": 5644,
      "args": ["--gpu-layers", "99", "--n-cpu-moe", "40", "..."]
    }

`args` must not repeat -m, --port or --ctx-size; the harness adds those.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
LEDGER = ROOT / "ledger.jsonl"
LIMIT_C = 75.0


def request(port, path, data=None, timeout=1800):
    payload = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def load_config(path):
    conf = json.loads(pathlib.Path(path).read_text())
    for key in ("name", "server", "model", "args"):
        if key not in conf:
            raise SystemExit(f"config missing {key!r}: {path}")
    conf.setdefault("port", 18100)
    conf.setdefault("context", 131072)
    return conf


def server_args(conf):
    args = [
        conf["server"], "-m", str((ROOT / conf["model"]) if not os.path.isabs(conf["model"]) else conf["model"]),
        "--alias", conf["name"], "--host", "127.0.0.1", "--port", str(conf["port"]),
        "--ctx-size", str(conf["context"]),
    ]
    return args + [str(a) for a in conf["args"]]


def gpu():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,temperature.gpu",
         "--format=csv,noheader,nounits"], text=True).strip().split(", ")
    return int(out[0]), int(out[1]), float(out[2])


def proc_gpu(pid):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], text=True)
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return int(parts[1])
    return None


def proc_mem(pid):
    rss = swap = 0
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) // 1024
                elif line.startswith("VmSwap:"):
                    swap = int(line.split()[1]) // 1024
    except OSError:
        pass
    return rss, swap


def cpu_temp():
    try:
        d = json.loads(subprocess.check_output(["sensors", "-j"], text=True))
    except Exception:
        return None
    return d.get("k10temp-pci-00c3", {}).get("Tctl", {}).get("temp1_input")


TEMPLATE = {"mode": None}


def chat_prompt(port, user):
    if TEMPLATE["mode"] != "chatml":
        try:
            r = request(port, "/apply-template", {"messages": [{"role": "user", "content": user}],
                                                  "chat_template_kwargs": {"enable_thinking": False}}, timeout=30)
            TEMPLATE["mode"] = "server"
            return r["prompt"]
        except Exception:
            if TEMPLATE["mode"] == "server":
                raise
            TEMPLATE["mode"] = "chatml"
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def filler(n_words, salt=""):
    words = ("The verifier records each build artifact, checks its digest, and writes an audit entry. "
             "A failed check stops the release.").split()
    out, i = [], 0
    while len(out) < n_words:
        out.extend(words)
        out.append(f"Item {salt}{i}.")
        i += 1
    return " ".join(out[:n_words])


def sized_prompt(port, target, salt):
    words = int(target / 1.25)
    text = filler(words, salt)
    n = len(request(port, "/tokenize", {"content": text})["tokens"])
    return filler(int(words * target / max(1, n)), salt)


def complete(port, prompt, n, sampling=None, cache=False, seed=42):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0.0, "seed": seed,
            "ignore_eos": True, "cache_prompt": cache}
    body.update(sampling or {})
    r = request(port, "/completion", body)
    t = r.get("timings", {})
    content = r.get("content") or ""
    generated = t.get("predicted_n") or 0
    accepted = t.get("draft_n_accepted") or 0
    steps = max(1, generated - accepted)
    return {
        "ms_step": round((t.get("predicted_ms") or 0) / steps, 3),
        "tok_step": round(generated / steps, 3),
        "prompt_n": t.get("prompt_n"),
        "pp_ts": t.get("prompt_per_second"),
        "gen_n": generated,
        "tg_ts": t.get("predicted_per_second"),
        "draft_n": t.get("draft_n"),
        "draft_acc": t.get("draft_n_accepted"),
        "sha": hashlib.sha256(content.encode()).hexdigest()[:12],
        "distinct_ratio": round(len(set(content.split())) / max(1, len(content.split())), 3),
    }


CODE = ("Write a complete Go program in one file that starts an HTTP server with graceful shutdown on "
        "SIGINT/SIGTERM, per-request timeouts, structured JSON logging, and a /healthz endpoint. Use only "
        "the standard library. Output only the code, no explanation.")
CODES = [CODE,
         "Implement an LRU cache in Python with O(1) get/put, type hints and a short pytest suite. Output only code.",
         "Write a Rust function that parses an RFC 3339 timestamp without external crates, with unit tests. Output only code.",
         "Write a bash script that rotates log files in a directory, keeps 7 compressed generations and is safe with spaces in names. Output only code.",
         "Write a TypeScript React hook useDebouncedSearch that cancels in-flight fetches with AbortController, plus a usage example. Output only code.",
         "Write a C function that reverses a singly linked list iteratively and recursively, with a main() that tests both. Output only code."]
PROSE = ("Explain how a write-ahead log keeps a small embedded database consistent after a power loss. "
         "Cover the commit record, checkpointing, and recovery scan. Be concrete but concise.")
PROSES = [PROSE,
          "Compare TCP congestion control algorithms Reno, CUBIC and BBR for a datacenter operator.",
          "Describe how a CPU branch predictor works and why mispredictions are expensive."]


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def run(conf, full=False, quick=False):
    RUNS.mkdir(exist_ok=True)
    port = conf["port"]
    min_free = conf.get("min_free_mib", 550)
    max_proc = conf.get("max_proc_mib")
    name = conf["name"]
    out = RUNS / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    res = {"name": name, "config": conf, "cmd": server_args(conf), "time": time.strftime("%F %T")}

    used0, _, _ = gpu()
    if used0 > 600:
        raise SystemExit(f"GPU not idle: {used0} MiB used")

    peak = {"proc": 0, "min_free": 10 ** 9, "rss": 0, "swap": 0, "cpu": 0, "gpu": 0}
    stop, abort = threading.Event(), []
    log = open(str(out) + ".log", "w")
    proc = subprocess.Popen(server_args(conf), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    def watch():
        k = 0
        while not stop.is_set():
            try:
                _, free, gt = gpu()
                pg = proc_gpu(proc.pid) or 0
                rss, swap = proc_mem(proc.pid)
                peak["proc"] = max(peak["proc"], pg)
                peak["min_free"] = min(peak["min_free"], free)
                peak["rss"] = max(peak["rss"], rss)
                peak["swap"] = max(peak["swap"], swap)
                peak["gpu"] = max(peak["gpu"], gt)
                if k % 10 == 0:
                    ct = cpu_temp() or 0
                    peak["cpu"] = max(peak["cpu"], ct)
                    if max(ct, gt) >= LIMIT_C:
                        abort.append(f"thermal {ct}/{gt}")
                        proc.terminate()
                        return
                if pg and free < min_free:
                    abort.append(f"free VRAM {free} < {min_free}")
                    proc.terminate()
                    return
                if pg and max_proc and pg > max_proc:
                    abort.append(f"process VRAM {pg} > {max_proc}")
                    proc.terminate()
                    return
            except Exception:
                pass
            k += 1
            stop.wait(0.2)

    threading.Thread(target=watch, daemon=True).start()
    t0 = time.monotonic()
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError("server exited: " + (out.with_suffix(".log")).read_text()[-1500:])
            try:
                if request(port, "/health", timeout=2).get("status") == "ok":
                    break
            except Exception:
                pass
            if time.monotonic() - t0 > 400:
                raise TimeoutError("load timeout")
            time.sleep(0.25)

        res["load_s"] = round(time.monotonic() - t0, 1)
        res["n_ctx"] = request(port, "/props")["default_generation_settings"]["n_ctx"]
        complete(port, sized_prompt(port, 1024 if quick else 4096, "w"), 4)
        res["pp"] = {}
        for n in ([2048, 8192] if quick else [2048, 8192, 32768]):
            r = complete(port, sized_prompt(port, n, f"p{n}"), 1)
            res["pp"][n] = round(r["pp_ts"], 1)
            print(f"  {name}: PP {r['prompt_n']}: {r['pp_ts']:.0f} t/s", flush=True)

        ngen = 256 if quick else 512
        code = [complete(port, chat_prompt(port, c), ngen) for c in (CODES[:2] if quick else CODES)]
        prose = [complete(port, chat_prompt(port, p), ngen) for p in ([PROSE] if quick else PROSES)]
        res["ms_step_med"] = med([x["ms_step"] for x in code + prose])
        res["tok_step_med"] = med([x["tok_step"] for x in code + prose])
        res["tg_code"] = [round(x["tg_ts"], 2) for x in code]
        res["tg_prose"] = [round(x["tg_ts"], 2) for x in prose]
        res["tg_code_med"] = med(res["tg_code"])
        res["tg_prose_med"] = med(res["tg_prose"])
        res["code_sha"] = code[0]["sha"]
        res["distinct_ratio"] = code[0]["distinct_ratio"]
        acc = [x["draft_acc"] / x["draft_n"] for x in code + prose if x.get("draft_n")]
        res["accept"] = round(statistics.mean(acc), 3) if acc else None
        print(f"  {name}: code {res['tg_code']} prose {res['tg_prose']} acc {res['accept']}", flush=True)

        if not quick:
            samp = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0}
            sampled = [complete(port, chat_prompt(port, p), 512, sampling=samp, seed=s)
                       for s in (1, 2) for p in CODES[:3] + [PROSE]]
            res["tg_sampled"] = [round(x["tg_ts"], 2) for x in sampled]
            res["tg_sampled_mean"] = round(statistics.mean(res["tg_sampled"]), 2)
            deep = complete(port, sized_prompt(port, 32000, "d") + "\n\n" + chat_prompt(port, CODE), 256)
            res["tg_at_32k"] = round(deep["tg_ts"], 2)
            print(f"  {name}: sampled {res['tg_sampled_mean']} tg@32k {res['tg_at_32k']}", flush=True)

        if full and not quick:
            r = complete(port, sized_prompt(port, conf["context"] - 600, "f") + "\n\n" + chat_prompt(port, CODE), 128)
            res["full_fill"] = {"prompt_n": r["prompt_n"], "pp_ts": round(r["pp_ts"], 1),
                                "tg_ts": round(r["tg_ts"], 2)}
            print(f"  {name}: full fill {res['full_fill']}", flush=True)
    except Exception as e:
        res["error"] = str(e)[-800:]
        print(f"  {name}: ERROR {res['error']}", flush=True)
    finally:
        stop.set()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log.close()
        res["abort"] = abort or None
        res["peak_proc_vram_mib"] = peak["proc"]
        res["min_free_vram_mib"] = peak["min_free"]
        res["peak_rss_mib"] = peak["rss"]
        res["peak_swap_mib"] = peak["swap"]
        res["max_cpu_c"] = peak["cpu"]
        res["max_gpu_c"] = peak["gpu"]
        out.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
        with open(LEDGER, "a") as f:
            f.write(json.dumps({k: res.get(k) for k in (
                "time", "name", "n_ctx", "pp", "tg_code_med", "tg_prose_med", "tg_at_32k",
                "full_fill", "accept", "ms_step_med", "tok_step_med", "tg_sampled_mean",
                "peak_proc_vram_mib", "min_free_vram_mib", "peak_rss_mib", "peak_swap_mib",
                "max_cpu_c", "error", "abort")}) + "\n")
        print(json.dumps({k: res.get(k) for k in (
            "name", "tg_code_med", "tg_prose_med", "tg_at_32k", "pp", "accept",
            "ms_step_med", "tok_step_med", "tg_sampled_mean", "peak_proc_vram_mib",
            "min_free_vram_mib", "peak_rss_mib", "peak_swap_mib", "error", "abort")}), flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--full", action="store_true")
    p.add_argument("--quick", action="store_true")
    o = p.parse_args()
    conf = load_config(o.config)
    if socket.socket().connect_ex(("127.0.0.1", conf["port"])) == 0:
        raise SystemExit(f"port {conf['port']} busy")
    run(conf, full=o.full, quick=o.quick)


if __name__ == "__main__":
    main()
