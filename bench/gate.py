#!/usr/bin/env python3
"""Quality gate for one benchmark config.

    python3 bench/gate.py --config configs/example.json [--min-free 550]

1. Long-context retrieval: three random codes hidden in a document at ~50% and ~97% of the
   configured context; all three must be read back exactly.
2. Coding: ten small Python tasks; the generated code is executed against assert-based tests.

The gate starts its own server on the configured port and records peak VRAM/RSS/swap.
Generated code is untrusted: it runs under bwrap with no network when bwrap is available,
otherwise in a throwaway directory with a timeout. Use a container if that is not enough.
"""
import argparse
import json
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench
import tasks

BWRAP = shutil.which("bwrap")


def sandbox(code, tests, timeout=20):
    if not code.strip():
        return False, "no code"
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="gate-"))
    path = tmp / "test.py"
    path.write_text(code + "\n" + tests)
    try:
        if BWRAP:
            cmd = [BWRAP, "--ro-bind", "/usr", "/usr", "--ro-bind-try", "/lib", "/lib",
                   "--ro-bind-try", "/lib64", "/lib64", "--dev", "/dev", "--proc", "/proc",
                   "--tmpfs", "/tmp", "--unshare-net", "--die-with-parent",
                   "--ro-bind", str(path), "/test.py", "python3", "/test.py"]
            cwd = None
        else:
            cmd = [sys.executable, str(path)]
            cwd = str(tmp)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tail = (r.stderr.strip() or r.stdout.strip()).splitlines()
    return r.returncode == 0, "" if r.returncode == 0 else (tail[-1] if tail else "fail")


def chat(conf, port, user, max_tokens, temp=None):
    body = {
        "messages": [{"role": "system", "content": conf.get("system", "")},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": conf.get("temperature", 0.0) if temp is None else temp,
        "top_p": conf.get("top_p", 0.95),
        "top_k": conf.get("top_k", 20),
        "seed": 42,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = bench.request(port, "/v1/chat/completions", body)
    content = r["choices"][0]["message"].get("content") or ""
    return content, r.get("timings", {})


def needle(conf, port, target):
    rng = random.Random(target)
    codes = [f"{rng.choice(['PINE', 'ROOK', 'DUNE', 'KITE', 'MOSS'])}-{rng.randint(1000, 9999)}"
             for _ in range(3)]
    words = bench.sized_prompt(port, target, "n").split()
    for i, code in enumerate(codes):
        pos = int(len(words) * (0.1 + 0.4 * i))
        words.insert(pos, f" IMPORTANT: vault code number {i + 1} is {code}. ")
    doc = " ".join(words)
    question = doc + "\n\nList vault code number 1, 2 and 3 exactly as written, one per line, nothing else."
    answer, t = chat(conf, port, question, 64, temp=0.0)
    found = [c for c in codes if c in answer]
    return {"target": target, "prompt_n": t.get("prompt_n"), "codes": codes,
            "answer": answer[:160], "ok": len(found) == 3,
            "pp_ts": round(t.get("prompt_per_second") or 0, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--min-free", type=int, default=None)
    p.add_argument("--coding", action="store_true", help="run the coding suite (slower)")
    o = p.parse_args()
    conf = bench.load_config(o.config)
    port = conf["port"]
    min_free = o.min_free or conf.get("min_free_mib", 550)

    out = bench.RUNS / f"gate-{conf['name']}-{time.strftime('%Y%m%d-%H%M%S')}"
    bench.RUNS.mkdir(exist_ok=True)
    peak, stop = {"proc": 0, "free": 10 ** 9, "rss": 0, "swap": 0, "temp": 0}, threading.Event()
    log = open(str(out) + ".log", "w")
    proc = subprocess.Popen(bench.server_args(conf), cwd=bench.ROOT, stdout=log, stderr=subprocess.STDOUT)

    def watch():
        while not stop.is_set():
            try:
                _, free, gt = bench.gpu()
                pg = bench.proc_gpu(proc.pid) or 0
                rss, swap = bench.proc_mem(proc.pid)
                ct = bench.cpu_temp() or 0
                peak["proc"] = max(peak["proc"], pg)
                peak["free"] = min(peak["free"], free)
                peak["rss"] = max(peak["rss"], rss)
                peak["swap"] = max(peak["swap"], swap)
                peak["temp"] = max(peak["temp"], ct, gt)
                if max(ct, gt) >= bench.LIMIT_C or (pg and free < min_free):
                    peak["abort"] = f"temp {ct}/{gt} free {free}"
                    proc.terminate()
                    return
            except Exception:
                pass
            stop.wait(0.5)

    threading.Thread(target=watch, daemon=True).start()
    res = {"name": conf["name"], "cmd": bench.server_args(conf)}
    t0 = time.monotonic()
    while proc.poll() is None and time.monotonic() - t0 < 400:
        try:
            if bench.request(port, "/health", timeout=2).get("status") == "ok":
                break
        except Exception:
            pass
        time.sleep(0.5)
    try:
        res["needles"] = [needle(conf, port, int(conf["context"] * 0.5)),
                          needle(conf, port, int(conf["context"] * 0.97) - 200)]
        for n in res["needles"]:
            print("needle", {k: n[k] for k in ("target", "prompt_n", "ok", "pp_ts")}, flush=True)
        if o.coding:
            res["coding"] = []
            for task in tasks.TASKS:
                answer, _ = chat(conf, port, task["prompt"], 2048)
                ok, err = sandbox(tasks.extract_code(answer), task["tests"])
                res["coding"].append({"task": task["id"], "passed": ok, "error": err})
                print(f"  {task['id']}: {'PASS' if ok else 'FAIL ' + err}", flush=True)
            res["coding_passed"] = sum(t["passed"] for t in res["coding"])
    except Exception as e:
        res["error"] = str(e)[-600:]
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
    res.update(peak_proc_vram_mib=peak["proc"], min_free_vram_mib=peak["free"],
               peak_rss_mib=peak["rss"], peak_swap_mib=peak["swap"], max_temp_c=peak["temp"],
               abort=peak.get("abort"))
    out.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res.get(k) for k in (
        "name", "needles_ok", "coding_passed", "peak_proc_vram_mib", "min_free_vram_mib",
        "max_temp_c", "abort", "error")} | {
        "needles_ok": [n["ok"] for n in res.get("needles", [])]}), flush=True)


if __name__ == "__main__":
    main()
