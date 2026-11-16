#!/usr/bin/env python3
"""Roda a matriz completa de experimentos (retomável) e gera os CSVs.

Saídas (em results/):
    features.csv   features geométricas/topológicas de cada instância
    runs.jsonl     um registro JSON por execução (fonte da verdade, inclui tours)
    runs.csv       o mesmo em tabela (listas viram texto JSON; sem os tours)
    errors.log     execuções que falharam (são tentadas de novo na próxima vez)

Cada execução roda em um subprocesso (`run_one.py`): isola a medição de
memória (pico RSS por processo) e garante o limite de 30 min mesmo se algo
travar (o B&B já para sozinho no limite e devolve as estatísticas parciais;
o subprocesso só é morto se passar do limite + 5 min de margem).

Exemplos:
    python run_experiments.py --dry-run                  # lista o que vai rodar
    python run_experiments.py --workers 6                # tudo, 6 processos em paralelo
    python run_experiments.py --no-bnb                   # só heurísticas/aproximativos
    python run_experiments.py --configs hk:index --starts nn hull
    python run_experiments.py --csv-only                 # só regenera runs.csv

Se interromper (Ctrl+C), rode de novo: execuções já registradas são puladas.
Paralelismo: cada processo usa 1 núcleo; usar mais workers que núcleos
físicos distorce os tempos. Os tempos de heurísticas são curtos e sensíveis a
ruído: se quiser medições limpas, rode-os com --workers 1 --no-bnb.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import experiment_config as cfg
from tsp.features import instance_features
from tsp.tsplib import find_instance, load_instance

KEY_FIELDS = ("instance", "algo", "start", "bound", "search", "child_order",
              "matching", "gc_enabled")
LIST_FIELDS = ("expanded_by_depth", "pruned_by_depth", "ub_history", "time_total_all")


def key_of(rec: dict) -> tuple:
    return tuple(rec.get(k) for k in KEY_FIELDS)


def make_jobs(args, sizes: dict[str, int]) -> list[dict]:
    jobs = []
    gc_on = not args.no_gc
    if not args.no_heuristics:
        for inst in args.instances:
            for algo, matching in cfg.HEURISTIC_RUNS:
                jobs.append({"instance": inst, "n": sizes[inst], "algo": algo, "start": None, "bound": None,
                             "search": None, "child_order": None,
                             "matching": matching if algo == "christofides" else None,
                             "gc_enabled": gc_on, "time_limit": None})
    if not args.no_bnb:
        bnb = []
        for ci, (bound, order) in enumerate(args.configs):
            for inst in args.instances:
                for si, start in enumerate(args.starts):
                    bnb.append(((ci, sizes[inst], si), {
                        "instance": inst, "n": sizes[inst], "algo": "bnb", "start": start, "bound": bound,
                        "search": args.search, "child_order": order,
                        "matching": "blossom" if start == "christofides" else None,
                        "gc_enabled": gc_on, "time_limit": args.time_limit}))
        jobs += [j for _, j in sorted(bnb, key=lambda x: x[0])]
    return jobs


def job_cmd(job: dict, args) -> list[str]:
    cmd = [sys.executable, "run_one.py", job["instance"], "--algo", job["algo"],
           "--data-dir", args.data_dir]
    if job["algo"] == "bnb":
        cmd += ["--start", job["start"], "--bound", job["bound"], "--search", job["search"],
                "--child-order", job["child_order"], "--time-limit", str(args.time_limit)]
    else:
        cmd += ["--reps", str(args.reps), "--trace-memory"]
        if job["matching"]:
            cmd += ["--matching", job["matching"]]
    if not job["gc_enabled"]:
        cmd.append("--no-gc")
    return cmd


def run_job(job: dict, args) -> tuple[dict | None, str | None]:
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    hard_limit = args.time_limit + 300 if job["algo"] == "bnb" else 3600
    t0 = time.time()
    try:
        p = subprocess.run(job_cmd(job, args), capture_output=True, text=True,
                           timeout=hard_limit, env=env)
    except subprocess.TimeoutExpired:
        rec = dict(job, status="killed", cost=None, time_total=time.time() - t0)
        return rec, None
    if p.returncode != 0:
        return None, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1]), None


def load_done(jsonl: Path) -> set[tuple]:
    if not jsonl.exists():
        return set()
    with jsonl.open() as f:
        return {key_of(json.loads(line)) for line in f if line.strip()}


def write_features(instances: list[str], args, out: Path) -> dict[str, int]:
    """Calcula as features e ATUALIZA features.csv (rodar com um subconjunto
    de instâncias não apaga as linhas das outras)."""
    rows, sizes = [], {}
    for name in instances:
        inst = load_instance(find_instance(name, args.data_dir))
        rows.append(instance_features(inst))
        sizes[name] = inst.n
    pd.DataFrame(rows).sort_values("n").to_csv(out, index=False)
    print(f"features de {len(rows)} instâncias -> {out}")
    return sizes


def build_csv(jsonl: Path, csv: Path):
    """runs.jsonl -> runs.csv. Uma configuração repetida fica só com o último
    registro; custo de execução sem ótimo (timeout) sai como NA."""
    if not jsonl.exists():
        return
    rows = []
    with jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec.pop("tour", None)
            for k in LIST_FIELDS:
                if k in rec:
                    rec[k] = json.dumps(rec[k])
            rows.append(rec)
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"{len(rows)} execuções -> {csv}")


def parse_configs(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for it in items:
        bound, _, order = it.partition(":")
        out.append((bound, order or "bound"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="+", default=cfg.DEFAULT_INSTANCES)
    ap.add_argument("--starts", nargs="+", default=cfg.STARTS)
    ap.add_argument("--configs", nargs="+", default=None,
                    help="configurações do B&B como bound:ordem (ex: hk:index onetree:bound)")
    ap.add_argument("--search", default="dfs", choices=("dfs", "best"))
    ap.add_argument("--time-limit", type=float, default=cfg.TIME_LIMIT)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-bnb", action="store_true")
    ap.add_argument("--no-heuristics", action="store_true")
    ap.add_argument("--no-gc", action="store_true", help="roda tudo com gc desligado")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--csv-only", action="store_true")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    args.configs = parse_configs(args.configs) if args.configs else cfg.BNB_CONFIGS

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    jsonl, csv, errlog = out / "runs.jsonl", out / "runs.csv", out / "errors.log"
    if args.csv_only:
        build_csv(jsonl, csv)
        return

    sizes = write_features(args.instances, args, out / "features.csv")
    done = load_done(jsonl)
    jobs = [j for j in make_jobs(args, sizes) if key_of(j) not in done]
    print(f"{len(jobs)} execuções pendentes ({len(done)} já registradas)")
    if args.dry_run:
        for j in jobs:
            print("  ", " ".join(job_cmd(j, args)[1:]))
        return

    lock = threading.Lock()
    finished = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, j, args): j for j in jobs}
        try:
            for fut in as_completed(futures):
                job = futures[fut]
                rec, err = fut.result()
                with lock:
                    finished += 1
                    tag = f"[{finished}/{len(jobs)}] {job['instance']} {job['algo']}"
                    if job["algo"] == "bnb":
                        tag += f" {job['bound']}/{job['child_order']}/{job['start']}"
                    elif job["matching"]:
                        tag += f" ({job['matching']})"
                    if rec is None:
                        with errlog.open("a") as f:
                            f.write(f"{key_of(job)}\n{err}\n\n")
                        print(f"{tag}: ERRO (ver {errlog})", flush=True)
                        continue
                    with jsonl.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"{tag}: {rec.get('status')} custo={rec.get('cost')} "
                          f"t={rec.get('time_total', 0):.2f}s", flush=True)
        except KeyboardInterrupt:
            print("\ninterrompido: aguardando execuções em andamento terminarem...")
            for f in futures:
                f.cancel()
            raise
        finally:
            build_csv(jsonl, csv)


if __name__ == "__main__":
    main()
