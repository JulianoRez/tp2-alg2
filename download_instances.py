#!/usr/bin/env python3
"""Baixa instâncias (e tours ótimos, quando existem) da TSPLIB95 para data/.

    python download_instances.py                 # conjunto padrão do trabalho
    python download_instances.py eil76 pr76      # instâncias específicas
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

from experiment_config import DEFAULT_INSTANCES

BASE = "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"


def fetch(fname: str, dest: Path) -> bool:
    target = dest / fname
    if target.exists():
        return True
    try:
        with urllib.request.urlopen(BASE + fname, timeout=60) as r:
            target.write_bytes(r.read())
        return True
    except urllib.error.HTTPError:
        return False


def main(names: list[str], dest: Path = Path("data")):
    dest.mkdir(exist_ok=True)
    for name in names:
        ok = fetch(f"{name}.tsp.gz", dest)
        has_tour = fetch(f"{name}.opt.tour.gz", dest)
        print(f"{name:12s} {'ok' if ok else 'FALHOU'}"
              f"{'  (+ tour ótimo)' if has_tour else ''}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_INSTANCES)
