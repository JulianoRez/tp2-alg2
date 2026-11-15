"""Instrumentação em nível de subrotina: tempo (perf_counter) e memória (tracemalloc).

Uso:
    prof = Profiler(trace_memory=True)
    with prof.section("mst"):
        ...
    prof.as_dict()  ->  {'time_mst': s, 'mem_peak_mst': bytes, ...}

Observações importantes para a análise:
* tracemalloc instrumenta TODA alocação do Python (inclusive os buffers do
  NumPy), por isso deixa o código bem mais lento. Os experimentos medem tempo
  em execuções SEM tracemalloc e memória numa execução separada COM ele.
* `mem_peak_<secao>` é o pico de memória alocada DURANTE a seção, acima do
  que já estava alocado quando ela começou (memória "extra" da subrotina).
* Seções podem ser aninhadas: antes de zerar o pico (reset_peak) para a seção
  interna, o pico corrente é repassado às seções externas abertas.
"""
from __future__ import annotations

import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter


class Profiler:
    def __init__(self, trace_memory: bool = False):
        self.trace_memory = trace_memory
        self.times: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.mem_peak: dict[str, int] = {}
        self._stack: list[list] = []  # [nome, memória_base, pico_absoluto]

    @contextmanager
    def section(self, name: str):
        if self.trace_memory:
            cur, peak = tracemalloc.get_traced_memory()
            for frame in self._stack:          # preserva pico das seções externas
                frame[2] = max(frame[2], peak)
            tracemalloc.reset_peak()
            self._stack.append([name, cur, cur])
        t0 = perf_counter()
        try:
            yield
        finally:
            self.times[name] += perf_counter() - t0
            self.calls[name] += 1
            if self.trace_memory:
                _, base, abs_peak = self._stack.pop()
                abs_peak = max(abs_peak, tracemalloc.get_traced_memory()[1])
                extra = abs_peak - base
                self.mem_peak[name] = max(self.mem_peak.get(name, 0), extra)

    def as_dict(self) -> dict:
        out = {f"time_{k}": v for k, v in self.times.items()}
        out.update({f"mem_peak_{k}": v for k, v in self.mem_peak.items()})
        return out


class NullProfiler(Profiler):
    """Profiler que não mede nada (evita `if prof:` espalhado pelo código)."""

    @contextmanager
    def section(self, name: str):
        yield
