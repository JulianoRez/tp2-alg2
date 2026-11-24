"""Profiling por subrotina: tempo (perf_counter) e pico de memória (tracemalloc).

mem_peak_<seção> é o pico alocado durante a seção acima do que havia ao entrar.
Tempo e memória devem ser medidos em execuções separadas (tracemalloc é lento).
"""
from __future__ import annotations

import gc
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
            # sem isso, lixo de seções anteriores liberado aqui esconde alocações
            gc.collect()
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

    @contextmanager
    def section(self, name: str):
        yield
