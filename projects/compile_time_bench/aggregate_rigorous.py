"""Aggregate the rigorous benchmark log: per (solver, kind), report
n / median / mad / min / max of total and each phase delta."""

import re
import statistics as stats
from collections import defaultdict
from pathlib import Path

LOG = Path(__file__).parent / "results" / "solver_rigorous.txt"
SECTION_RE = re.compile(r"^===== iter=(\d+) / (\w+) / (\S+) =====")
PHASE_RE = re.compile(r"^\s+after_(\w+)\s+total=\s*([\d.]+)\s+delta=\s*([\d.]+)")
TOTAL_RE = re.compile(r"^TOTAL: ([\d.]+)s")

# key: (solver, kind, phase) -> list[float deltas]
phase_data = defaultdict(list)
total_data = defaultdict(list)

current = None
for line in LOG.read_text().splitlines():
    m = SECTION_RE.match(line)
    if m:
        current = (m.group(3), m.group(2))  # (solver, kind)
        continue
    m = PHASE_RE.match(line)
    if m and current is not None:
        phase = m.group(1)
        delta = float(m.group(3))
        phase_data[(current[0], current[1], phase)].append(delta)
        continue
    m = TOTAL_RE.match(line)
    if m and current is not None:
        total_data[current].append(float(m.group(1)))
        current = None  # consume one TOTAL per section

solvers = ["kamino_robot_dr_legs", "robot_anymal_d", "cloth_hanging"]
phases = ["import", "viewer_ctor", "example_init", "step", "render"]

def fmt_stat(xs, w=7, dec=2):
    if not xs:
        return f"{'':>{w}}"
    return f"{stats.median(xs):>{w}.{dec}f}"

def fmt_range(xs, w=15, dec=2):
    if not xs:
        return f"{'':>{w}}"
    return f"{min(xs):.{dec}f}–{max(xs):.{dec}f}"

print("=" * 100)
print("RIGOROUS BENCHMARK — 3 iterations, randomized order")
print("Caches cleared before each cold: Warp source-hash + CUDA NVRTC ComputeCache")
print("=" * 100)
print()

for kind in ("cold", "warm"):
    print(f"--- {kind.upper()} (median over 3 runs; range in brackets) ---")
    header = f"{'phase':<14} | " + " | ".join(
        f"{s.split('_')[0]:>22}" for s in solvers
    )
    print(header)
    print("-" * len(header))
    for phase in phases:
        cells = []
        for s in solvers:
            xs = phase_data[(s, kind, phase)]
            med = stats.median(xs) if xs else 0
            rng = f"[{min(xs):.1f}–{max(xs):.1f}]" if xs else ""
            cells.append(f"{med:>7.2f} {rng:>14}")
        print(f"{phase:<14} | " + " | ".join(cells))
    # totals
    cells = []
    for s in solvers:
        xs = total_data[(s, kind)]
        med = stats.median(xs) if xs else 0
        rng = f"[{min(xs):.1f}–{max(xs):.1f}]" if xs else ""
        cells.append(f"{med:>7.2f} {rng:>14}")
    print(f"{'TOTAL':<14} | " + " | ".join(cells))
    print()

print("--- Raw data ---")
for s in solvers:
    for k in ("cold", "warm"):
        xs = total_data[(s, k)]
        print(f"  {s:<22} {k:<5} totals: {[round(x, 2) for x in xs]}")
