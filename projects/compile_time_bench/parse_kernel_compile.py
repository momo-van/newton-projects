"""Parse the Warp `Module X took N ms (compiled)` lines from the bench log,
grouped by (run kind / example) section, and print a per-kernel breakdown.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

LOG = Path(__file__).parent / "results" / "solver_compile_phases.txt"
SECTION_RE = re.compile(r"^===== (\w+) / (\S+) =====")
MOD_RE = re.compile(r"^Module (.+?)\s+[0-9a-f]+ load on device .* took ([\d.]+) ms\s+\((compiled|cached)\)")

sections = defaultdict(list)  # (kind, example) -> [(name, ms, state)]
current = None
for line in LOG.read_text().splitlines():
    m = SECTION_RE.match(line)
    if m:
        current = (m.group(1), m.group(2))
        continue
    m = MOD_RE.match(line)
    if m and current is not None:
        sections[current].append((m.group(1), float(m.group(2)), m.group(3)))

# Order
example_order = ["kamino_robot_dr_legs", "robot_anymal_d", "cloth_hanging"]
KIND = "cold"

print(f"=== KERNEL COMPILE BREAKDOWN ({KIND} runs) ===\n")
for ex in example_order:
    key = (KIND, ex)
    items = sections.get(key, [])
    compiled = [(n, ms) for (n, ms, st) in items if st == "compiled"]
    cached = [(n, ms) for (n, ms, st) in items if st == "cached"]
    total_compiled = sum(ms for _, ms in compiled) / 1000.0
    total_cached = sum(ms for _, ms in cached) / 1000.0

    print(f"--- {ex} ---")
    print(f"  modules compiled : {len(compiled)}  ({total_compiled:6.2f} s total)")
    print(f"  modules cached   : {len(cached)}  ({total_cached:6.2f} s total)")
    print(f"  top-10 kernels by compile time:")
    compiled.sort(key=lambda kv: -kv[1])
    for name, ms in compiled[:10]:
        short = name if len(name) <= 70 else name[:67] + "..."
        print(f"    {ms/1000:6.2f} s   {short}")
    # bucket by prefix
    bucket = defaultdict(float)
    for name, ms in compiled:
        # group by first 3 dotted segments of module path or by leading word
        parts = name.split(".")
        if len(parts) >= 3:
            head = ".".join(parts[:3])
        else:
            head = parts[0].split("_")[0]
        bucket[head] += ms
    print(f"  buckets (sum compile time):")
    for head, ms in sorted(bucket.items(), key=lambda kv: -kv[1]):
        if ms / 1000.0 < 0.5:
            continue
        print(f"    {ms/1000:6.2f} s   {head}")
    print()
