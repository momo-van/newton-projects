"""Phase-by-phase timing for any newton example, headless (ViewerNull).

Usage:
    python time_example_phases.py <example_key>

Examples:
    python time_example_phases.py kamino_robot_dr_legs
    python time_example_phases.py robot_anymal_d
    python time_example_phases.py cloth_hanging
"""

import importlib
import os
import sys
import time

os.environ.setdefault("NEWTON_CACHE_PATH", r"C:\nc")

EXAMPLE_PKG_MAP = {
    "kamino_robot_dr_legs": "newton.examples.kamino.example_kamino_robot_dr_legs",
    "robot_anymal_d": "newton.examples.robot.example_robot_anymal_d",
    "cloth_hanging": "newton.examples.cloth.example_cloth_hanging",
}


def main():
    if len(sys.argv) < 2:
        print("usage: time_example_phases.py <example_key>")
        sys.exit(2)
    example_key = sys.argv[1]
    if example_key not in EXAMPLE_PKG_MAP:
        print(f"unknown example: {example_key}")
        sys.exit(2)
    module_path = EXAMPLE_PKG_MAP[example_key]

    marks = {"start": time.perf_counter()}

    import newton  # noqa: F401
    import newton.examples  # noqa: F401
    mod = importlib.import_module(module_path)
    Example = mod.Example

    marks["after_import"] = time.perf_counter()

    viewer = newton.viewer.ViewerNull(num_frames=1)
    marks["after_viewer_ctor"] = time.perf_counter()

    # Use the example's own parser so we get its defaults, then force
    # headless-friendly fields.
    parser = Example.create_parser()
    ns = parser.parse_args([])
    ns.viewer = "null"
    ns.num_frames = 1
    ns.quiet = True
    ns.headless = True
    ns.benchmark = False
    ns.realtime = False
    ns.test = False
    # device may be missing in some examples; set if None
    if getattr(ns, "device", None) is None:
        ns.device = None

    example = Example(viewer, ns)
    marks["after_example_init"] = time.perf_counter()

    try:
        example.step()
    except Exception as e:
        print(f"WARN step failed: {e}")
    marks["after_step"] = time.perf_counter()

    try:
        example.render()
    except Exception as e:
        print(f"WARN render failed: {e}")
    marks["after_render"] = time.perf_counter()

    try:
        viewer.close()
    except Exception:
        pass
    marks["end"] = time.perf_counter()

    base = marks["start"]
    prev = base
    order = [
        "after_import",
        "after_viewer_ctor",
        "after_example_init",
        "after_step",
        "after_render",
        "end",
    ]
    print(f"=== PHASE TIMING (s) :: {example_key} ===")
    for name in order:
        t = marks[name]
        print(f"  {name:<24s}  total={t - base:7.2f}  delta={t - prev:7.2f}")
        prev = t
    print(f"TOTAL: {marks['end'] - base:.2f}s")


if __name__ == "__main__":
    main()
