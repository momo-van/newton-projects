# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Scripted insertion trajectory for Phase 3 solver comparison.

Forces scaled for a 1 kg shaft with 0.4 mm diametral clearance (sliding fit).
Shaft weight = ~10 N.  All insertion forces stay in the 1–3× gravity range,
matching realistic hand-assembly of a clearance-fit spline (not a press-fit).

Controller relationships (gains in sim_dynamics.py):
    F_axial  (N) = DESCENT_K × v_target  = 800 × v_target
    T_torque (Nm) = ROT_D     × ω_target  = 0.1  × ω_target

Physical reference (clearance fit, 1 kg shaft):
    Shaft weight equivalent      : 10 N   → v = 12.5 mm/s
    Light manual axial press     : 15 N   → v = 18.75 mm/s  (search / sit-on-hub)
    Gentle seating push          : 25 N   → v = 31.25 mm/s  (engage + push-in, 2.5× g)
    Friction-demo descent        : 12 N   → v = 15.0 mm/s   (light hold-down while rotating)
    Withdrawal force             : 40 N   → v = 50.0 mm/s   (4× g, overcomes friction + gravity)
    Search rotation (free)       : 0.1 Nm → ω = 60°/s      (standard bench assembly torque)
    Engage rotation (maintain)   : 0.01 Nm→ ω = 5°/s       (minimal, keep keys aligned)
    Torsion test (3% of capacity): 0.05 Nm→ ω = 30°/s      (spline capacity >> 1.6 Nm)

Phase structure (≤ 33.0 s worst case — search exit is event-driven, not time-based):
  Phase         | t / dt range                  | F_axial  | T_rot         | What it shows
  --------------|-------------------------------|----------|---------------|----------------------------------
  vertical_demo | t:  0 – 2 s                   | ±5 N     | 0             | shaft DOF, keys stay at 30° offset
  approach      | t:  2 – 4.5 s                 | 10 N     | 0             | weight-equivalent descent, misaligned
  sit_on_hub    | t:  4.5 – 5.5 s               | 15 N     | 0             | 1 s dwell on hub lands, clear contact
  search        | t:  5.5 s → groove caught     | 15 N     | 0 → 0.064 Nm | 80°/s sweep; exits on position trigger
  hold          | dt: 0 – 1.5 s after catch     | 0 N      | 0             | hands off: shaft settles under gravity
  push          | dt: 1.5 – 2.5 s after catch   | 25 N     | 0             | moderate force seats shaft against cap
  retract       | dt: 2.5 – 3.4 s after catch   | –20 N    | 0.05 Nm       | torque wedges teeth → resists pull-out
  withdraw      | dt: 3.4 – 7.5 s after catch   | –40 N    | 0 (fade)      | torque removed → shaft exits freely
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Pre-search phases end at this fixed time; search is event-driven after this.
_SEARCH_START: float     = 5.5
# Safety cap: give up if groove never caught within this many seconds of searching.
_SEARCH_TIMEOUT: float   = 20.0
# Total time budget from groove catch to end (hold + push + retract + withdraw).
_POST_CATCH_TOTAL: float = 7.5
# Worst-case duration: all pre-search time + full search timeout + post-catch budget.
TOTAL_DURATION: float = _SEARCH_START + _SEARCH_TIMEOUT + _POST_CATCH_TOTAL  # 33.0 s

# ── Groove-catch state (reset on sim reset) ───────────────────────────────────
# Sim time at which the shaft first dropped into the groove.  None = not yet.
_groove_caught_t: float | None = None


def reset_trajectory() -> None:
    """Call this whenever the simulation resets so groove detection restarts."""
    global _groove_caught_t
    _groove_caught_t = None


def expected_duration() -> float:
    """Best-estimate total run time.  Exact once groove is caught; worst-case otherwise."""
    if _groove_caught_t is not None:
        return _groove_caught_t + _POST_CATCH_TOTAL
    return TOTAL_DURATION


@dataclass
class ControlState:
    rotation_dps: float   # commanded rotation speed (°/s); +ve = CCW about Z
    descent_mps:  float   # commanded descent speed (m/s);  +ve = downward
    phase:        str
    done:         bool


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _smooth(t: float) -> float:
    """Smooth-step (3t²-2t³) eliminates impulsive force spikes at transitions."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# Hub bore entrance in world Z (mm).
_HUB_TOP_MM: float = 22.5
# Shaft half-length (mm) — shaft bottom = shaft_centre_z - this value.
_SHAFT_HALF_MM: float = 50.0
# Groove is caught when shaft bottom drops this far below hub top.
_GROOVE_CATCH_MM: float = 5.0
# Detection threshold on shaft CENTRE Z: shaft_z < this → groove caught.
_GROOVE_CATCH_CENTRE_MM: float = _HUB_TOP_MM + _SHAFT_HALF_MM - _GROOVE_CATCH_MM  # 67.5 mm


def get_controls(t: float, shaft_z_mm: float = 87.5) -> ControlState:
    """Return commanded controls at simulation time *t* (seconds).

    ``shaft_z_mm`` is the current shaft body Z position (world, mm).  During
    the search phase it is used to detect groove capture: as soon as the shaft
    drops more than ``_GROOVE_CATCH_MM`` below the hub bore entrance the search
    rotation is killed immediately and seat behaviour begins.

    All phase transitions use smooth-step ramps to avoid force impulses.
    """
    global _groove_caught_t

    # State-based termination — not tied to absolute wall-clock time.
    if _groove_caught_t is not None:
        if (t - _groove_caught_t) >= _POST_CATCH_TOTAL:
            return ControlState(0.0, 0.0, "done", True)
    elif t >= _SEARCH_START + _SEARCH_TIMEOUT:
        return ControlState(0.0, 0.0, "done", True)  # search safety timeout

    # ── Phase 1: vertical DOF demo (0 – 2 s) ─────────────────────────────────
    # F = ±5 N oscillation, NO rotation.
    # Shaft stays at initial half_pitch offset (30°) → guaranteed misaligned for approach.
    if t < 2.0:
        # v_target = ±5 N / 800 = ±0.00625 m/s ≈ ±6 mm/s
        desc = 6.25e-3 * math.sin(2 * math.pi * t / 1.4)
        return ControlState(0.0, desc, "vertical_demo", False)

    # ── Phase 2: approach misaligned (2 – 4.5 s) ─────────────────────────────
    # F = 10 N (shaft weight equivalent).  v = 10/800 = 12.5 mm/s.
    # Shaft descends 15 mm gap in ~1.2 s then presses against hub lands at 10 N.
    if t < 4.5:
        alpha = _smooth((t - 2.0) / 0.4)       # 0.4 s ramp-in
        desc  = _lerp(0.0, 12.5e-3, alpha)
        return ControlState(0.0, desc, "approach", False)

    # ── Phase 3: sit on hub (4.5 – 5.5 s) ───────────────────────────────────
    # F = 15 N (1.5× shaft weight).  v = 15/800 = 18.75 mm/s.
    if t < 5.5:
        alpha = _smooth((t - 4.5) / 0.4)
        desc  = _lerp(12.5e-3, 18.75e-3, alpha)
        return ControlState(0.0, desc, "sit_on_hub", False)

    # ── Phase 4: search (5.5 s → groove caught) ──────────────────────────────
    # Runs until position detection fires for EITHER solver — no timed cutoff.
    # MuJoCo (force-level): descends into bore under 15 N.
    # VBD (position-level): snaps in when teeth geometrically align.
    if _groove_caught_t is None:
        if shaft_z_mm < _GROOVE_CATCH_CENTRE_MM:
            _groove_caught_t = t          # record catch time, fall through below
        else:
            alpha = _smooth((t - 5.5) / 0.8)
            rot   = _lerp(0.0, 80.0, alpha)
            return ControlState(rot, 18.75e-3, "search", False)

    # All remaining phases use time relative to groove catch.
    dt = t - _groove_caught_t

    # ── Phase 5: hold (0 – 1.5 s after catch) ────────────────────────────────
    if dt < 1.5:
        return ControlState(0.0, 0.0, "hold", False)

    # ── Phase 6: push (1.5 – 2.5 s after catch) ──────────────────────────────
    if dt < 2.5:
        alpha = _smooth((dt - 1.5) / 0.3)
        desc  = _lerp(0.0, 31.25e-3, alpha)
        return ControlState(0.0, desc, "push", False)

    # ── Phase 7: retract (2.5 – 3.4 s after catch) ───────────────────────────
    # Pull up 20 N + torque 30°/s — torque wedges teeth, resists axial pull.
    if dt < 3.4:
        alpha_f = _smooth((dt - 2.5) / 0.5)
        desc    = _lerp(31.25e-3, -25.0e-3, alpha_f)
        alpha_r = _smooth((dt - 2.5) / 0.4)
        rot     = _lerp(0.0, 30.0, alpha_r)
        return ControlState(rot, desc, "retract", False)

    # ── Phase 8: withdraw (3.4 s after catch → _POST_CATCH_TOTAL) ────────────
    # Drop torque → shaft exits freely at 40 N (50 mm/s).
    # Done is triggered by the state-based check at the top when dt ≥ _POST_CATCH_TOTAL.
    alpha_f = _smooth((dt - 3.4) / 0.5)
    desc    = _lerp(-25.0e-3, -50.0e-3, alpha_f)
    alpha_r = _smooth((dt - 3.4) / 0.3)
    rot     = _lerp(30.0, 0.0, alpha_r)
    return ControlState(rot, desc, "withdraw", False)
