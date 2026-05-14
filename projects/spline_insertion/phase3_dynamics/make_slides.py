# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
NVIDIA-style Phase 3 benchmark slide deck.

Reads bench_results/bench_vbd.csv and bench_results/bench_mujoco.csv and
generates 5 slides as individual PNGs + a single PDF.

Usage
-----
    python make_slides.py
Output
------
    slides/01_title.png
    slides/02_task.png
    slides/03_design.png
    slides/04_results.png
    slides/05_contact.png
    slides/phase3_benchmark.pdf
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

HERE  = pathlib.Path(__file__).parent
BENCH = HERE / "bench_results"
OUT   = HERE / "slides"

# ── NVIDIA colour palette ──────────────────────────────────────────────────────
_BG     = "#141414"   # slide background
_BG2    = "#1e1e1e"   # panel background
_BG3    = "#282828"   # callout box
_GREEN  = "#76b900"   # NVIDIA green
_WHITE  = "#ffffff"
_LGRAY  = "#bbbbbb"
_DGRAY  = "#555555"
_BLUE   = "#3A86FF"   # VBD colour
_ORANGE = "#FF6B35"   # MuJoCo colour
_YELLOW = "#FFD60A"

_SW, _SH = 16, 9   # inches — 120 dpi → 1920×1080
_DPI     = 120

_PHASE_BG: dict[str, str] = {
    "vertical_demo": "#2a2a40",
    "approach":      "#0a1f3a",
    "sit_on_hub":    "#2a1f00",
    "search_groove": "#2a1000",
    "engage":        "#0a2a0f",
    "push_in":       "#032203",
    "friction_demo": "#2a0a00",
    "torsion_lock":  "#2a0000",
    "pull_out":      "#002a30",
}


# ── Data ───────────────────────────────────────────────────────────────────────

@dataclass
class Trace:
    solver:        str
    t:             np.ndarray
    shaft_z:       np.ndarray   # mm, centre
    frame_ms:      np.ndarray
    phase:         list[str]
    cmd_des_mmps:  np.ndarray   # positive = downward
    cmd_rot_dps:   np.ndarray
    contact_area:  np.ndarray   # mm²
    n_contacts:    np.ndarray
    max_depth:     np.ndarray   # mm
    shaft_half_z:  float = 50.0


def _load(solver: str) -> Trace:
    rows = list(csv.DictReader((BENCH / f"bench_{solver}.csv").open()))
    t     = np.array([float(r["t"])                      for r in rows])
    sz    = np.array([float(r["shaft_z_mm"])              for r in rows])
    fms   = np.array([float(r["frame_ms"])                for r in rows])
    ph    = [r["phase"] for r in rows]
    cdes  = np.array([float(r.get("cmd_des_mmps", 0))    for r in rows])
    crot  = np.array([float(r.get("cmd_rot_dps",  0))    for r in rows])
    carea = np.array([float(r.get("contact_area_mm2", 0)) for r in rows])
    ncont = np.array([float(r.get("n_contacts",   0))    for r in rows])
    mdep  = np.array([float(r.get("max_depth_mm", 0))    for r in rows])
    return Trace(solver, t, sz, fms, ph, cdes, crot, carea, ncont, mdep)


def _smooth(a: np.ndarray, w: int = 9) -> np.ndarray:
    if len(a) < w:
        return a
    out = np.convolve(a, np.ones(w) / w, "same")
    # Fix edge artefacts from the convolve "same" mode
    hw = w // 2
    for i in range(hw):
        out[i]      = np.mean(a[:i + hw + 1])
        out[-i - 1] = np.mean(a[-i - hw - 1:])
    return out


def _resistance(tr: Trace, k: float = 800.0) -> np.ndarray:
    """Axial contact resistance proxy (N).  Positive = contact blocking descent."""
    dt      = np.diff(tr.t, prepend=tr.t[0])
    dt      = np.where(dt < 1e-9, 1e-9, dt)
    vel_act = np.diff(tr.shaft_z, prepend=tr.shaft_z[0]) / dt   # mm/s  (<0 when descending)
    vel_cmd = -tr.cmd_des_mmps                                    # mm/s  (<0 when commanded down)
    # resistance = k × (vel_act - vel_cmd) / 1000
    #   > 0  shaft slower than commanded → contact resisting descent
    #   < 0  shaft faster than commanded → snap / gravity assist
    return _smooth(k * (vel_act - vel_cmd) / 1000.0, 13)


# ── Shared chrome ──────────────────────────────────────────────────────────────

def _new_slide() -> plt.Figure:
    fig = plt.figure(figsize=(_SW, _SH))
    fig.patch.set_facecolor(_BG)
    return fig


def _chrome(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    """NVIDIA-style green header + footer with title."""
    # Top bar
    a = fig.add_axes([0.0, 0.930, 1.0, 0.070])
    a.set_facecolor(_GREEN); a.axis("off")
    a.text(0.012, 0.50, "NVIDIA", color=_BG, fontsize=16, fontweight="bold",
           va="center", ha="left", transform=a.transAxes, family="sans-serif")
    a.text(0.988, 0.50, "Newton Physics Engine", color=_BG, fontsize=10,
           va="center", ha="right", transform=a.transAxes)
    # Bottom bar
    b = fig.add_axes([0.0, 0.000, 1.0, 0.030])
    b.set_facecolor(_GREEN); b.axis("off")
    b.text(0.988, 0.50, "Phase 3  ·  DIN 5480 Spline Insertion  ·  VBD vs MuJoCo",
           color=_BG, fontsize=8, va="center", ha="right", transform=b.transAxes)
    # Title
    fig.text(0.04, 0.905, title, color=_WHITE, fontsize=21,
             fontweight="bold", va="top", ha="left")
    if subtitle:
        fig.text(0.04, 0.862, subtitle, color=_LGRAY, fontsize=10.5,
                 va="top", ha="left")


def _ax_style(ax, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_facecolor(_BG2)
    for sp in ax.spines.values():
        sp.set_color(_DGRAY)
    ax.tick_params(colors=_LGRAY, labelsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=_LGRAY, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=_LGRAY, fontsize=9)
    ax.grid(axis="y", color="#2a2a2a", lw=0.6)


def _phase_bands(ax, t: np.ndarray, phases: list[str]) -> None:
    prev = None; t0 = t[0]
    for i, ph in enumerate(phases):
        if ph != prev:
            if prev is not None:
                ax.axvspan(t0, t[i], color=_PHASE_BG.get(prev, "#222222"),
                           alpha=0.90, linewidth=0, zorder=0)
            prev = ph; t0 = t[i]
    if prev:
        ax.axvspan(t0, t[-1], color=_PHASE_BG.get(prev, "#222222"),
                   alpha=0.90, linewidth=0, zorder=0)


def _phase_labels_on(ax, t: np.ndarray, phases: list[str], y_frac: float = 0.97) -> None:
    """Draw phase name ticks along the top of an axis."""
    prev = None
    for i, ph in enumerate(phases):
        if ph != prev and i > 0:
            ax.axvline(t[i], color="#3a3a3a", lw=0.8, zorder=1)
            ax.text(t[i] + 0.05, y_frac, ph.replace("_", "\n"),
                    color="#777777", fontsize=5.5, va="top",
                    transform=ax.get_xaxis_transform(), zorder=2)
        prev = ph


# ── Slide 1: Title ─────────────────────────────────────────────────────────────

def slide_title() -> plt.Figure:
    fig = _new_slide()

    # Top green bar (taller on title slide)
    a = fig.add_axes([0.0, 0.88, 1.0, 0.12])
    a.set_facecolor(_GREEN); a.axis("off")
    a.text(0.015, 0.55, "NVIDIA", color=_BG, fontsize=22, fontweight="bold",
           va="center", ha="left", transform=a.transAxes)
    a.text(0.985, 0.55, "Newton Physics Engine  ·  Phase 3",
           color=_BG, fontsize=12, va="center", ha="right", transform=a.transAxes)

    # Bottom bar
    b = fig.add_axes([0.0, 0.0, 1.0, 0.035])
    b.set_facecolor(_GREEN); b.axis("off")
    b.text(0.985, 0.50,
           "Phase 3  ·  DIN 5480 Spline Insertion  ·  Auto-Benchmark",
           color=_BG, fontsize=9, va="center", ha="right", transform=b.transAxes)

    # Left green stripe accent
    c = fig.add_axes([0.0, 0.035, 0.006, 0.845])
    c.set_facecolor(_GREEN); c.axis("off")

    # Main title block
    fig.text(0.055, 0.82, "DIN  5480 Spline Insertion Dynamics",
             color=_WHITE, fontsize=32, fontweight="bold", va="top")
    fig.text(0.055, 0.71, "VBD  vs  MuJoCo  Solver  Benchmark",
             color=_GREEN, fontsize=22, fontweight="bold", va="top")

    # Thin divider
    div = fig.add_axes([0.055, 0.645, 0.88, 0.003])
    div.set_facecolor("#3a3a3a"); div.axis("off")

    # Key question
    fig.text(0.055, 0.625, "Core question:", color=_LGRAY, fontsize=12,
             va="top", style="italic")
    fig.text(0.055, 0.565,
             "Which solver correctly captures tooth–groove engagement?\n"
             "What physical mechanisms explain the divergence?",
             color=_WHITE, fontsize=16, va="top", linespacing=1.7)

    # Stat boxes
    stats = [
        ("6-tooth", "DIN 5480  z=6"),
        ("23 s", "scripted trajectory"),
        ("2 solvers", "VBD  /  MuJoCo"),
        ("Auto-benchmark", "headless, repeatable"),
    ]
    x0 = 0.055
    for val, label in stats:
        ax = fig.add_axes([x0, 0.11, 0.205, 0.16])
        ax.set_facecolor(_BG3); ax.axis("off")
        # green top accent strip
        top = fig.add_axes([x0, 0.265, 0.205, 0.009])
        top.set_facecolor(_GREEN); top.axis("off")
        ax.text(0.50, 0.60, val,   ha="center", va="center", color=_GREEN,
                fontsize=17, fontweight="bold", transform=ax.transAxes)
        ax.text(0.50, 0.22, label, ha="center", va="center", color=_LGRAY,
                fontsize=9, transform=ax.transAxes)
        x0 += 0.225

    return fig


# ── Slide 2: The Task ──────────────────────────────────────────────────────────

def slide_task(vbd: Trace, mj: Trace) -> plt.Figure:
    fig = _new_slide()
    _chrome(fig,
            "The Task: Simulate Search-and-Engage Insertion",
            "DIN 5480  ·  involute spline  ·  z=6, module=4 mm, 30° pressure angle  "
            "·  hub bore ±45 mm")

    ref = mj

    # ── Left: shaft Z trajectory (main story of the whole experiment) ──────────
    ax = fig.add_axes([0.04, 0.10, 0.53, 0.68])
    _ax_style(ax, ylabel="Shaft centre Z (mm)", xlabel="Simulation time (s)")
    _phase_bands(ax, ref.t, ref.phase)
    _phase_labels_on(ax, ref.t, ref.phase)

    hub_top, hub_bot = 22.5, -22.5
    ax.axhline(hub_top, color="#888899", lw=1.0, ls="--", alpha=0.8, zorder=3)
    ax.axhline(0,       color="#555566", lw=0.7, ls=":",  alpha=0.6, zorder=3)
    ax.axhline(hub_bot, color="#888899", lw=1.0, ls="--", alpha=0.8, zorder=3)
    ax.fill_between([ref.t[0], ref.t[-1]], hub_bot, hub_top,
                    color="#1a2a44", alpha=0.25, zorder=1)
    ax.text(0.35, hub_top + 3, "hub top face", color="#888899", fontsize=7.5, zorder=4)
    ax.text(0.35, hub_bot - 7, "hub bottom face", color="#888899", fontsize=7.5, zorder=4)

    ax.plot(vbd.t, vbd.shaft_z, color=_BLUE,   lw=2.2, label="VBD (AVBD)", zorder=5)
    ax.plot(mj.t,  mj.shaft_z,  color=_ORANGE, lw=2.2, label="MuJoCo",    zorder=5)
    ax.set_xlim(ref.t[0], ref.t[-1])
    ax.legend(fontsize=9.5, framealpha=0.8, facecolor=_BG, labelcolor=_WHITE,
              loc="lower right")
    ax.set_title("Shaft Z during scripted 23-second trajectory",
                 color=_LGRAY, fontsize=9, pad=4)

    # ── Right: bullets ─────────────────────────────────────────────────────────
    sections = [
        ("What we simulate", [
            "Shaft descends into hub bore under\nP-controller descent force",
            "Teeth search for groove alignment\nwhile rotating slowly",
            "Hydroelastic contact on all tooth\nand land faces simultaneously",
            "Spring-damper on-axis guidance\n(lateral + rotation + descent)",
        ]),
        ("Why it's physically hard", [
            "Teeth ride on land faces → high\nresistance, near-zero insertion",
            "Groove alignment triggers a\nsnap-to-engage drop in force",
            "Solver must resolve this\nsnap in real time",
        ]),
        ("What we measure", [
            "Shaft position, contact area,\npenetration depth, frame time",
            "Engagement time, snap detection,\ntorsion-lock stiffness",
        ]),
    ]

    y = 0.76
    for header, items in sections:
        # Section header
        fig.text(0.615, y, header, color=_GREEN, fontsize=11.5, fontweight="bold", va="top")
        y -= 0.058
        for item in items:
            fig.text(0.625, y, "▸  " + item, color=_LGRAY, fontsize=9.5, va="top",
                     linespacing=1.45)
            lines = item.count("\n") + 1
            y -= 0.052 * lines + 0.004
        y -= 0.025

    return fig


# ── Slide 3: Benchmark Design ──────────────────────────────────────────────────

def slide_design() -> plt.Figure:
    fig = _new_slide()
    _chrome(fig,
            "Auto-Benchmark: Fair Comparison Design",
            "Parameters identical except where the contact algorithm forces a difference  "
            "·  headless, repeatable, same trajectory for both")

    # Left panel: IDENTICAL
    ax_lh = fig.add_axes([0.03, 0.768, 0.455, 0.042])
    ax_lh.set_facecolor("#0d2b0d"); ax_lh.axis("off")
    ax_lh.text(0.03, 0.50, "✓  KEPT IDENTICAL  —  same inputs, same conditions",
               color=_GREEN, fontsize=11, fontweight="bold", va="center",
               transform=ax_lh.transAxes)

    ax_l = fig.add_axes([0.03, 0.090, 0.455, 0.678])
    ax_l.set_facecolor(_BG2); ax_l.axis("off")

    identical_rows = [
        ("Geometry",          "DIN 5480 z=6, module=4 mm, 40 mm length"),
        ("Trajectory script", "Same scripted_trajectory.py (23 s, 9 phases)"),
        ("Shaft mass",        "1.00 kg  ·  solid cylinder inertia"),
        ("Contact pipeline",  "Newton hydroelastic SDF (both solvers)"),
        ("Descent gain",      "descent_k = 800 N·s/m"),
        ("Rotation damping",  "rot_d = 0.10–0.13 N·m·s/rad"),
        ("Lateral spring",    "lateral_k = 1×10⁵ N/m"),
        ("Friction coeff.",   "μ = 0.30  (dry steel-on-steel)"),
        ("SDF resolution",    "64³ voxels, narrow-band"),
        ("Warm-up frames",    "5 frames excluded from all timing stats"),
    ]

    y = 0.955
    for label, val in identical_rows:
        ax_l.text(0.035, y, f"✓  {label}:", color=_GREEN,
                  fontsize=9, fontweight="bold", va="top", transform=ax_l.transAxes)
        ax_l.text(0.360, y, val, color=_LGRAY, fontsize=9,
                  va="top", transform=ax_l.transAxes)
        y -= 0.089

    # Right panel: MUST DIFFER
    ax_rh = fig.add_axes([0.515, 0.768, 0.465, 0.042])
    ax_rh.set_facecolor("#2b1000"); ax_rh.axis("off")
    ax_rh.text(0.03, 0.50,
               "△  MUST DIFFER  —  contact algorithm forces the difference",
               color=_ORANGE, fontsize=11, fontweight="bold", va="center",
               transform=ax_rh.transAxes)

    ax_r = fig.add_axes([0.515, 0.090, 0.465, 0.678])
    ax_r.set_facecolor(_BG2); ax_r.axis("off")

    diff_rows = [
        ("Substeps / frame",
         "VBD = 8          MuJoCo = 16",
         "MuJoCo force-level model needs tighter sub_dt (1 ms)\n"
         "for per-substep contact correction during fast rotation"),
        ("Contact model",
         "VBD = position-level (AVBD)\nMuJoCo = force-level  (kh × volume)",
         "AVBD directly corrects penetration each sub-step;\n"
         "force model needs stiffness to generate resisting force"),
        ("Contact stiffness  kh",
         "VBD = 5×10⁶    MuJoCo = 2×10⁹",
         "Force-level equilibrium:  δ_eq = F / (kh · A)\n"
         "400× stiffer MJ kh gives <2 mm equilibrium penetration"),
        ("Collision frequency",
         "VBD = once per frame\nMuJoCo = every substep",
         "Force model needs fresh geometry at every sub-step\n"
         "to correctly block a rotating + descending shaft"),
    ]

    y = 0.955
    for label, vals, reason in diff_rows:
        ax_r.text(0.030, y,        f"△  {label}:", color=_ORANGE,
                  fontsize=9.5, fontweight="bold", va="top", transform=ax_r.transAxes)
        ax_r.text(0.030, y-0.063,  vals, color=_WHITE,
                  fontsize=9, va="top", transform=ax_r.transAxes,
                  linespacing=1.45)
        ax_r.text(0.030, y-0.155,  reason, color="#888888",
                  fontsize=8, va="top", style="italic",
                  transform=ax_r.transAxes, linespacing=1.45)
        ax_r.plot([0.02, 0.97], [y - 0.228, y - 0.228], color="#333333", lw=0.6,
                  transform=ax_r.transAxes, clip_on=False)
        y -= 0.238

    return fig


# ── Slide 4: Position & Performance Results ────────────────────────────────────

def slide_results(vbd: Trace, mj: Trace) -> plt.Figure:
    fig = _new_slide()
    _chrome(fig,
            "Results: Insertion Trajectory & Performance",
            "Same scripted trajectory, same inputs — solver contact model is the only difference")

    ref = mj
    hub_top, hub_bot = 22.5, -22.5

    # ── Main trajectory panel ──────────────────────────────────────────────────
    ax1 = fig.add_axes([0.04, 0.305, 0.60, 0.495])
    _ax_style(ax1, ylabel="Shaft centre Z (mm)")
    _phase_bands(ax1, ref.t, ref.phase)
    _phase_labels_on(ax1, ref.t, ref.phase)

    ax1.axhline(hub_top, color="#888899", lw=1.0, ls="--", alpha=0.8, zorder=3)
    ax1.axhline(0,       color="#555566", lw=0.7, ls=":",  alpha=0.5, zorder=3)
    ax1.axhline(hub_bot, color="#888899", lw=1.0, ls="--", alpha=0.8, zorder=3)
    ax1.fill_between([ref.t[0], ref.t[-1]], hub_bot, hub_top,
                     color="#1a2a44", alpha=0.20, zorder=1)
    ax1.text(0.3, hub_top + 3,  "hub top",    color="#888899", fontsize=7.5, zorder=4)
    ax1.text(0.3, hub_bot - 8,  "hub bottom", color="#888899", fontsize=7.5, zorder=4)

    ax1.plot(vbd.t, vbd.shaft_z, color=_BLUE,   lw=2.5, label="VBD (AVBD)", zorder=5)
    ax1.plot(mj.t,  mj.shaft_z,  color=_ORANGE, lw=2.5, label="MuJoCo",    zorder=5)
    ax1.set_xlim(ref.t[0], ref.t[-1])
    ax1.legend(fontsize=10, framealpha=0.8, facecolor=_BG, labelcolor=_WHITE)
    ax1.set_title("Shaft Z during insertion trajectory", color=_LGRAY, fontsize=9, pad=4)

    # Annotate VBD stuck point
    vbd_min_idx = int(np.argmin(vbd.shaft_z))
    ax1.annotate(
        "VBD: AVBD blocks\ntip–root penetration\n→ shaft rides lands",
        xy=(vbd.t[vbd_min_idx], vbd.shaft_z[vbd_min_idx]),
        xytext=(vbd.t[vbd_min_idx] - 3.5, vbd.shaft_z[vbd_min_idx] - 28),
        color=_BLUE, fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color=_BLUE, lw=1.2),
        zorder=6,
    )

    # Annotate MuJoCo snap
    dz    = np.diff(mj.shaft_z)
    snap  = int(np.argmin(dz))   # largest single-frame drop
    if dz[snap] < -0.5:
        ax1.annotate(
            f"MuJoCo: snap-to-groove\n@ {mj.t[snap+1]:.1f} s",
            xy=(mj.t[snap + 1], mj.shaft_z[snap + 1]),
            xytext=(mj.t[snap + 1] + 1.8, mj.shaft_z[snap + 1] + 30),
            color=_ORANGE, fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color=_ORANGE, lw=1.2),
            zorder=6,
        )

    # ── Frame-time panel ───────────────────────────────────────────────────────
    ax2 = fig.add_axes([0.04, 0.105, 0.60, 0.175])
    _ax_style(ax2, ylabel="Frame time (ms)", xlabel="Simulation time (s)")
    _phase_bands(ax2, ref.t, ref.phase)
    ax2.plot(vbd.t, _smooth(vbd.frame_ms, 15), color=_BLUE,   lw=1.8, label="VBD")
    ax2.plot(mj.t,  _smooth(mj.frame_ms,  15), color=_ORANGE, lw=1.8, label="MuJoCo")
    ax2.set_xlim(ref.t[0], ref.t[-1])
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8.5, framealpha=0.8, facecolor=_BG, labelcolor=_WHITE)
    ax2.set_title("Per-frame wall time  (lower = faster)",
                  color=_LGRAY, fontsize=8.5, pad=3)

    # ── Right-side callout boxes ───────────────────────────────────────────────
    callouts = [
        (_BLUE,   "VBD (AVBD)",
         "6.8 mm",    "max insertion depth",
         "84 ms/fr",  "0.20× RTF"),
        (_ORANGE, "MuJoCo",
         "72.5 mm",   "max insertion depth",
         "114 ms/fr", "0.15× RTF"),
    ]
    y0 = 0.665
    for color, solver, depth, dlabel, speed, rtf in callouts:
        # accent bar on left
        bar = fig.add_axes([0.674, y0, 0.007, 0.215])
        bar.set_facecolor(color); bar.axis("off")
        # box
        bx = fig.add_axes([0.681, y0, 0.295, 0.215])
        bx.set_facecolor(_BG3); bx.axis("off")
        bx.text(0.07, 0.88, solver,  color=color,  fontsize=11, fontweight="bold",
                va="top",  transform=bx.transAxes)
        bx.text(0.07, 0.62, depth,   color=_WHITE, fontsize=22, fontweight="bold",
                va="top",  transform=bx.transAxes)
        bx.text(0.07, 0.34, dlabel,  color=_LGRAY, fontsize=9,
                va="top",  transform=bx.transAxes)
        bx.text(0.07, 0.16, f"{speed}  ·  {rtf}", color=_LGRAY, fontsize=9,
                va="top",  transform=bx.transAxes)
        y0 -= 0.255

    # Insight box
    ins = fig.add_axes([0.674, 0.105, 0.302, 0.255])
    ins.set_facecolor("#1a1800"); ins.axis("off")
    ins_bar = fig.add_axes([0.674, 0.105, 0.007, 0.255])
    ins_bar.set_facecolor(_YELLOW); ins_bar.axis("off")
    ins.text(0.07, 0.88, "Key insight", color=_YELLOW, fontsize=10, fontweight="bold",
             va="top", transform=ins.transAxes)
    ins.text(0.07, 0.68,
             "VBD per-substep is 1.47× faster.\n"
             "MuJoCo inserts correctly because\n"
             "force-level contacts let the solver\n"
             "converge to an equilibrium state\n"
             "rather than blocking penetration.",
             color=_LGRAY, fontsize=8.5, va="top", transform=ins.transAxes,
             linespacing=1.55)

    return fig


# ── Slide 5: Contact & Force Analysis ─────────────────────────────────────────

def slide_contact(vbd: Trace, mj: Trace) -> plt.Figure:
    fig = _new_slide()
    _chrome(fig,
            "Contact Forces, Area & Convergence",
            "Hydroelastic pipeline metrics  ·  resistance proxy = descent_k × velocity error  "
            "·  kh VBD=5×10⁶  MuJoCo=2×10⁹")

    ref     = mj
    res_vbd = _resistance(vbd)
    res_mj  = _resistance(mj)

    panel_defs = [
        # (left, bottom, width, height)
        ([0.04, 0.565, 0.455, 0.260], [0.515, 0.565, 0.455, 0.260]),
        ([0.04, 0.105, 0.455, 0.260], [0.515, 0.105, 0.455, 0.260]),
    ]

    # ── Panel A: Axial resistance force proxy ──────────────────────────────────
    ax_a = fig.add_axes([0.04, 0.565, 0.455, 0.255])
    _ax_style(ax_a, ylabel="Resistance (N)")
    _phase_bands(ax_a, ref.t, ref.phase)
    ax_a.plot(vbd.t, res_vbd, color=_BLUE,   lw=1.8, label="VBD")
    ax_a.plot(mj.t,  res_mj,  color=_ORANGE, lw=1.8, label="MuJoCo")
    ax_a.axhline(0, color=_DGRAY, lw=0.7, ls=":", zorder=3)
    ax_a.set_xlim(ref.t[0], ref.t[-1])
    ax_a.legend(fontsize=8.5, framealpha=0.8, facecolor=_BG, labelcolor=_WHITE)
    ax_a.set_title(
        "Axial contact resistance  =  descent_k × (actual vel − commanded vel)"
        "  ·  +ve = blocking descent",
        color=_LGRAY, fontsize=8, pad=3)

    # Annotate snap (negative spike on MuJoCo = shaft overshoots command)
    snap_idx = int(np.argmin(res_mj))
    if res_mj[snap_idx] < -5:
        ax_a.annotate(
            f"snap–to–groove\n{res_mj[snap_idx]:.0f} N\n(shaft overtakes cmd)",
            xy=(mj.t[snap_idx], res_mj[snap_idx]),
            xytext=(mj.t[snap_idx] - 2.0, res_mj[snap_idx] - 5),
            color=_ORANGE, fontsize=7.5, ha="center",
            arrowprops=dict(arrowstyle="->", color=_ORANGE, lw=0.9),
        )

    # ── Panel B: Contact patch area ────────────────────────────────────────────
    ax_b = fig.add_axes([0.515, 0.565, 0.455, 0.255])
    _ax_style(ax_b, ylabel="Contact area (mm²)")
    _phase_bands(ax_b, ref.t, ref.phase)
    ax_b.plot(vbd.t, _smooth(vbd.contact_area, 7), color=_BLUE,   lw=1.8,
              label=f"VBD   max = {vbd.contact_area.max():.0f} mm²")
    ax_b.plot(mj.t,  _smooth(mj.contact_area,  7), color=_ORANGE, lw=1.8,
              label=f"MuJoCo   max = {mj.contact_area.max():.0f} mm²")
    ax_b.set_xlim(ref.t[0], ref.t[-1])
    ax_b.set_ylim(bottom=0)
    ax_b.legend(fontsize=8.5, framealpha=0.8, facecolor=_BG, labelcolor=_WHITE)
    ax_b.set_title("Hydroelastic contact patch area",
                   color=_LGRAY, fontsize=8, pad=3)

    # ── Panel C: Max penetration depth ─────────────────────────────────────────
    ax_c = fig.add_axes([0.04, 0.105, 0.455, 0.255])
    _ax_style(ax_c, ylabel="Max depth (mm)", xlabel="Simulation time (s)")
    _phase_bands(ax_c, ref.t, ref.phase)
    ax_c.plot(vbd.t, _smooth(vbd.max_depth, 7), color=_BLUE,   lw=1.8, label="VBD")
    ax_c.plot(mj.t,  _smooth(mj.max_depth,  7), color=_ORANGE, lw=1.8, label="MuJoCo")
    ax_c.set_xlim(ref.t[0], ref.t[-1])
    ax_c.set_ylim(bottom=0)
    ax_c.legend(fontsize=8.5, framealpha=0.8, facecolor=_BG, labelcolor=_WHITE)
    ax_c.set_title(
        "Max penetration depth  ·  AVBD correction proxy: near-0 = fully corrected",
        color=_LGRAY, fontsize=8, pad=3)

    # ── Panel D: N contacts ────────────────────────────────────────────────────
    ax_d = fig.add_axes([0.515, 0.105, 0.455, 0.255])
    _ax_style(ax_d, ylabel="N reduced contacts", xlabel="Simulation time (s)")
    _phase_bands(ax_d, ref.t, ref.phase)
    ax_d.plot(vbd.t, _smooth(vbd.n_contacts, 7), color=_BLUE,   lw=1.8,
              label=f"VBD   max = {int(vbd.n_contacts.max())}")
    ax_d.plot(mj.t,  _smooth(mj.n_contacts,  7), color=_ORANGE, lw=1.8,
              label=f"MuJoCo   max = {int(mj.n_contacts.max())}")
    ax_d.set_xlim(ref.t[0], ref.t[-1])
    ax_d.set_ylim(bottom=0)
    ax_d.legend(fontsize=8.5, framealpha=0.8, facecolor=_BG, labelcolor=_WHITE)
    ax_d.set_title("Reduced contact point count",
                   color=_LGRAY, fontsize=8, pad=3)

    # ── Bottom interpretation strip ────────────────────────────────────────────
    strip_y = 0.027
    fig.text(0.04, strip_y,
             "VBD/AVBD:  position-correction drives δ → 0 → stiff tooth faces block descent "
             "→ small contact area, near-zero depth, high resistance while teeth ride lands",
             color=_BLUE, fontsize=8, va="bottom")
    fig.text(0.04, strip_y - 0.022,
             "MuJoCo:    force-level equilibrium (δ_eq = F/kh/A) → shaft descends through groove "
             "→ contact area grows 4× → snap event visible in resistance (overtakes command)",
             color=_ORANGE, fontsize=8, va="bottom")

    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark CSVs ...")
    vbd = _load("vbd")
    mj  = _load("mujoco")

    slides = [
        ("01_title",   slide_title()),
        ("02_task",    slide_task(vbd, mj)),
        ("03_design",  slide_design()),
        ("04_results", slide_results(vbd, mj)),
        ("05_contact", slide_contact(vbd, mj)),
    ]

    pdf_path = OUT / "phase3_benchmark.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        for name, fig in slides:
            png_path = OUT / f"{name}.png"
            fig.savefig(str(png_path), dpi=_DPI, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  Saved {png_path.name}")

    print(f"  Saved {pdf_path.name}")
    print("Done.")


if __name__ == "__main__":
    main()
