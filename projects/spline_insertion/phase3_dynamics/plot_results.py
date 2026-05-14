# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Plot Phase 3 solver benchmark results.

Reads bench_results/bench_vbd.csv and bench_results/bench_mujoco.csv and
produces a four-panel comparison figure saved to bench_results/comparison_plot.png.

Usage
-----
    python plot_results.py
    python plot_results.py --out bench_results/my_plot.png
    python plot_results.py --dir ./other_results
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = pathlib.Path(__file__).parent

# ── Phase colours — muted palette legible on both dark and white backgrounds
_PHASE_COLORS: dict[str, tuple] = {
    "vertical_demo": (0.60, 0.60, 0.60),
    "approach":      (0.26, 0.56, 0.87),
    "sit_on_hub":    (0.98, 0.82, 0.08),
    "search_groove": (0.98, 0.50, 0.05),
    "engage":        (0.46, 0.76, 0.00),   # NVIDIA green family
    "push_in":       (0.25, 0.62, 0.00),
    "friction_demo": (0.90, 0.30, 0.10),
    "torsion_lock":  (0.85, 0.12, 0.12),
    "pull_out":      (0.20, 0.70, 0.90),
}

# Abbreviated labels for the narrow phase strip
_PHASE_ABBREV: dict[str, str] = {
    # current names
    "vertical_demo": "vert.",
    "approach":      "approach",
    "sit_on_hub":    "sit",
    "search":        "search",
    "hold":          "hold",
    "push":          "push",
    "retract":       "retract",
    "withdraw":      "withdraw",
    # legacy names (old CSV data)
    "search_groove": "search",
    "engage":        "hold",
    "push_in":       "push",
    "friction_demo": "push",
    "torsion_lock":  "retract",
    "pull_out":      "withdraw",
}

# NVIDIA-inspired solver palette: NVIDIA green vs deep sky blue
_SOLVER_COLORS_DARK  = {"vbd": "#76B900", "mujoco": "#00A3E0"}
_SOLVER_COLORS_LIGHT = {"vbd": "#5A8C00", "mujoco": "#0078A8"}
_SOLVER_LABELS       = {"vbd": "VBD (AVBD)", "mujoco": "MuJoCo"}


# ── Plot style ────────────────────────────────────────────────────────────────

@dataclass
class PlotStyle:
    fig_bg:        str
    ax_bg:         str
    text:          str
    spine:         str
    grid:          str
    band_alpha:    float
    vline_alpha:   float
    legend_face:   str
    legend_text:   str
    title_color:   str
    annot_color:   str
    lw:            float   # default line width
    solver_colors: dict
    suffix:        str


DARK_STYLE = PlotStyle(
    fig_bg="#111111",  ax_bg="#1a1a1a",  text="#e0e0e0",  spine="#333333",
    grid="#2a2a2a",    band_alpha=0.12,   vline_alpha=0.0,
    legend_face="#1a1a1a", legend_text="#e0e0e0",
    title_color="#ffffff", annot_color="#888888",
    lw=1.4, solver_colors=_SOLVER_COLORS_DARK,
    suffix="",
)

LIGHT_STYLE = PlotStyle(
    fig_bg="#ffffff",  ax_bg="#ffffff",  text="#1a1a1a",  spine="#dddddd",
    grid="#eeeeee",    band_alpha=0.13,   vline_alpha=0.35,
    legend_face="#ffffff", legend_text="#1a1a1a",
    title_color="#000000", annot_color="#888888",
    lw=1.1, solver_colors=_SOLVER_COLORS_LIGHT,
    suffix="_light",
)


def _setup_axes(fig, axes, style: PlotStyle) -> None:
    fig.patch.set_facecolor(style.fig_bg)
    for ax in axes:
        ax.set_facecolor(style.ax_bg)
        # NVIDIA-clean: only left + bottom spines, others hidden
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(style.spine)
            ax.spines[side].set_linewidth(0.7)
        ax.tick_params(colors=style.text, labelsize=9, length=3, width=0.7)
        ax.yaxis.label.set_color(style.text)
        ax.xaxis.label.set_color(style.text)
        ax.title.set_color(style.title_color)


# ── Data loading ──────────────────────────────────────────────────────────────

@dataclass
class TraceData:
    solver:           str
    t:                np.ndarray
    shaft_z:          np.ndarray   # mm, shaft centre
    bot_z:            np.ndarray   # mm, shaft bottom (= shaft_z - shaft_half_z)
    angle:            np.ndarray   # degrees mod 60
    frame_ms:         np.ndarray
    phase:            list[str]
    cmd_des_mmps:     np.ndarray   # commanded descent speed (mm/s)
    cmd_rot_dps:      np.ndarray   # commanded rotation speed (deg/s)
    contact_area_mm2: np.ndarray   # hydroelastic patch area (mm²); zeros if old CSV
    n_contacts:       np.ndarray   # reduced contact count; zeros if old CSV
    max_depth_mm:     np.ndarray   # max penetration depth (mm); zeros if old CSV
    # Net body force/torque vectors from Δ(mv)/dt (contact + applied)
    net_fx:           np.ndarray   # N, lateral
    net_fy:           np.ndarray   # N, lateral
    net_fz:           np.ndarray   # N, axial (+ = upward / opposing insertion)
    net_tx:           np.ndarray   # N·m
    net_ty:           np.ndarray   # N·m
    net_tz:           np.ndarray   # N·m, axial torque (+ = CCW)
    applied_fz:       np.ndarray   # N, commanded axial force proxy
    applied_tz:       np.ndarray   # N·m, commanded torque proxy


def load_csv(path: pathlib.Path, shaft_half_z_mm: float = 50.0) -> TraceData:
    solver = path.stem.replace("bench_", "")
    rows = []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"ERROR: {path} is empty")
    t        = np.array([float(r["t"])               for r in rows])
    shaft_z  = np.array([float(r["shaft_z_mm"])      for r in rows])
    angle    = np.array([float(r["shaft_angle_mod60"]) for r in rows])
    frame_ms = np.array([float(r["frame_ms"])         for r in rows])
    phase    = [r["phase"] for r in rows]
    bot_z    = shaft_z - shaft_half_z_mm

    cmd_des  = np.array([float(r.get("cmd_des_mmps", 0.0)) for r in rows])
    cmd_rot  = np.array([float(r.get("cmd_rot_dps",  0.0)) for r in rows])
    c_area   = np.array([float(r.get("contact_area_mm2", 0.0)) for r in rows])
    n_cont   = np.array([float(r.get("n_contacts",    0.0)) for r in rows])
    max_dep  = np.array([float(r.get("max_depth_mm",  0.0)) for r in rows])
    net_fx   = np.array([float(r.get("net_fx_N",     0.0)) for r in rows])
    net_fy   = np.array([float(r.get("net_fy_N",     0.0)) for r in rows])
    net_fz   = np.array([float(r.get("net_fz_N",     0.0)) for r in rows])
    net_tx   = np.array([float(r.get("net_tx_Nm",    0.0)) for r in rows])
    net_ty   = np.array([float(r.get("net_ty_Nm",    0.0)) for r in rows])
    net_tz   = np.array([float(r.get("net_tz_Nm",    0.0)) for r in rows])
    app_fz   = np.array([float(r.get("applied_fz_N",  0.0)) for r in rows])
    app_tz   = np.array([float(r.get("applied_tz_Nm", 0.0)) for r in rows])

    return TraceData(
        solver, t, shaft_z, bot_z, angle, frame_ms, phase,
        cmd_des, cmd_rot, c_area, n_cont, max_dep,
        net_fx, net_fy, net_fz, net_tx, net_ty, net_tz, app_fz, app_tz,
    )


# ── Phase band helper ─────────────────────────────────────────────────────────

def _collect_phase_spans(t: np.ndarray, phases: list[str]):
    """Return list of (t_start, t_end, phase_name) spans."""
    spans = []
    prev_ph = None
    t0 = t[0]
    for i, ph in enumerate(phases):
        if ph != prev_ph:
            if prev_ph is not None:
                spans.append((t0, t[i], prev_ph))
            prev_ph = ph
            t0 = t[i]
    if prev_ph is not None:
        spans.append((t0, t[-1], prev_ph))
    return spans


def _add_phase_bands(
    ax, t: np.ndarray, phases: list[str], style: PlotStyle,
) -> None:
    """Alternating gray vertical bands + subtle divider lines."""
    is_light = style.fig_bg == "#ffffff"
    band_even = "#f0f0f0" if is_light else "#2a2a2a"
    band_odd  = style.ax_bg
    for idx, (t0, t1, _ph) in enumerate(_collect_phase_spans(t, phases)):
        color = band_even if idx % 2 == 0 else band_odd
        ax.axvspan(t0, t1, color=color, linewidth=0, zorder=0)
    transitions = [t1 for _, t1, _ in _collect_phase_spans(t, phases)[:-1]]
    for tx in transitions:
        ax.axvline(tx, color=style.spine, lw=0.5, alpha=0.35, zorder=1)


def _label_phases_centered(
    ax, t: np.ndarray, phases: list[str], style: PlotStyle,
) -> None:
    """Horizontal phase labels — 7 pt normal weight, truncated to fit each band."""
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    spans = _collect_phase_spans(t, phases)
    if not spans:
        return

    fig       = ax.get_figure()
    ax_w_in   = ax.get_position().width * fig.get_size_inches()[0]
    t_total   = max(t[-1] - t[0], 1e-9)
    CHAR_W_IN = 0.052   # empirical: 7 pt proportional font ≈ 0.052 in/char

    for _t0, _t1, ph in spans:
        t_mid       = (_t0 + _t1) / 2.0
        box_w_in    = ax_w_in * (_t1 - _t0) / t_total
        char_budget = max(1, int(box_w_in / CHAR_W_IN) - 1)
        abbrev      = _PHASE_ABBREV.get(ph, ph.replace("_", " "))
        label       = abbrev if len(abbrev) <= char_budget else abbrev[:char_budget]
        ax.text(
            t_mid, 0.5, label,
            va="center", ha="center",
            fontsize=7, color=style.text,
            rotation=0,
            transform=ax.get_xaxis_transform(),
            clip_on=True,
        )


def _phase_legend_patches() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=c, alpha=0.5, label=n.replace("_", " "))
        for n, c in _PHASE_COLORS.items()
    ]


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(
    traces: list[TraceData],
    hub_half_z_mm: float = 22.5,
    out: pathlib.Path | None = None,
    style: PlotStyle = DARK_STYLE,
) -> pathlib.Path:
    # 3 content panels + compact phase strip — landscape for slide use
    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 2.0, 2.0, 0.6], "hspace": 0.08},
    )
    _setup_axes(fig, axes, style)

    ref = max(traces, key=lambda tr: len(tr.t))
    ref_col    = "#888888" if style.fig_bg == "#ffffff" else "#888899"
    groove_col = "#2a7a2a" if style.fig_bg == "#ffffff" else "#446644"
    land_col   = "#aa3333" if style.fig_bg == "#ffffff" else "#664444"
    lw = style.lw

    # ── Panel 0: Shaft Z trajectory ───────────────────────────────────────────
    ax0 = axes[0]
    _add_phase_bands(ax0, ref.t, ref.phase, style)
    hub_top = hub_half_z_mm
    hub_bot = -hub_half_z_mm
    ax0.axhline(hub_top, color=ref_col, lw=0.8, ls="--", label=f"hub top (+{hub_top:.0f} mm)")
    ax0.axhline(0,       color=ref_col, lw=0.6, ls=":",  label="hub centre (0 mm)")
    ax0.axhline(hub_bot, color=ref_col, lw=0.8, ls="--", label=f"hub bottom ({hub_bot:.0f} mm)")
    for tr in traces:
        c = style.solver_colors[tr.solver]
        ax0.plot(tr.t, tr.shaft_z, color=c, lw=lw,       label=f"{_SOLVER_LABELS[tr.solver]}")
        ax0.plot(tr.t, tr.bot_z,   color=c, lw=lw*0.6,   ls="--", alpha=0.55)
    ax0.fill_between([ref.t[0], ref.t[-1]], hub_bot, hub_top,
                     color="#dde8ff" if style.fig_bg == "#ffffff" else "#3344aa",
                     alpha=0.08, zorder=0)
    ax0.set_ylabel("Shaft Z (mm)", fontsize=10)
    ax0.set_title("VBD vs MuJoCo: Spline Insertion Trajectory", fontsize=12, pad=6)
    ax0.legend(loc="upper right", fontsize=8.5, framealpha=0.7,
               facecolor=style.legend_face, labelcolor=style.legend_text, ncol=3)
    ax0.grid(axis="y", color=style.grid, lw=0.5)

    # ── Panel 1: Frame timing ─────────────────────────────────────────────────
    ax1 = axes[1]
    _add_phase_bands(ax1, ref.t, ref.phase, style)
    for tr in traces:
        c = style.solver_colors[tr.solver]
        ms_s = np.convolve(tr.frame_ms, np.ones(15)/15, mode="same")
        ax1.plot(tr.t, tr.frame_ms, color=c, lw=0.4, alpha=0.20)
        ax1.plot(tr.t, ms_s,        color=c, lw=lw,  label=_SOLVER_LABELS[tr.solver])
    ax1.set_ylabel("Frame time (ms)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.7,
               facecolor=style.legend_face, labelcolor=style.legend_text)
    ax1.grid(axis="y", color=style.grid, lw=0.5)
    ax1.set_ylim(bottom=0)
    perf_lines = []
    for tr in traces:
        mean_ms = np.mean(tr.frame_ms)
        total_s = np.sum(tr.frame_ms) / 1000.0
        rtf     = tr.t[-1] / total_s if total_s > 0 else 0
        perf_lines.append(f"{_SOLVER_LABELS[tr.solver]}: {mean_ms:.0f} ms/frame · RTF {rtf:.2f}×")
    ax1.text(0.01, 0.97, "  ".join(perf_lines),
             transform=ax1.transAxes, fontsize=7.5, color=style.annot_color,
             va="top", family="monospace")

    # ── Panel 2: Shaft angle mod 60 ───────────────────────────────────────────
    ax2 = axes[2]
    _add_phase_bands(ax2, ref.t, ref.phase, style)
    for tr in traces:
        c = style.solver_colors[tr.solver]
        ax2.plot(tr.t, tr.angle, color=c, lw=lw, label=_SOLVER_LABELS[tr.solver])
    ax2.axhline(0.0,  color=groove_col, lw=0.9, ls="--", alpha=0.7)
    ax2.axhline(30.0, color=land_col,   lw=0.9, ls=":",  alpha=0.7)
    ax2.axhline(60.0, color=groove_col, lw=0.9, ls="--", alpha=0.7)
    ax2.set_yticks([0, 15, 30, 45, 60])
    ax2.set_yticklabels(["0° groove", "15°", "30° land", "45°", "60° groove"],
                        fontsize=8, color=style.text)
    ax2.set_ylabel("Angle mod 60° (°)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.7,
               facecolor=style.legend_face, labelcolor=style.legend_text)
    ax2.grid(axis="y", color=style.grid, lw=0.5)
    ax2.set_xlabel("Simulation time (s)", fontsize=10)

    # ── Panel 3: Phase strip ──────────────────────────────────────────────────
    ax3 = axes[3]
    _add_phase_bands(ax3, ref.t, ref.phase, style)
    _label_phases_centered(ax3, ref.t, ref.phase, style)
    ax3.set_ylabel("Phase", fontsize=8, color=style.text)

    for ax in axes:
        ax.set_xlim(ref.t[0], ref.t[-1])

    if out is None:
        out = HERE / "bench_results" / f"comparison_plot{style.suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {out}")
    return out


# ── Figure 2: Forces, contact dynamics, convergence ───────────────────────────

def _smooth(arr: np.ndarray, win: int = 9) -> np.ndarray:
    if len(arr) < win:
        return arr
    return np.convolve(arr, np.ones(win) / win, mode="same")


def _unwrap_vel(angle_mod60: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Differentiate angle_mod60 with wrap-around handling -> deg/s."""
    dt   = np.diff(t, prepend=t[0])
    dt   = np.where(dt < 1e-9, 1e-9, dt)
    da   = np.diff(angle_mod60, prepend=angle_mod60[0])
    # Correct 60-degree wrap-arounds
    da   = np.where(da >  30.0, da - 60.0, da)
    da   = np.where(da < -30.0, da + 60.0, da)
    return da / dt


def make_figure_contacts(
    traces:        list[TraceData],
    descent_k:     float = 800.0,
    hub_half_z_mm: float = 22.5,
    out:           pathlib.Path | None = None,
    style:         PlotStyle = DARK_STYLE,
) -> pathlib.Path:
    """Contact dynamics: axial velocity, contact area, N contacts, penetration depth."""
    fig, axes = plt.subplots(
        5, 1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 2.0, 2.0, 2.0, 0.6], "hspace": 0.08},
    )
    _setup_axes(fig, axes, style)
    ref = max(traces, key=lambda tr: len(tr.t))
    lw = style.lw

    # ── Panel 0: Axial velocity actual vs commanded ───────────────────────────
    ax0 = axes[0]
    _add_phase_bands(ax0, ref.t, ref.phase, style)
    for tr in traces:
        c   = style.solver_colors[tr.solver]
        lbl = _SOLVER_LABELS[tr.solver]
        dt      = np.diff(tr.t, prepend=tr.t[0])
        dt      = np.where(dt < 1e-9, 1e-9, dt)
        vel_act = _smooth(np.diff(tr.shaft_z, prepend=tr.shaft_z[0]) / dt, 7)
        vel_cmd = -tr.cmd_des_mmps
        ax0.plot(tr.t, vel_act, color=c, lw=lw,      label=f"{lbl} actual")
        ax0.plot(tr.t, vel_cmd, color=c, lw=lw*0.6,  ls="--", alpha=0.55,
                 label=f"{lbl} cmd")
    ax0.axhline(0, color=style.spine, lw=0.6, ls=":")
    ax0.set_ylabel("Axial velocity\n(mm/s)", fontsize=9)
    ax0.set_title("VBD vs MuJoCo: Contact Dynamics & Convergence",
                  fontsize=12, pad=6)
    ax0.legend(loc="upper right", fontsize=8, framealpha=0.7,
               facecolor=style.legend_face, labelcolor=style.legend_text, ncol=2)
    ax0.grid(axis="y", color=style.grid, lw=0.5)

    # ── Panel 1: Contact area ─────────────────────────────────────────────────
    ax1 = axes[1]
    _add_phase_bands(ax1, ref.t, ref.phase, style)
    for tr in traces:
        c   = style.solver_colors[tr.solver]
        ax1.plot(tr.t, tr.contact_area_mm2, color=c, lw=0.4, alpha=0.25)
        ax1.plot(tr.t, _smooth(tr.contact_area_mm2, 5), color=c, lw=lw,
                 label=_SOLVER_LABELS[tr.solver])
    ax1.set_ylabel("Contact area (mm²)", fontsize=9)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.7,
               facecolor=style.legend_face, labelcolor=style.legend_text)
    ax1.set_ylim(bottom=0)
    ax1.grid(axis="y", color=style.grid, lw=0.5)

    # ── Panel 2: N contacts + max penetration depth (dual axis) ──────────────
    ax2  = axes[2]
    ax2r = ax2.twinx()
    _add_phase_bands(ax2, ref.t, ref.phase, style)
    for tr in traces:
        c = style.solver_colors[tr.solver]
        ax2.plot( tr.t, _smooth(tr.n_contacts,  5), color=c, lw=lw,
                  label=_SOLVER_LABELS[tr.solver])
        ax2r.plot(tr.t, _smooth(tr.max_depth_mm, 5), color=c, lw=lw*0.7,
                  ls=":", alpha=0.7)
    ax2.set_ylabel("N contacts (reduced)", fontsize=9, color=style.text)
    ax2r.set_ylabel("Max depth (mm)", fontsize=9,
                    color="#777777" if style.fig_bg == "#ffffff" else "#aaaaaa")
    ax2r.tick_params(colors="#777777" if style.fig_bg == "#ffffff" else "#aaaaaa",
                     labelsize=8)
    ax2.set_ylim(bottom=0); ax2r.set_ylim(bottom=0)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.7,
               facecolor=style.legend_face, labelcolor=style.legend_text)
    ax2.grid(axis="y", color=style.grid, lw=0.5)
    ax2.text(0.01, 0.97, "solid = N contacts  |  dotted = max penetration depth (mm)",
             transform=ax2.transAxes, fontsize=7, color=style.annot_color,
             va="top", style="italic")

    # ── Panel 3: Angular velocity actual vs commanded ─────────────────────────
    ax3 = axes[3]
    _add_phase_bands(ax3, ref.t, ref.phase, style)
    for tr in traces:
        c   = style.solver_colors[tr.solver]
        lbl = _SOLVER_LABELS[tr.solver]
        omega_act = _smooth(_unwrap_vel(tr.angle, tr.t), 9)
        ax3.plot(tr.t, omega_act,      color=c, lw=lw,      label=f"{lbl} actual")
        ax3.plot(tr.t, tr.cmd_rot_dps, color=c, lw=lw*0.6,  ls="--", alpha=0.55)
    ax3.axhline(0, color=style.spine, lw=0.6, ls=":")
    ax3.set_ylabel("Angular vel (°/s)", fontsize=9)
    ax3.set_xlabel("Simulation time (s)", fontsize=10)
    ax3.legend(loc="upper right", fontsize=9, framealpha=0.7,
               facecolor=style.legend_face, labelcolor=style.legend_text)
    ax3.grid(axis="y", color=style.grid, lw=0.5)

    # ── Panel 4: Phase strip ──────────────────────────────────────────────────
    ax4 = axes[4]
    _add_phase_bands(ax4, ref.t, ref.phase, style)
    _label_phases_centered(ax4, ref.t, ref.phase, style)
    ax4.set_ylabel("Phase", fontsize=8, color=style.text)

    for ax in axes:
        ax.set_xlim(ref.t[0], ref.t[-1])

    if out is None:
        out = HERE / "bench_results" / f"comparison_contacts{style.suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {out}")
    return out


# ── Figure: Slide-deck compact (3 panels + phase strip) ──────────────────────

def make_figure_slide(
    traces:        list[TraceData],
    hub_half_z_mm: float = 22.5,
    out:           pathlib.Path | None = None,
    style:         PlotStyle = DARK_STYLE,
) -> pathlib.Path:
    """Compact 16:9 slide figure: axial velocity | contacts + area | angular velocity."""
    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 2.0, 2.0, 0.6], "hspace": 0.08},
    )
    _setup_axes(fig, axes, style)
    ref = max(traces, key=lambda tr: len(tr.t))
    lw  = style.lw

    # ── Panel 0: Axial velocity actual vs commanded ───────────────────────────
    ax0 = axes[0]
    _add_phase_bands(ax0, ref.t, ref.phase, style)
    for tr in traces:
        c   = style.solver_colors[tr.solver]
        lbl = _SOLVER_LABELS[tr.solver]
        dt      = np.diff(tr.t, prepend=tr.t[0])
        dt      = np.where(dt < 1e-9, 1e-9, dt)
        vel_act = _smooth(np.diff(tr.shaft_z, prepend=tr.shaft_z[0]) / dt, 7)
        ax0.plot(tr.t, vel_act,      color=c, lw=lw,      label=f"{lbl} actual")
        ax0.plot(tr.t, -tr.cmd_des_mmps, color=c, lw=lw*0.55, ls="--", alpha=0.5,
                 label=f"{lbl} cmd")
    ax0.axhline(0, color=style.spine, lw=0.5, ls=":")
    ax0.set_ylabel("Axial vel\n(mm/s)", fontsize=9, color=style.text)
    ax0.set_title("VBD vs MuJoCo: Spline Insertion Dynamics",
                  fontsize=12, pad=6)
    ax0.legend(loc="upper right", fontsize=8.5, framealpha=0.7, ncol=2,
               facecolor=style.legend_face, labelcolor=style.legend_text)
    ax0.grid(axis="y", color=style.grid, lw=0.5)

    # ── Panel 1: Contact area (left) + N contacts (right, dual-axis) ──────────
    ax1  = axes[1]
    ax1r = ax1.twinx()
    _add_phase_bands(ax1, ref.t, ref.phase, style)
    for tr in traces:
        c = style.solver_colors[tr.solver]
        ax1.plot( tr.t, _smooth(tr.contact_area_mm2, 5), color=c, lw=lw,
                  label=_SOLVER_LABELS[tr.solver])
        ax1r.plot(tr.t, _smooth(tr.n_contacts, 5), color=c, lw=lw * 0.7,
                  ls=":", alpha=0.75)
    ax1.set_ylabel("Contact area\n(mm²)", fontsize=9, color=style.text)
    ax1.set_ylim(bottom=0)
    dim_col = "#777777" if style.fig_bg == "#ffffff" else "#aaaaaa"
    ax1r.set_ylabel("N contacts", fontsize=9, color=dim_col)
    ax1r.tick_params(colors=dim_col, labelsize=8)
    ax1r.set_ylim(bottom=0)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.7,
               facecolor=style.legend_face, labelcolor=style.legend_text)
    ax1.text(0.005, 0.96, "solid = contact area  |  dotted = N contacts",
             transform=ax1.transAxes, fontsize=7, color=style.annot_color,
             va="top", style="italic")
    ax1.grid(axis="y", color=style.grid, lw=0.5)

    # ── Panel 2: Angular velocity actual vs commanded ─────────────────────────
    ax2 = axes[2]
    _add_phase_bands(ax2, ref.t, ref.phase, style)
    for tr in traces:
        c   = style.solver_colors[tr.solver]
        lbl = _SOLVER_LABELS[tr.solver]
        omega_act = _smooth(_unwrap_vel(tr.angle, tr.t), 9)
        ax2.plot(tr.t, omega_act,      color=c, lw=lw,      label=f"{lbl} actual")
        ax2.plot(tr.t, tr.cmd_rot_dps, color=c, lw=lw*0.55, ls="--", alpha=0.5)
    ax2.axhline(0, color=style.spine, lw=0.5, ls=":")
    ax2.set_ylabel("Angular vel\n(°/s)", fontsize=9, color=style.text)
    ax2.set_xlabel("Simulation time (s)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.7,
               facecolor=style.legend_face, labelcolor=style.legend_text)
    ax2.grid(axis="y", color=style.grid, lw=0.5)

    # ── Panel 3: Phase strip ──────────────────────────────────────────────────
    ax3 = axes[3]
    _add_phase_bands(ax3, ref.t, ref.phase, style)
    _label_phases_centered(ax3, ref.t, ref.phase, style)
    ax3.set_ylabel("Phase", fontsize=8, color=style.text)

    for ax in axes:
        ax.set_xlim(ref.t[0], ref.t[-1])

    if out is None:
        out = HERE / "bench_results" / f"comparison_slide{style.suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {out}")
    return out


# ── Figure 3: Force & torque vectors on shaft teeth ──────────────────────────


def make_figure_forces(
    traces: list[TraceData],
    out:    pathlib.Path | None = None,
    style:  PlotStyle = DARK_STYLE,
) -> pathlib.Path:
    """Force & torque vectors on shaft teeth: axial Fz, torque Tz, lateral |Fxy|."""
    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 2.0, 2.0, 0.6], "hspace": 0.08},
    )
    _setup_axes(fig, axes, style)
    ref = max(traces, key=lambda tr: len(tr.t))
    lw = style.lw

    # -- Panel 0: Axial contact force
    ax0 = axes[0]
    _add_phase_bands(ax0, ref.t, ref.phase, style)
    for tr in traces:
        c   = style.solver_colors[tr.solver]
        lbl = _SOLVER_LABELS[tr.solver]
        contact_fz = _smooth(tr.net_fz + tr.applied_fz, 5)
        ax0.plot(tr.t, contact_fz,            color=c, lw=lw,      label=lbl)
        ax0.plot(tr.t, _smooth(tr.net_fz, 5), color=c, lw=lw*0.5, ls="--", alpha=0.40)
    ax0.axhline(0, color=style.spine, lw=0.6, ls=":")
    ax0.set_ylabel("Axial contact force (N)\n+ = opposing insertion", fontsize=9)
    ax0.set_title(
        "VBD vs MuJoCo: Force & Torque on Shaft Teeth  [delta(mv)/dt]",
        fontsize=12, pad=6,
    )
    ax0.legend(loc="upper right", fontsize=9, framealpha=0.7,
               facecolor=style.legend_face, labelcolor=style.legend_text)
    ax0.grid(axis="y", color=style.grid, lw=0.5)
    ax0.text(0.01, 0.97,
             "solid = contact Fz = net Fz + applied Fz   |   dashed = raw net",
             transform=ax0.transAxes, fontsize=7, color=style.annot_color, va="top", style="italic")

    # -- Panel 1: Axial torque + lateral (dual axis)
    ax1  = axes[1]
    ax1r = ax1.twinx()
    _add_phase_bands(ax1, ref.t, ref.phase, style)
    for tr in traces:
        c   = style.solver_colors[tr.solver]
        lbl = _SOLVER_LABELS[tr.solver]
        contact_tz = _smooth(tr.net_tz - tr.applied_tz, 5)
        f_lat      = _smooth(np.sqrt(tr.net_fx**2 + tr.net_fy**2), 5)
        ax1.plot( tr.t, contact_tz * 1000.0, color=c, lw=lw,     label=f"{lbl} Tz")
        ax1r.plot(tr.t, f_lat,               color=c, lw=lw*0.7, ls=":", alpha=0.75)
    ax1.axhline(0, color=style.spine, lw=0.6, ls=":"  )
    ax1.set_ylabel("Contact torque Tz (mN*m)", fontsize=9, color=style.text)
    right_col = "#777777" if style.fig_bg == "#ffffff" else "#aaaaaa"
    ax1r.set_ylabel("|F lateral| (N)", fontsize=9, color=right_col)
    ax1r.tick_params(colors=right_col, labelsize=8)
    ax1r.set_ylim(bottom=0)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.7,
               facecolor=style.legend_face, labelcolor=style.legend_text)
    ax1.grid(axis="y", color=style.grid, lw=0.5)
    ax1.text(0.01, 0.97,
             "solid = contact Tz (mN*m, neg = groove resists rotation)   |   dotted = |F lateral| (N)",
             transform=ax1.transAxes, fontsize=7, color=style.annot_color, va="top", style="italic")

    # -- Panel 2: Contact count + area
    ax2  = axes[2]
    ax2r = ax2.twinx()
    _add_phase_bands(ax2, ref.t, ref.phase, style)
    for tr in traces:
        c   = style.solver_colors[tr.solver]
        lbl = _SOLVER_LABELS[tr.solver]
        ax2.plot( tr.t, _smooth(tr.n_contacts,       5), color=c, lw=lw,     label=lbl)
        ax2r.plot(tr.t, _smooth(tr.contact_area_mm2, 5), color=c, lw=lw*0.7, ls="--", alpha=0.7)
    ax2.set_ylabel("N contacts (reduced)", fontsize=9, color=style.text)
    ax2r.set_ylabel("Contact area (mm^2)", fontsize=9, color=right_col)
    ax2r.tick_params(colors=right_col, labelsize=8)
    ax2.set_ylim(bottom=0); ax2r.set_ylim(bottom=0)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.7,
               facecolor=style.legend_face, labelcolor=style.legend_text)
    ax2.grid(axis="y", color=style.grid, lw=0.5)
    ax2.set_xlabel("Simulation time (s)", fontsize=10)
    ax2.text(0.01, 0.97, "solid = N contacts   |   dashed = contact area (mm^2)",
             transform=ax2.transAxes, fontsize=7, color=style.annot_color, va="top", style="italic")

    # -- Panel 3: Phase strip
    ax3 = axes[3]
    _add_phase_bands(ax3, ref.t, ref.phase, style)
    _label_phases_centered(ax3, ref.t, ref.phase, style)
    ax3.set_ylabel("Phase", fontsize=8, color=style.text)

    for ax in axes:
        ax.set_xlim(ref.t[0], ref.t[-1])

    fig.text(0.01, 0.005,
             "Force decomp:  net = delta(mv)/dt   |   contact Fz = net Fz + applied Fz",
             fontsize=7.5, color=style.annot_color,
             verticalalignment="bottom", family="monospace")

    if out is None:
        out = HERE / "bench_results" / f"comparison_forces{style.suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {out}")
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=None, metavar="DIR",
                        help="Directory containing bench_vbd.csv and bench_mujoco.csv "
                             "(default: bench_results/ next to this script)")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Output PNG path (default: <dir>/comparison_plot.png)")
    parser.add_argument("--shaft-half-z", type=float, default=50.0, metavar="MM",
                        help="Shaft half-length in mm for bottom-Z calculation (default: 50)")
    parser.add_argument("--hub-half-z", type=float, default=22.5, metavar="MM",
                        help="Hub half-height in mm for reference lines (default: 22.5)")
    parser.add_argument("--light", action="store_true", default=False,
                        help="White background theme for slide decks (default: dark)")
    parser.add_argument("--slide", action="store_true", default=False,
                        help="Also produce compact 16:9 slide figure")
    args = parser.parse_args()

    bench_dir = pathlib.Path(args.dir) if args.dir else HERE / "bench_results"
    style     = LIGHT_STYLE if args.light else DARK_STYLE
    if args.out:
        p = pathlib.Path(args.out)
        out_path = p.parent / (p.stem + style.suffix + p.suffix)
    else:
        out_path = None

    traces: list[TraceData] = []
    for solver in ("vbd", "mujoco"):
        csv_path = bench_dir / f"bench_{solver}.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found — skipping {solver}")
            continue
        traces.append(load_csv(csv_path, shaft_half_z_mm=args.shaft_half_z))

    if not traces:
        sys.exit("ERROR: No CSV files found. Run bench_solvers.py first.")

    make_figure(traces, hub_half_z_mm=args.hub_half_z, out=out_path, style=style)

    out2 = None
    if out_path is not None:
        p = pathlib.Path(out_path)
        out2 = p.parent / (p.stem + "_contacts" + p.suffix)
    make_figure_contacts(traces, hub_half_z_mm=args.hub_half_z, out=out2, style=style)

    out3 = None
    if out_path is not None:
        p = pathlib.Path(out_path)
        out3 = p.parent / (p.stem + "_forces" + p.suffix)
    make_figure_forces(traces, out=out3, style=style)

    if args.slide:
        out4 = None
        if out_path is not None:
            p = pathlib.Path(out_path)
            out4 = p.parent / (p.stem + "_slide" + p.suffix)
        make_figure_slide(traces, hub_half_z_mm=args.hub_half_z, out=out4, style=style)
