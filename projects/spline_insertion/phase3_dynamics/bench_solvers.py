# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Headless VBD vs MuJoCo solver benchmark.

Runs the identical scripted insertion trajectory under each solver and
prints a structured performance + physical-correctness comparison table.
Both solvers share the same geometry, kh, gains, and control inputs --
the only differences are algorithmic (integration scheme, contact model).

Fair-comparison notes
---------------------
Each solver ships with its own default substep count (VBD=8, MuJoCo=16)
tuned for stability at the default frame rate.  This means raw frame times
are NOT directly comparable -- use the per-substep time and work-units rows
for a normalised view.  To force equal substeps, pass --substeps N.

Usage
-----
    python bench_solvers.py                    # compare at solver defaults
    python bench_solvers.py --substeps 8       # force equal substeps (VBD native)
    python bench_solvers.py --substeps 16      # force equal substeps (MuJoCo native)
    python bench_solvers.py --solver mujoco    # single solver
    python bench_solvers.py --out ./results    # also write per-frame CSVs
    python bench_solvers.py --frames 300       # early cutoff (debug)
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import warp as wp

import newton
import newton.usd
from newton.geometry import HydroelasticSDF

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE   = pathlib.Path(__file__).parent
PHASE1 = HERE.parent / "phase1_geometry"
sys.path.insert(0, str(HERE))

# ── Re-use constants + GPU kernel from sim_dynamics ────────────────────────────
# sim_dynamics.py has no top-level side effects beyond registering the warp
# kernel; the viewer/shader code is guarded under __main__.
from sim_dynamics import (                      # noqa: E402
    Z, VBD_KH, MJ_KH, CONTACT_GAP, MESH_SDF_RES,
    SIM_FPS, N_SUBSTEPS_MJ, N_SUBSTEPS_VBD,
    CONTACT_MAX, MJ_NJMAX, SHAFT_MASS, SHAFT_RADIUS, SHAFT_LENGTH,
    LATERAL_K, STOP_K,
    MJ_ROT_D, MJ_DESCENT_K, MJ_LATERAL_D,
    VBD_ROT_D, VBD_DESCENT_K, VBD_LATERAL_D,
    VBD_ITERS, VBD_CONTACT_K,
    MJ_ITERS, MJ_LS_ITERS, MJ_IMPRATIO,
    INSERT_CLEARANCE,
    _make_shape_cfg, _load_mesh, _apply_shaft_forces,
)
from scripted_trajectory import get_controls, reset_trajectory, TOTAL_DURATION  # noqa: E402

N_WARMUP        = 5    # frames before timing — burns JIT + fills GPU caches
_SNAP_MM        = 0.8  # mm/frame drop that flags acceleration past commanded rate
                       # commanded descent during search = 25 mm/s = 0.42 mm/frame;
                       # 0.8 mm = 2× commanded → clear signature of reduced resistance
_BORE_ENTRY_MM  = 3.0  # shaft bottom must be this far inside bore to count as "entered"


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class FrameRecord:
    t:                 float
    phase:             str
    shaft_z_mm:        float
    shaft_angle_mod60: float   # degrees mod groove pitch (0°=aligned, 30°=worst)
    cmd_rot_dps:       float
    cmd_des_mmps:      float
    frame_ms:          float
    contact_area_mm2:  float = 0.0  # hydroelastic contact patch area
    n_contacts:        int   = 0    # number of reduced contact points
    max_depth_mm:      float = 0.0  # max penetration depth (AVBD convergence proxy)


@dataclass
class BenchResult:
    solver:       str
    n_substeps:   int
    solver_iters: int
    frames:       list[FrameRecord] = field(default_factory=list)

    # Filled by summarise()
    mean_frame_ms:        float = 0.0
    std_frame_ms:         float = 0.0
    mean_substep_ms:      float = 0.0   # mean_frame_ms / n_substeps  (normalised cost)
    work_units:           int   = 0     # n_substeps × solver_iters   (total work/frame)
    total_wall_s:         float = 0.0
    rtf:                  float = 0.0   # sim-time / wall-time
    shaft_half_z_mm:      float = 0.0
    # Engagement — two independent detectors:
    #   depth: shaft bottom crossed bore entrance by > _BORE_ENTRY_MM during search/engage
    #   snap:  single-frame Z drop > _SNAP_MM (2× commanded rate) during search/engage
    bore_entered:         bool  = False  # depth detector
    bore_entry_t_s:       float = float("inf")
    bore_entry_angle_deg: float = float("nan")
    snap_detected:        bool  = False  # velocity detector
    snap_t_s:             float = float("inf")
    snap_angle_deg:       float = float("nan")
    snap_max_mm:          float = 0.0   # largest single-frame drop during search/engage
    engaged:              bool  = False  # True if EITHER detector fired
    engagement_t_s:       float = float("inf")
    engagement_angle_deg: float = float("nan")
    deepest_center_z_mm:  float = 0.0
    deepest_bottom_z_mm:  float = 0.0
    insertion_depth_mm:   float = 0.0   # distance shaft bottom entered bore from hub top
    torsion_drift_dps:    float = 0.0   # mean |dθ/dt| during torsion_lock (ideal = 0)
    pullout_z_final_mm:   float = float("nan")  # shaft centre Z at trajectory end

    def summarise(self, hub_half_z_mm: float, shaft_half_z_mm: float) -> None:
        if not self.frames:
            return

        self.shaft_half_z_mm = shaft_half_z_mm
        ms = [f.frame_ms for f in self.frames]
        self.mean_frame_ms   = float(np.mean(ms))
        self.std_frame_ms    = float(np.std(ms))
        self.mean_substep_ms = self.mean_frame_ms / self.n_substeps
        self.work_units      = self.n_substeps * self.solver_iters
        self.total_wall_s    = sum(ms) / 1000.0
        sim_end = max(f.t for f in self.frames) + 1.0 / SIM_FPS
        self.rtf = sim_end / self.total_wall_s if self.total_wall_s > 0 else 0.0

        # ── Engagement: two complementary detectors ──────────────────────────
        # Commanded descent during search_groove = 25 mm/s = 0.42 mm/frame.
        # Detector 1 — depth: shaft bottom entered bore by > _BORE_ENTRY_MM.
        #   Catches gradual P-controller descent that finds the groove slowly.
        # Detector 2 — snap: single-frame drop > _SNAP_MM (2× commanded rate).
        #   Catches sudden acceleration when contact resistance collapses.
        # MuJoCo: force-level contacts → smooth descent, depth detector expected.
        # VBD: position-level AVBD → possible sharper snap, both may fire.
        bore_entry_thresh = hub_half_z_mm + shaft_half_z_mm - _BORE_ENTRY_MM
        prev_z: Optional[float] = None
        max_drop = 0.0

        for f in self.frames:
            if f.phase not in ("search", "hold"):
                prev_z = None
                continue

            # Depth detector
            if not self.bore_entered and f.shaft_z_mm < bore_entry_thresh:
                self.bore_entered      = True
                self.bore_entry_t_s    = f.t
                self.bore_entry_angle_deg = f.shaft_angle_mod60

            # Snap detector
            if prev_z is not None:
                drop = prev_z - f.shaft_z_mm
                max_drop = max(max_drop, drop)
                if not self.snap_detected and drop > _SNAP_MM:
                    self.snap_detected  = True
                    self.snap_t_s       = f.t
                    self.snap_angle_deg = f.shaft_angle_mod60
            prev_z = f.shaft_z_mm

        self.snap_max_mm = round(max_drop, 3)
        # Combined engaged flag: either detector suffices
        if self.bore_entered or self.snap_detected:
            self.engaged = True
            # Report whichever fired first
            if self.bore_entry_t_s <= self.snap_t_s:
                self.engagement_t_s       = self.bore_entry_t_s
                self.engagement_angle_deg = self.bore_entry_angle_deg
            else:
                self.engagement_t_s       = self.snap_t_s
                self.engagement_angle_deg = self.snap_angle_deg

        valid_z = [f.shaft_z_mm for f in self.frames if not math.isnan(f.shaft_z_mm)]
        self.deepest_center_z_mm = min(valid_z) if valid_z else float("nan")
        self.deepest_bottom_z_mm = self.deepest_center_z_mm - shaft_half_z_mm
        # Insertion depth measured from bore entrance (hub top face, +hub_half_z).
        # Positive = shaft entered bore; 0 = just touching; negative = didn't reach.
        self.insertion_depth_mm = hub_half_z_mm - self.deepest_bottom_z_mm

        # Torsion lock drift: how much does the shaft rotate while trapped?
        # Small = groove walls are stiff; large = contacts too soft / slipping.
        tlock = [f for f in self.frames if f.phase == "retract"]
        if len(tlock) >= 2:
            dt  = 1.0 / SIM_FPS
            diffs = []
            for i in range(len(tlock) - 1):
                d = abs(tlock[i + 1].shaft_angle_mod60 - tlock[i].shaft_angle_mod60)
                if d > 30.0:   # mod-60 wrap-around
                    d = abs(d - 60.0)
                diffs.append(d)
            self.torsion_drift_dps = float(np.mean(diffs)) / dt

        pullout = [f for f in self.frames if f.phase == "withdraw"
                   and not math.isnan(f.shaft_z_mm)]
        if pullout:
            self.pullout_z_final_mm = pullout[-1].shaft_z_mm


# ── Headless runner ────────────────────────────────────────────────────────────

class BenchRunner:
    """Newton simulation without viewer -- runs as fast as GPU allows."""

    def __init__(self, solver: str, substeps_override: Optional[int] = None) -> None:
        self._solver_type = solver
        self._frame_dt    = 1.0 / SIM_FPS

        for p in (PHASE1 / "shaft.usd", PHASE1 / "hub.usd"):
            if not p.exists():
                sys.exit(
                    f"ERROR: {p} not found.\n"
                    "Run  python gen_geometry.py  from phase1_geometry/ first."
                )

        hub_mesh,   hub_v   = _load_mesh(PHASE1 / "hub.usd",   "Hub",   build_sdf=True)
        shaft_mesh, shaft_v = _load_mesh(PHASE1 / "shaft.usd", "Shaft", build_sdf=True)

        hub_half_z   = float(hub_v[:,2].max()   - hub_v[:,2].min())   / 2.0
        shaft_half_z = float(shaft_v[:,2].max() - shaft_v[:,2].min()) / 2.0

        half_pitch_rad = math.pi / Z
        shaft_start_z  = hub_half_z + shaft_half_z + INSERT_CLEARANCE

        self._hub_half_z   = hub_half_z
        self._shaft_half_z = shaft_half_z
        self._shaft_z_min  = -hub_half_z - shaft_half_z
        self._shaft_z_max  = shaft_start_z

        kh        = VBD_KH if solver == "vbd" else MJ_KH
        shape_cfg = _make_shape_cfg(kh)

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_shape_mesh(
            -1,
            mesh=hub_mesh,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            cfg=shape_cfg,
            label="hub",
        )

        rot0 = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), half_pitch_rad)
        self._shaft_body = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, shaft_start_z), rot0),
            label="shaft",
        )
        builder.add_shape_mesh(
            self._shaft_body,
            mesh=shaft_mesh,
            cfg=shape_cfg,
            label="shaft_shape",
        )

        m   = SHAFT_MASS
        Izz = 0.5 * m * SHAFT_RADIUS ** 2
        Ixx = m * (3.0 * SHAFT_RADIUS ** 2 + SHAFT_LENGTH ** 2) / 12.0
        builder.body_mass[self._shaft_body]        = m
        builder.body_inv_mass[self._shaft_body]    = 1.0 / m
        builder.body_inertia[self._shaft_body]     = wp.mat33(
            Ixx, 0.0, 0.0,  0.0, Ixx, 0.0,  0.0, 0.0, Izz)
        builder.body_inv_inertia[self._shaft_body] = wp.mat33(
            1.0/Ixx, 0.0, 0.0,  0.0, 1.0/Ixx, 0.0,  0.0, 0.0, 1.0/Izz)

        joint_idx = builder.add_joint_free(self._shaft_body, parent=-1)
        builder.add_articulation(joints=[joint_idx])
        builder.color()
        self.model = builder.finalize()

        self._n_substeps = (
            substeps_override if substeps_override is not None
            else (N_SUBSTEPS_VBD if solver == "vbd" else N_SUBSTEPS_MJ)
        )
        if solver == "vbd":
            self._rot_d     = VBD_ROT_D
            self._descent_k = VBD_DESCENT_K
            self._lateral_d = VBD_LATERAL_D
            from newton.solvers import SolverVBD
            self.solver = SolverVBD(
                self.model,
                iterations=VBD_ITERS,
                rigid_contact_k_start=VBD_CONTACT_K,
            )
        else:
            self._rot_d     = MJ_ROT_D
            self._descent_k = MJ_DESCENT_K
            self._lateral_d = MJ_LATERAL_D
            from newton.solvers import SolverMuJoCo
            self.solver = SolverMuJoCo(
                self.model,
                use_mujoco_contacts=False,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                nconmax=CONTACT_MAX,
                njmax=MJ_NJMAX,
                iterations=MJ_ITERS,
                ls_iterations=MJ_LS_ITERS,
                impratio=MJ_IMPRATIO,
            )

        self.control = self.model.control()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Snapshot for state reset after warm-up
        self._init_body_q  = self.state_0.body_q.numpy().copy()
        self._init_body_qd = np.zeros(
            self.state_0.body_qd.numpy().shape, dtype=np.float32)

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=CONTACT_MAX,
            broad_phase="explicit",
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                output_contact_surface=True,
                reduce_contacts=True,
            ),
        )
        self.contacts = self.collision_pipeline.contacts()

    # ── internal helpers ───────────────────────────────────────────────────

    def _reset_state(self) -> None:
        """Restore body state to the initial configuration."""
        self.state_0.body_q.assign(self._init_body_q)
        self.state_0.body_qd.assign(self._init_body_qd)
        wp.synchronize_device(self.model.device)

    def _simulate_frame(self, target_omega_z: float, target_vel_z: float) -> None:
        """Run one full frame (n_substeps) of physics — no timing, no readback."""
        n_sub  = self._n_substeps
        sub_dt = self._frame_dt / n_sub
        device = self.model.device

        if self._solver_type != "mujoco":
            self.collision_pipeline.collide(self.state_0, self.contacts)

        for _ in range(n_sub):
            if self._solver_type == "mujoco":
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            wp.launch(
                kernel=_apply_shaft_forces,
                dim=1,
                inputs=(
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self._shaft_body,
                    float(target_omega_z),
                    float(target_vel_z),
                    float(self._rot_d),
                    float(self._descent_k),
                    float(LATERAL_K),
                    float(self._lateral_d),
                    float(self._shaft_z_min),
                    float(self._shaft_z_max),
                    float(STOP_K),
                ),
                device=device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _read_contact_metrics(self) -> tuple[float, int, float]:
        """Return (contact_area_mm2, n_contacts, max_depth_mm) — one readback per frame."""
        try:
            n_contacts = 0
            rcc = self.contacts.rigid_contact_count
            n_contacts = int(rcc.numpy()[0]) if hasattr(rcc, "numpy") else int(rcc)
        except Exception:
            n_contacts = 0

        contact_area_mm2 = 0.0
        max_depth_mm     = 0.0
        try:
            hydro = self.collision_pipeline.hydroelastic_sdf
            if hydro is not None:
                surface = hydro.get_contact_surface()
                n_faces = int(surface.face_contact_count.numpy()[0])
                if n_faces > 0:
                    depths = surface.contact_surface_depth.numpy()[:n_faces]
                    pts    = surface.contact_surface_point.numpy()[:n_faces * 3]
                    max_depth_mm = float(np.max(np.abs(depths))) * 1000.0
                    v0 = pts[0::3]; v1 = pts[1::3]; v2 = pts[2::3]
                    cross = np.cross(v1 - v0, v2 - v0)
                    contact_area_mm2 = float(
                        np.sum(0.5 * np.linalg.norm(cross, axis=1))
                    ) * 1.0e6
        except Exception:
            pass

        return contact_area_mm2, n_contacts, max_depth_mm

    # ── timed run ──────────────────────────────────────────────────────────

    def run(self, max_frames: Optional[int] = None, verbose: bool = True) -> BenchResult:
        n_sub  = self._n_substeps
        device = self.model.device

        # ── Warm-up: burns JIT compilation + fills GPU caches ──────────────
        # Run N_WARMUP frames at t=0 (shaft still, hub stationary -- no contacts),
        # then reset state so the timed run starts from clean initial conditions.
        if verbose:
            tag = self._solver_type.upper().ljust(6)
            print(f"  {tag} warm-up ({N_WARMUP} frames) ...", end=" ", flush=True)
        ctrl0 = get_controls(0.0)
        for _ in range(N_WARMUP):
            self._simulate_frame(math.radians(ctrl0.rotation_dps), -ctrl0.descent_mps)
        wp.synchronize_device(device)
        self._reset_state()
        reset_trajectory()   # clear groove-catch state from warm-up frames
        if verbose:
            print("done", flush=True)

        # ── Timed trajectory ───────────────────────────────────────────────
        result = BenchResult(
            solver=self._solver_type,
            n_substeps=n_sub,
            solver_iters=VBD_ITERS if self._solver_type == "vbd" else MJ_ITERS,
        )

        total_frames = int(TOTAL_DURATION * SIM_FPS)
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        sim_time  = 0.0
        bar_width = 40
        if verbose:
            print(f"  {tag} [", end="", flush=True)

        shaft_z_mm = 87.5   # initial shaft centre Z (mm) — updated each frame from GPU state
        for frame_no in range(total_frames):
            ctrl = get_controls(sim_time, shaft_z_mm=shaft_z_mm)
            if ctrl.done:
                break

            target_omega_z = math.radians(ctrl.rotation_dps)
            target_vel_z   = -ctrl.descent_mps   # trajectory +ve=down; kernel +ve=+Z=up

            wp.synchronize_device(device)
            t0 = time.perf_counter()

            self._simulate_frame(target_omega_z, target_vel_z)

            wp.synchronize_device(device)
            frame_ms = (time.perf_counter() - t0) * 1000.0

            # GPU -> CPU readback: body state
            q           = self.state_0.body_q.numpy()[self._shaft_body]
            shaft_z_mm  = float(q[2]) * 1000.0
            angle_rad   = 2.0 * math.atan2(float(q[5]), float(q[6]))
            angle_mod60 = math.degrees(angle_rad) % 60.0

            # GPU -> CPU readback: contact metrics (hydroelastic surface)
            contact_area_mm2, n_contacts, max_depth_mm = self._read_contact_metrics()

            result.frames.append(FrameRecord(
                t=round(sim_time, 4),
                phase=ctrl.phase,
                shaft_z_mm=round(shaft_z_mm, 3),
                shaft_angle_mod60=round(angle_mod60, 2),
                cmd_rot_dps=round(ctrl.rotation_dps, 2),
                cmd_des_mmps=round(ctrl.descent_mps * 1000.0, 3),
                frame_ms=round(frame_ms, 2),
                contact_area_mm2=round(contact_area_mm2, 4),
                n_contacts=n_contacts,
                max_depth_mm=round(max_depth_mm, 5),
            ))

            sim_time += self._frame_dt

            if verbose and frame_no % max(1, total_frames // bar_width) == 0:
                print(".", end="", flush=True)

        if verbose:
            print("]", flush=True)

        result.summarise(self._hub_half_z * 1000.0, self._shaft_half_z * 1000.0)
        return result


# ── Reporting ──────────────────────────────────────────────────────────────────

def _row(label: str, *vals: str, width: int = 26, note: str = "") -> str:
    label_col = f"  {label:<{width}}"
    row = label_col + "   ".join(f"{v:<14}" for v in vals)
    if note:
        row += f"  [{note}]"
    return row


def print_comparison(results: list[BenchResult]) -> None:
    sep   = "=" * 72
    thin  = "-" * 72
    names = [r.solver.upper() for r in results]

    all_same_sub = len({r.n_substeps for r in results}) == 1
    sub_note = (
        f"equal substeps ({results[0].n_substeps})"
        if all_same_sub else
        f"VBD={next(r.n_substeps for r in results if r.solver=='vbd')} "
        f"MJ={next(r.n_substeps for r in results if r.solver=='mujoco')} "
        f"substeps — use --substeps N for equal comparison"
    )
    print(f"\n{sep}")
    print("  PHASE 3 SOLVER BENCHMARK")
    print(f"  Trajectory : {TOTAL_DURATION:.0f} s scripted insertion (DIN 5480 z={Z})")
    print(f"  Geometry   : phase1_geometry/   kh VBD={VBD_KH:.0e}  MJ={MJ_KH:.0e}")
    print(f"  Substeps   : {sub_note}")
    print(sep)

    # Config
    print("\n  SOLVER CONFIGURATION")
    print(thin)
    print(_row("Substeps / frame",     *[str(r.n_substeps)   for r in results]))
    print(_row("Solver iterations",    *[str(r.solver_iters) for r in results]))
    print(_row("Work units / frame",   *[str(r.work_units)   for r in results],
               note="substeps × iters"))
    print(_row("Contact stiffness kh", *[
        f"{VBD_KH:.0e}" if r.solver == "vbd" else f"{MJ_KH:.0e}"
        for r in results], note="must differ: VBD=position-level, MJ=force-level"))
    print(_row("sub_dt (ms)",          *[
        f"{1000.0 / (SIM_FPS * r.n_substeps):.2f}" for r in results]))
    print(_row("Trajectory forces",    *["identical" for _ in results],
               note="same scripted_trajectory.py for both"))

    # Performance
    print("\n  PERFORMANCE  (warm-up excluded)")
    print(thin)
    print(_row("Mean frame time (ms)",
               *[f"{r.mean_frame_ms:.1f} ±{r.std_frame_ms:.1f}" for r in results]))
    print(_row("Per-substep time (ms)", *[f"{r.mean_substep_ms:.2f}" for r in results],
               note="normalised"))
    print(_row("Total wall time (s)",   *[f"{r.total_wall_s:.1f}" for r in results]))
    print(_row("Real-time factor",      *[f"{r.rtf:.2f}×" for r in results]))
    if len(results) == 2:
        speedup = results[0].mean_frame_ms / results[1].mean_frame_ms
        faster  = names[0] if speedup < 1 else names[1]
        ratio   = min(speedup, 1.0/speedup)
        sub_speedup = results[0].mean_substep_ms / results[1].mean_substep_ms
        sub_faster  = names[0] if sub_speedup < 1 else names[1]
        sub_ratio   = min(sub_speedup, 1.0/sub_speedup)
        print(_row("Frame speedup",     *["", f"{faster} is {1.0/ratio:.2f}×"]))
        print(_row("Per-substep speedup", *["", f"{sub_faster} is {1.0/sub_ratio:.2f}×"],
                   note="apple-to-apple"))

    # Physical correctness
    print("\n  PHYSICAL CORRECTNESS")
    print(thin)

    def bore_str(r: BenchResult) -> str:
        if r.bore_entered:
            return f"YES @ {r.bore_entry_t_s:.1f}s  {r.bore_entry_angle_deg:.1f}°"
        return "NO"

    def snap_str(r: BenchResult) -> str:
        if r.snap_detected:
            return f"YES @ {r.snap_t_s:.1f}s  {r.snap_angle_deg:.1f}°"
        return f"NO  (max {r.snap_max_mm:.2f} mm)"

    print(_row("Bore entered (depth)",    *[bore_str(r) for r in results],
               note=f">{_BORE_ENTRY_MM:.0f} mm past hub top"))
    print(_row("Groove snap (velocity)",  *[snap_str(r) for r in results],
               note=f">{_SNAP_MM:.1f} mm/frame = 2× cmd"))
    print(_row("Insertion depth (mm)",    *[f"{r.insertion_depth_mm:.1f}" for r in results],
               note="VBD AVBD blocks tip-root contact; MJ force-level allows small penetration"))
    print(_row("Shaft bottom deepest (mm)", *[f"{r.deepest_bottom_z_mm:.1f}" for r in results],
               note="negative = past hub centerline"))
    print(_row("Torsion drift (deg/s)",  *[f"{r.torsion_drift_dps:.2f}" for r in results],
               note="mean |dtheta/dt| while torsion-locked; 0 = groove walls fully stiff"))
    print(_row("Pull-out final Z (mm)",
               *[f"{r.pullout_z_final_mm:.1f}" if not math.isnan(r.pullout_z_final_mm) else "—"
                 for r in results],
               note="shaft centre; ~87mm = fully withdrawn to start position"))

    # Per-phase shaft-Z trajectory — shows insertion progress
    print("\n  SHAFT Z AT PHASE END (mm)  [centre; hub top = +hub_half_z]")
    print(thin)
    # hub_half_z = insertion_depth + deepest_bottom_z  (from insertion_depth = hub_half_z - deepest_bottom_z)
    hub_half_z_mm = results[0].insertion_depth_mm + results[0].deepest_bottom_z_mm
    all_phases = list(dict.fromkeys(
        f.phase for r in results for f in r.frames
    ))
    print(_row("Hub top at Z =",
               *[f"+{hub_half_z_mm:.1f} mm" for _ in results],
               note="shaft bottom = ctr_z - shaft_half_z"))
    header = _row("Phase -> ctr_z (bot_z)", *[r.solver.upper() for r in results])
    print(header)
    for phase in all_phases:
        vals = []
        for r in results:
            phase_frames = [f for f in r.frames if f.phase == phase
                            and not math.isnan(f.shaft_z_mm)]
            if phase_frames:
                cz = phase_frames[-1].shaft_z_mm
                bz = cz - r.shaft_half_z_mm
                vals.append(f"{cz:6.1f} ({bz:+6.1f})")
            else:
                vals.append("—")
        print(_row(phase, *vals))

    # Per-phase timing
    print("\n  MEAN FRAME TIME BY PHASE (ms)")
    print(thin)
    header = _row("Phase", *[r.solver.upper() for r in results])
    print(header)
    for phase in all_phases:
        vals = []
        for r in results:
            phase_ms = [f.frame_ms for f in r.frames if f.phase == phase]
            vals.append(f"{np.mean(phase_ms):.1f}" if phase_ms else "—")
        print(_row(phase, *vals))

    print(f"\n{sep}\n")


def save_csv(result: BenchResult, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bench_{result.solver}.csv"
    with path.open("w", newline="") as f:
        if not result.frames:
            return
        writer = csv.DictWriter(f, fieldnames=result.frames[0].__dataclass_fields__)
        writer.writeheader()
        for rec in result.frames:
            writer.writerow(rec.__dict__)
    print(f"  Saved {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver", choices=["vbd", "mujoco", "both"], default="both",
                        help="Solver(s) to benchmark (default: both)")
    parser.add_argument("--substeps", type=int, default=None, metavar="N",
                        help="Override substeps for BOTH solvers (equal comparison)")
    parser.add_argument("--frames", type=int, default=None,
                        help="Max frames per solver (default: full trajectory)")
    parser.add_argument("--out", default=None, metavar="DIR",
                        help="Directory to write per-frame CSV files")
    parser.add_argument("--device", default="cuda:0",
                        help="Warp device (default: cuda:0)")
    args = parser.parse_args()

    wp.init()

    solvers = (["vbd", "mujoco"] if args.solver == "both" else [args.solver])

    substep_note = (
        f"  Substeps forced to {args.substeps} for both solvers (--substeps)\n"
        if args.substeps else
        f"  Each solver at its default substeps (VBD={N_SUBSTEPS_VBD}, MuJoCo={N_SUBSTEPS_MJ})\n"
        f"  Use --substeps N for an equal-substep comparison.\n"
    )

    results: list[BenchResult] = []

    print(f"\nPhase 3 Solver Benchmark  —  device: {args.device}")
    print(f"Trajectory: {TOTAL_DURATION:.0f} s  |  z={Z} geometry")
    print(substep_note)

    for solver in solvers:
        print(f"  Building {solver.upper()} model ...")
        runner = BenchRunner(solver, substeps_override=args.substeps)
        result = runner.run(max_frames=args.frames, verbose=True)
        print(f"  Done — {result.total_wall_s:.1f} s wall  |  RTF {result.rtf:.2f}×\n")
        results.append(result)

        if args.out:
            save_csv(result, pathlib.Path(args.out))

    print_comparison(results)
