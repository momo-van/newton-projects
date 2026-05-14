# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Phase 3 -- dynamic spline insertion.

Supports two solvers selectable via ``--solver``:

``mujoco`` (default)
    SolverMuJoCo with ``use_mujoco_contacts=False`` -- MuJoCo/MJWarp integrates
    rigid-body dynamics while Newton's hydroelastic pipeline supplies all
    contact points.  Recommended for rigid-body contact dynamics.

``vbd``
    SolverVBD (Variational Body Dynamics).  Primarily designed for
    deformables/cables; the rigid-body path is a secondary code path.
    Kept as a fallback option.

Both solvers share the same model setup and the same simulation loop so the
only code path that differs is solver construction.

Spring-damper forces keep the shaft on-axis while the user drives:

  Rotation (deg)   -45 ... +45   direct angle position
  Descent (mm/s)   -20 ... +20   +ve = insert, -ve = withdraw

Controls (ImGui side panel -- press H to toggle)

Run
---
::

    python sim_dynamics.py                                   # MuJoCo solver (default)
    python sim_dynamics.py --solver vbd                      # VBD solver
    python sim_dynamics.py --auto                            # scripted trajectory
    python sim_dynamics.py --auto --record videos/run.mp4   # record to MP4 (auto-exits)
"""

from __future__ import annotations

import atexit
import csv
import math
import pathlib
import sys
import time

try:
    from video_capture import ScreenRecorder as _ScreenRecorder
    _HAS_RECORDER = True
except ImportError:
    _HAS_RECORDER = False

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.geometry import HydroelasticSDF

HERE   = pathlib.Path(__file__).parent
PHASE1 = HERE.parent / "phase1_geometry"

# ── Parameters ────────────────────────────────────────────────────────────────

Z: int = 6

INSERT_CLEARANCE: float = 0.015
# VBD uses AVBD position-level correction → soft kh is fine (AVBD corrects penetration directly).
# MuJoCo with use_mujoco_contacts=False is force-based: contact force = kh × volume.
# At kh=5e6 and descent force ≈ 4 N the equilibrium penetration is ~5 mm → shaft sinks through.
# MuJoCo needs kh ≈ 10× higher so a sub-mm penetration generates enough resisting force.
# kh back-of-envelope at max rotation torque T=0.31 N·m, groove wall area A≈1e-3 m²:
#   δ_eq = T / (r × kh × A) where r=0.025 m
#   VBD: AVBD position correction drives δ→0 regardless of kh (kh only sets AVBD penalty scale)
#   MuJoCo: force-based → δ_eq = 0.31 / (0.025 × kh × 1e-3) → kh≥5e8 gives δ<0.025mm
VBD_KH: float = 5.0e6   # AVBD corrects to zero; kh just sets AVBD penalty magnitude
MJ_KH:  float = 2.0e9   # force-based: δ_eq = F/(kh×A); at 25N/tooth, A≈1e-5m² → δ≈1.25mm
CONTACT_GAP:      float = 3.0e-4
# 512 voxels over the hub OD (~90 mm) → ~0.176 mm/voxel.
# Geometry uses GAP=0.4 mm (0.2 mm/side clearance): 0.2/0.176 = 1.14× margin.
# Same margin as the res=256/GAP=0.8mm pair that confirmed snapping works.
MESH_SDF_RES:     int   = 512

SIM_FPS:    int = 60
# Both solvers apply spring-damper forces via a Warp kernel that writes body_f.
# Those external forces are consumed as *explicit* loads regardless of the solver's
# internal integrator.  Stability ceiling: d < m/sub_dt (damper), d < 2m/sub_dt (vel ctrl).
# MuJoCo uses 16 substeps (sub_dt=1.04ms) for tighter per-step contact correction;
# at that dt the body_f ceiling becomes m/sub_dt=960 N·s/m — comfortable for all gains.
N_SUBSTEPS_MJ:  int = 8   # 8 substeps: matches VBD cadence
N_SUBSTEPS_VBD: int = 8   # 8 substeps: sub_dt≈2ms, adequate for AVBD position correction
READBACK_INTERVAL: int = 6

# nconmax: slots for contact points in the pipeline and MuJoCo.
# Peak observed: ~104 contacts at full engagement (6-tooth, hydroelastic).
CONTACT_MAX: int = 132   # nconmax -- peak observed ~130 contacts at full engagement
# njmax: must be >= nconmax * 3 (elliptic cone, 3 rows/contact).
# 130 contacts × 3 = 390 rows → 512 is the minimum safe tile.
# Reducing below 512 is not viable: the hydroelastic pipeline produces 65-130 contacts
# throughout the trajectory (even during search phase), so no smaller tile fits.
MJ_NJMAX:    int = 512

_SDF_BAND = (-CONTACT_GAP * 4.0, CONTACT_GAP * 4.0)

# Shaft material (solid-cylinder approximation)
SHAFT_MASS:   float = 1.0    # kg
SHAFT_RADIUS: float = 0.025  # m
SHAFT_LENGTH: float = 0.100  # m

# Shared gains used when no solver-specific override applies
LATERAL_K: float = 1.0e5   # N/m  (lateral position spring -- purely geometric, same for both)
STOP_K:    float = 1.0e5   # N/m  (end-stop penalty spring)

# Controller gains -- explicit semi-implicit Euler limits at sub_dt=1/(60*8)=2.08ms.
# For m=1 kg, Izz=3.125e-4 kg·m²:
#   translational damper: d  < m  / sub_dt = 480  N·s/m
#   velocity controller:  d  < 2m / sub_dt = 960  N·s/m
#   rotational damper:    d  < Izz/ sub_dt = 0.15 N·m·s/rad
# Both solvers obey these limits because body_f is applied as an explicit external load.
MJ_ROT_D:     float = 0.1     # N·m·s/rad  (< 0.15)
MJ_DESCENT_K: float = 800.0   # N·s/m      (< 960)
MJ_LATERAL_D: float = 400.0   # N·s/m      (< 480)

VBD_ROT_D:     float = 0.13    # N·m·s/rad  (< 0.15 limit at 8 substeps)
VBD_DESCENT_K: float = 800.0   # N·s/m      (< 960)
VBD_LATERAL_D: float = 400.0   # N·s/m      (< 480)

# VBD-specific
VBD_ITERS:    int   = 32   # 32 iters: halves AVBD residual penetration vs 16; safe at 8 substeps
VBD_CONTACT_K: float = 3.0e4  # 3× previous; safe with 32 iters (ratio/iter = 937 vs 1250 at explosion)

# MuJoCo-specific
# Implicit integrator is unconditionally stable -- no explicit stability ceiling.
# Tuning here targets *convergence quality* when tooth geometry engages groove walls:
#   iterations  : Newton solver iterations per step; more = better contact convergence
#                 (analogous to VBD_ITERS: both improve per-step constraint resolution)
#   ls_iterations: line-search steps within each Newton iteration; more = smaller/safer
#                 step when the Jacobian changes abruptly at first tooth contact
#   impratio    : friction-to-normal impedance ratio; 1.0 = MuJoCo default (equal impedance).
#                 Newton's own nut-bolt hydroelastic example uses 1.0; higher values
#                 ill-condition the constraint matrix and can cause CG divergence.
MJ_SOLVER:   str   = "newton"  # "newton" | "cg"
MJ_ITERS:    int   = 15     # Newton iterations/step -- matches newton nut-bolt example
MJ_LS_ITERS: int   = 100    # line-search steps -- matches newton nut-bolt example (robust convergence)
MJ_IMPRATIO: float = 2.0    # slightly above MuJoCo default; adds friction stiffness without ill-conditioning

CONTACT_MU: float = 0.30   # dry steel-on-steel (kinetic); was 0.15 (lubricated)

def _make_shape_cfg(kh: float, gap_override: float | None = None) -> newton.ModelBuilder.ShapeConfig:
    gap = (gap_override if gap_override is not None else CONTACT_GAP) * 4.0
    return newton.ModelBuilder.ShapeConfig(
        mu=CONTACT_MU,
        kh=kh,
        ke=2.0e6,
        kd=5.0e3,
        gap=gap,
        is_hydroelastic=True,
    )


# ── Warp kernel ───────────────────────────────────────────────────────────────

@wp.kernel
def _apply_shaft_forces(
    body_q:         wp.array(dtype=wp.transform),
    body_qd:        wp.array(dtype=wp.spatial_vector),
    body_f:         wp.array(dtype=wp.spatial_vector),
    shaft_idx:      int,
    target_omega_z: float,
    target_vel_z:   float,
    rot_d:          float,
    descent_k:      float,
    lateral_k:      float,
    lateral_d:      float,
    z_min:          float,
    z_max:          float,
    stop_k:         float,
):
    q  = body_q[shaft_idx]
    qd = body_qd[shaft_idx]

    pos = wp.transform_get_translation(q)

    # Spatial velocity: (vx, vy, vz, wx, wy, wz) -- linear first
    vx = qd[0];  vy = qd[1];  vz = qd[2]
    wx = qd[3];  wy = qd[4];  wz = qd[5]

    # Lateral centering spring-damper
    fx = -lateral_k * pos[0] - lateral_d * vx
    fy = -lateral_k * pos[1] - lateral_d * vy

    # Descent velocity P-controller
    fz = descent_k * (target_vel_z - vz)

    # Soft stops at Z limits
    if pos[2] < z_min:
        fz = fz + stop_k * (z_min - pos[2])
    if pos[2] > z_max:
        fz = fz + stop_k * (z_max - pos[2])

    # Rotation velocity P-controller -- user sets spin speed, not target angle.
    # Contact resistance naturally limits rotation when engaged in groove;
    # no unbounded torque buildup when the shaft is trapped.
    torque_z = rot_d * (target_omega_z - wz)
    torque_x = -rot_d * wx
    torque_y = -rot_d * wy

    # Spatial force: (fx, fy, fz, tx, ty, tz) -- linear first
    body_f[shaft_idx] = wp.spatial_vector(fx, fy, fz, torque_x, torque_y, torque_z)


# ── USD mesh loader ───────────────────────────────────────────────────────────

def _load_mesh(
    usd_path: pathlib.Path,
    prim_name: str,
    build_sdf: bool = False,
) -> tuple[newton.Mesh, np.ndarray]:
    stage    = Usd.Stage.Open(str(usd_path))
    prim     = stage.GetPrimAtPath(f"/World/{prim_name}")
    usd_mesh = newton.usd.get_mesh(prim, load_normals=True)
    vertices = np.array(usd_mesh.vertices, dtype=np.float32)
    indices  = np.array(usd_mesh.indices,  dtype=np.int32)
    normals  = (
        np.array(usd_mesh.normals, dtype=np.float32)
        if usd_mesh.normals is not None else None
    )
    mesh = newton.Mesh(vertices, indices, normals=normals)
    if build_sdf:
        mesh.build_sdf(
            max_resolution=MESH_SDF_RES,
            narrow_band_range=_SDF_BAND,
            margin=CONTACT_GAP,
        )
    return mesh, vertices


# ── Example ───────────────────────────────────────────────────────────────────

class Example:
    """
    Phase 3 dynamic spline insertion.

    The solver is selected at construction time via ``args.solver``
    (``"mujoco"`` or ``"vbd"``).  Both solvers share the same model,
    collision pipeline, and ``simulate()`` loop.
    """

    def __init__(self, viewer, args) -> None:
        self.fps      = SIM_FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer   = viewer
        self._paused  = False
        self._solver_type = args.solver

        shaft_usd = PHASE1 / "shaft.usd"
        hub_usd   = PHASE1 / "hub.usd"
        for p in (shaft_usd, hub_usd):
            if not p.exists():
                sys.exit(
                    f"ERROR: {p} not found.\n"
                    "Run  python gen_geometry.py  from phase1_geometry/ first."
                )

        print("Loading geometry and building SDFs ...")
        hub_mesh,   hub_verts   = _load_mesh(hub_usd,   "Hub",   build_sdf=True)
        shaft_mesh, shaft_verts = _load_mesh(shaft_usd, "Shaft", build_sdf=True)

        hub_half_z   = float(hub_verts[:, 2].max()   - hub_verts[:, 2].min()) / 2.0
        shaft_half_z = float(shaft_verts[:, 2].max() - shaft_verts[:, 2].min()) / 2.0

        half_pitch_rad  = math.pi / Z
        shaft_start_z   = hub_half_z + shaft_half_z + INSERT_CLEARANCE
        self._shaft_z_min = -hub_half_z - shaft_half_z
        self._shaft_z_max = shaft_start_z

        self._shaft_z         = shaft_start_z
        self._shaft_angle_rad = half_pitch_rad
        self._hub_half_z      = hub_half_z
        self._shaft_half_z    = shaft_half_z

        self._rotation_dps  = 0.0   # target rotation speed (deg/s)
        self._descent_mps   = 0.0
        self._readback_tick = 0
        self._vis_mode      = 0     # 0=Solid, 1=X-Ray, 2=Wire
        self._mesh_alpha    = 1.0   # controlled by _apply_vis_mode
        self._shape_shader  = None  # set after viewer init
        self._mesh_alpha_loc = None

        # ── Scripted / auto mode ──────────────────────────────────────────
        self._auto          = getattr(args, "auto", False)
        self._metrics_out   = getattr(args, "metrics_out", None)
        self._metrics:      list[dict] = []
        self._auto_wall_t0: float      = 0.0   # set on first auto step
        self._auto_phase:   str        = "init"
        self._last_net_force: list     = [0.0] * 6   # [fx,fy,fz,tx,ty,tz] from Δ(mv)/dt

        # ── Control recording / replay ────────────────────────────────────
        self._rec_mode:   str                       = "idle"  # idle | recording | replaying
        self._rec_frames: list[tuple[float, float]] = []      # (rotation_dps, descent_mps)
        self._replay_idx: int                       = 0
        self._rec_name:   str                       = "control_recording"
        self._rec_file:   pathlib.Path              = HERE / f"{self._rec_name}.json"
        self._rec_saved:  bool                      = False   # frames saved to _rec_file

        # ── Video recording ───────────────────────────────────────────────
        record_path = getattr(args, "record", None)
        self._record_path = pathlib.Path(record_path) if record_path else None
        self._recorder    = None   # ScreenRecorder, created on first step
        self._step_count  = 0
        if self._record_path and not _HAS_RECORDER:
            print("WARNING: --record requires mss + opencv-python + pywin32. Skipping.")
            self._record_path = None
        if self._record_path:
            atexit.register(self._stop_recording)

        # ── Newton model ─────────────────────────────────────────────────
        builder = newton.ModelBuilder(gravity=0.0)

        # VBD requires tight contact gap + soft AVBD beta to snap teeth into grooves.
        # ShapeConfig gap = contact_gap * 4.  Must satisfy:
        #   gap > voxel_size (~0.176 mm at SDF_RES=512) so contacts aren't missed, AND
        #   gap < per-side clearance (0.2 mm, GAP=0.4 mm diametral) so contact only
        #   fires near groove walls, not when tooth is freely inside groove.
        # 4.8e-5 m → ShapeConfig gap = 0.192 mm: 0.176 mm < 0.192 mm < 0.2 mm  ✓
        # MuJoCo uses standard CONTACT_GAP (force-based, naturally tolerates coarser gap).
        contact_gap = 4.8e-5 if self._solver_type == "vbd" else CONTACT_GAP
        shape_cfg = _make_shape_cfg(VBD_KH if self._solver_type == "vbd" else MJ_KH,
                                    gap_override=contact_gap)

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

        # Real mass and inertia (solid cylinder)
        m   = SHAFT_MASS
        Izz = 0.5 * m * SHAFT_RADIUS ** 2
        Ixx = m * (3.0 * SHAFT_RADIUS ** 2 + SHAFT_LENGTH ** 2) / 12.0
        I     = wp.mat33(Ixx, 0.0, 0.0,  0.0, Ixx, 0.0,  0.0, 0.0, Izz)
        I_inv = wp.mat33(1.0 / Ixx, 0.0, 0.0,  0.0, 1.0 / Ixx, 0.0,  0.0, 0.0, 1.0 / Izz)
        builder.body_mass[self._shaft_body]        = m
        builder.body_inv_mass[self._shaft_body]    = 1.0 / m
        builder.body_inertia[self._shaft_body]     = I
        builder.body_inv_inertia[self._shaft_body] = I_inv

        # Free joint + articulation (required for VBD; harmless for MuJoCo)
        joint_idx = builder.add_joint_free(self._shaft_body, parent=-1)
        builder.add_articulation(joints=[joint_idx])

        builder.color()
        self.model = builder.finalize()

        # ── Solver ───────────────────────────────────────────────────────
        default_substeps = N_SUBSTEPS_VBD if self._solver_type == "vbd" else N_SUBSTEPS_MJ
        self._n_substeps = getattr(args, "substeps", None) or default_substeps

        # Both solvers apply body_f as explicit external loads → same stability limits.
        if self._solver_type == "vbd":
            self._rot_d     = VBD_ROT_D
            self._descent_k = VBD_DESCENT_K
            self._lateral_d = VBD_LATERAL_D
        else:
            self._rot_d     = MJ_ROT_D
            self._descent_k = MJ_DESCENT_K
            self._lateral_d = MJ_LATERAL_D

        if self._solver_type == "vbd":
            from newton.solvers import SolverVBD
            # beta=500: soft AVBD lets shaft settle ~0.3 mm into groove before correction,
            # preventing SDF voxel noise (±0.176 mm) from being treated as hard contact
            # and blocking groove entry.  High beta (100 000 default) makes AVBD too stiff:
            # teeth get pushed back out of grooves before snap can occur.
            vbd_beta  = 500
            vbd_iters = 10
            vbd_k     = 1e3
            self.solver = SolverVBD(
                self.model,
                iterations=vbd_iters,
                rigid_contact_k_start=vbd_k,
                rigid_avbd_beta=vbd_beta,
                rigid_enable_dahl_friction=True,
            )
            print(f"  VBD: gap={contact_gap*1e3:.2f}mm  beta={vbd_beta}"
                  f"  iters={vbd_iters}  substeps={self._n_substeps}"
                  f"  dahl_friction=True")
        else:  # mujoco (default)
            from newton.solvers import SolverMuJoCo
            self.solver = SolverMuJoCo(
                self.model,
                use_mujoco_contacts=False,   # use Newton hydroelastic contacts
                solver=MJ_SOLVER,
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

        # Ensure body_q is consistent with joint_q (important for MuJoCo)
        newton.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            self.state_0,
        )

        # ── Hydroelastic pipeline ────────────────────────────────────────
        sdf_hydro_cfg = HydroelasticSDF.Config(
            output_contact_surface=True,
            reduce_contacts=True,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=CONTACT_MAX,
            broad_phase="explicit",
            sdf_hydroelastic_config=sdf_hydro_cfg,
        )
        self.contacts = self.collision_pipeline.contacts()

        self._initial_body_q  = self.state_0.body_q.numpy().copy()
        self._initial_joint_q = self.model.joint_q.numpy().copy()

        # ── Shape colours ─────────────────────────────────────────────────
        # Orange hub + purple shaft: both outside the jet heatmap palette
        # (jet uses blue→cyan→green→yellow→red) so contact patches stay legible.
        # Hub = shape 0 (added first, body=-1); shaft = shape 1 (body=shaft_body).
        sc = self.model.shape_color.numpy()
        sc[0] = [0.48, 0.55, 0.72]   # hub   -- pale blue-gray (distinct from jet heatmap)
        sc[1] = [0.78, 0.70, 0.54]   # shaft -- pale warm tan (contrasts hub, pale vs hydro)
        self.model.shape_color.assign(sc)

        # ── Viewer + viz ─────────────────────────────────────────────────
        from contact_viz import SplineContactViz
        self.viewer.set_model(self.model)
        # Camera: closer + steeper pitch for insertion close-up.
        # pos=(0.07,-0.07,0.10), pitch=-40, yaw=135, fov=38 zooms in on
        # the hub-shaft interface at z≈20–60 mm.
        _CAM_POS   = wp.vec3(0.07, -0.07, 0.10)
        _CAM_PITCH = -40.0
        _CAM_YAW   = 135.0
        _CAM_FOV   = 38.0
        self.viewer.set_camera(pos=_CAM_POS, pitch=_CAM_PITCH, yaw=_CAM_YAW)
        self.viewer.camera.fov = _CAM_FOV

        # Default visual state: wireframe on, contact normals on.
        self.viewer.renderer.draw_wireframe = True
        self.viewer.show_contacts           = True

        # Grab shape shader for mesh_alpha uniform (X-ray mode).
        try:
            self._shape_shader   = self.viewer.renderer._shape_shader
            self._mesh_alpha_loc = self._shape_shader._get_uniform_location("mesh_alpha")
        except Exception:
            pass

        self.viz = SplineContactViz(self.collision_pipeline, self.model, viewer)
        self.viewer.register_ui_callback(self._controls_ui, position="side")
        self.viewer.register_ui_callback(self._overlay_ui,  position="side")

        # Store for Reset Camera button
        self._cam_pos   = _CAM_POS
        self._cam_pitch = _CAM_PITCH
        self._cam_yaw   = _CAM_YAW
        self._cam_fov   = _CAM_FOV

        n_hydro = int(
            (self.model.shape_flags.numpy()
             & newton.ShapeFlags.HYDROELASTIC).astype(bool).sum()
        )
        print("\n" + "=" * 60)
        print("  Phase 3 -- Spline Insertion (Dynamics)")
        solver_detail = f"{MJ_SOLVER.upper()} iters={MJ_ITERS} ls={MJ_LS_ITERS}" if self._solver_type == "mujoco" else ""
        print(f"  Solver              : {self._solver_type.upper()}  {solver_detail}")
        print(f"  Geometry            : phase1_geometry/ (z={Z}, 6-tooth)")
        print(f"  Hydroelastic shapes : {n_hydro} / {self.model.shape_count}")
        print(f"  Shaft mass          : {m:.2f} kg")
        print(f"  N substeps          : {self._n_substeps}  (60 fps)")
        print(f"  Shaft start Z       : {shaft_start_z * 1000:.1f} mm")
        print(f"  Hub bore span       : +/-{hub_half_z * 1000:.0f} mm")
        print("=" * 60)
        print("  H -- toggle ImGui panel   F -- frame camera")
        print("=" * 60 + "\n")

    # -----------------------------------------------------------------------
    # ImGui controls panel
    # -----------------------------------------------------------------------

    def _apply_vis_mode(self) -> None:
        """Apply the current _vis_mode (0=Solid, 1=X-Ray, 2=Wire) to the viewer."""
        if self._vis_mode == 0:    # Solid — opaque, no wireframe
            self._mesh_alpha = 1.0
            self.viewer.show_visual             = True
            self.viewer.renderer.draw_wireframe = False
        elif self._vis_mode == 1:  # X-Ray — semi-transparent, hydro viz visible through hull
            self._mesh_alpha = 0.18
            self.viewer.show_visual             = True
            self.viewer.renderer.draw_wireframe = False
        else:                      # Wire — opaque + wireframe edges
            self._mesh_alpha = 1.0
            self.viewer.show_visual             = True
            self.viewer.renderer.draw_wireframe = True

    def _apply_xray_gl(self) -> None:
        """Update mesh_alpha uniform + GL state every frame (called from UI callback AND render()).
        UI-callback call: state persists to the NEXT frame's shape render.
        render() call:    state active for the CURRENT frame (belt-and-suspenders).

        Newton only resets GL_DEPTH_TEST at frame start — blend, depthMask, and cullFace
        are NOT reset, so our settings persist across frames.

        Three-part fix for flicker-free X-ray transparency:
          1. glDepthMask(FALSE)      — shapes don't write depth → no Z-fighting at same depth
          2. glDisable(GL_CULL_FACE) — both faces rendered → no culling flicker on rotation
          3. vertex shader 0.9999·w  — actual contact depths always win GL_LESS depth test
        """
        if self._mesh_alpha_loc is None:
            return
        from OpenGL import GL as gl
        alpha = self._mesh_alpha
        prev = int(gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM) or 0)
        gl.glUseProgram(self._shape_shader.shader_program.id)
        gl.glUniform1f(self._mesh_alpha_loc, float(alpha))
        gl.glUseProgram(prev)
        if alpha < 1.0:
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            # No glDepthMask or glCullFace changes — depth writes stay ON so contact
            # normal lines are depth-tested normally against the pushed shape depths.
            # Z-fighting between hub/shaft eliminated via per-shape depth bias in vertex shader.

    def _controls_ui(self, imgui) -> None:
        self._apply_xray_gl()   # GL state set here persists to next frame's shape render
        imgui.separator()
        imgui.text(f"Solver: {self._solver_type.upper()}  |  t {self.sim_time:.1f}s")
        if self._auto:
            from scripted_trajectory import expected_duration
            _est = expected_duration()
            pct = min(100.0, self.sim_time / _est * 100.0)
            imgui.text_colored(
                (0.4, 0.9, 1.0, 1.0),
                f"AUTO  [{self._auto_phase}]  {self.sim_time:.1f}/{_est:.0f}s  ({pct:.0f}%)",
            )
        imgui.separator()
        imgui.text("Shaft State")
        imgui.separator()
        imgui.spacing()

        angle_mod = math.degrees(self._shaft_angle_rad) % 60.0
        depth_m   = self._hub_half_z - (self._shaft_z + self._shaft_half_z)

        imgui.text(f"  Z position   : {self._shaft_z * 1000:.1f} mm")
        imgui.text(f"  Angle mod 60 : {angle_mod:.1f} deg  (0 = engaged)")
        imgui.text(f"  Bore depth   : {max(depth_m, 0.0) * 1000:.1f} mm")
        imgui.spacing()

        imgui.separator()
        imgui.text("Drive Controls")
        imgui.separator()
        imgui.spacing()

        # Drive sliders disabled during replay (values come from recording)
        _replaying = self._rec_mode == "replaying"
        if _replaying:
            imgui.begin_disabled()

        # Rotation: slider for quick sweep + input field for exact value
        imgui.set_next_item_width(150)
        ch, val = imgui.slider_float(
            "##rot_sl", self._rotation_dps, -180.0, 180.0, "%.1f"
        )
        if ch:
            self._rotation_dps = float(val)
        imgui.same_line(spacing=4)
        imgui.set_next_item_width(70)
        ch2, val2 = imgui.input_float("Rot (°/s)", self._rotation_dps, 0.0, 0.0, "%.1f")
        if ch2:
            self._rotation_dps = max(-360.0, min(360.0, float(val2)))

        imgui.spacing()

        # Descent: slider for quick sweep + input field for exact value
        imgui.set_next_item_width(150)
        ch3, val3 = imgui.slider_float(
            "##des_sl", self._descent_mps * 1000.0, -20.0, 20.0, "%.1f"
        )
        if ch3:
            self._descent_mps = float(val3) / 1000.0
        imgui.same_line(spacing=4)
        imgui.set_next_item_width(70)
        ch4, val4 = imgui.input_float("Des (mm/s)", self._descent_mps * 1000.0, 0.0, 0.0, "%.1f")
        if ch4:
            self._descent_mps = max(-50.0, min(50.0, float(val4))) / 1000.0

        if _replaying:
            imgui.end_disabled()

        imgui.spacing()

        label = "Resume" if self._paused else "Pause"
        if imgui.button(label):
            if self._paused and self._auto and self._auto_phase == "done":
                self._reset()
            else:
                self._paused = not self._paused
        imgui.same_line()
        if imgui.button("Reset"):
            self._reset()

        imgui.spacing()
        imgui.separator()
        imgui.text("Recording")
        imgui.separator()
        imgui.spacing()

        n_frames = len(self._rec_frames)
        dur_s    = n_frames / self.fps

        # Filename input — editable at any time except during active recording/replay
        if self._rec_mode == "idle":
            imgui.set_next_item_width(200)
            ch_name, new_name = imgui.input_text("##recname", self._rec_name)
            if ch_name and new_name.strip():
                self._rec_name  = new_name.strip()
                self._rec_file  = HERE / f"{self._rec_name}.json"
                self._rec_saved = False   # name changed, existing frames no longer match file
        else:
            imgui.text_colored((0.6, 0.6, 0.6, 1.0), self._rec_name)

        imgui.spacing()

        if self._rec_mode == "idle":
            # ── REC button ──────────────────────────────────────────────────
            if imgui.button("● REC"):
                self._rec_frames = []
                self._rec_saved  = False
                self._rec_mode   = "recording"

            # ── Save / Replay (only if frames exist) ────────────────────────
            if n_frames > 0:
                imgui.same_line()
                if imgui.button("Save"):
                    self._save_recording()
                imgui.same_line()
                if imgui.button("▶ Replay"):
                    if not self._rec_saved:
                        self._save_recording()
                    self._reset()
                    self._rec_mode   = "replaying"
                    self._replay_idx = 0
                    self._paused     = False
            elif self._rec_file.exists():
                # No in-memory frames but file exists on disk — offer to load & replay
                imgui.same_line()
                if imgui.button("▶ Replay from disk"):
                    self._load_recording()
                    self._reset()
                    self._rec_mode   = "replaying"
                    self._replay_idx = 0
                    self._paused     = False

            # Status line
            if n_frames > 0:
                saved_str = f"  saved → {self._rec_file.name}" if self._rec_saved else "  unsaved"
                imgui.text(f"  {n_frames} frames  ({dur_s:.1f}s){saved_str}")
            elif self._rec_file.exists():
                imgui.text_colored((0.6, 0.6, 0.6, 1.0),
                                   f"  {self._rec_file.name} on disk")

        elif self._rec_mode == "recording":
            imgui.text_colored((1.0, 0.25, 0.25, 1.0), "● REC")
            imgui.same_line()
            if imgui.button("Stop"):
                self._rec_mode = "idle"
            imgui.text(f"  {n_frames} frames  ({dur_s:.1f}s)")

        elif self._rec_mode == "replaying":
            pct = self._replay_idx / n_frames * 100 if n_frames else 0
            imgui.text_colored((0.25, 1.0, 0.45, 1.0),
                               f"▶ {self._replay_idx}/{n_frames}  ({pct:.0f}%)")
            imgui.same_line()
            if imgui.button("Stop"):
                self._rec_mode = "idle"
                self._paused   = True
            imgui.text(f"  {self._rec_file.name}")

        imgui.separator()
        imgui.text("View")
        imgui.separator()
        imgui.spacing()

        # Geometry visibility: cycle Solid → X-Ray → Wire
        vis_modes = ["Solid", "X-Ray", "Wire"]
        vis_mode  = vis_modes[self._vis_mode]
        if imgui.button(f"Geo: {vis_mode}"):
            self._vis_mode = (self._vis_mode + 1) % 3
            self._apply_vis_mode()
        imgui.same_line()
        # Hydroelastic contact surface overlay
        hs_label = "Hydro: ON " if self.viewer.show_hydro_contact_surface else "Hydro: OFF"
        if imgui.button(hs_label):
            self.viewer.show_hydro_contact_surface = not self.viewer.show_hydro_contact_surface
        imgui.same_line()
        if imgui.button("Reset Camera"):
            self.viewer.set_camera(pos=self._cam_pos, pitch=self._cam_pitch, yaw=self._cam_yaw)
            self.viewer.camera.fov = self._cam_fov

        if self.viz.contact_area_mm2 < 1.0 and depth_m > 0.001:
            imgui.spacing()
            imgui.text_colored(
                (0.2, 1.0, 0.3, 1.0),
                "  Keys aligned -- push down!",
            )

    # -----------------------------------------------------------------------
    # Telemetry overlay (auto mode only)
    # -----------------------------------------------------------------------

    _PHASE_COLORS: dict = {
        "vertical_demo": (0.70, 0.70, 0.70, 1.0),
        "approach":      (0.40, 0.70, 1.00, 1.0),
        "sit_on_hub":    (1.00, 0.90, 0.20, 1.0),
        "search_groove": (1.00, 0.60, 0.10, 1.0),
        "engage":        (0.30, 1.00, 0.40, 1.0),
        "push_in":       (0.10, 1.00, 0.10, 1.0),
        "friction_demo": (1.00, 0.40, 0.10, 1.0),
        "torsion_lock":  (1.00, 0.20, 0.20, 1.0),
        "pull_out":      (0.30, 0.90, 1.00, 1.0),
        "done":          (0.60, 0.60, 0.60, 1.0),
    }

    def _overlay_ui(self, imgui) -> None:
        from scripted_trajectory import expected_duration

        io  = imgui.get_io()
        W   = 240
        PAD = 10
        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size.x - PAD, io.display_size.y - PAD),
            imgui.Cond_.always,
            imgui.ImVec2(1.0, 1.0),   # pivot: anchor bottom-right corner
        )
        imgui.set_next_window_size(imgui.ImVec2(W, 0), imgui.Cond_.always)
        imgui.set_next_window_bg_alpha(0.78)
        flags = (
            imgui.WindowFlags_.no_title_bar.value
            | imgui.WindowFlags_.no_resize.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_scrollbar.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.always_auto_resize.value
        )
        visible, _ = imgui.begin("##telem", flags=flags)
        if not visible:
            imgui.end()
            return

        # Phase / mode header
        if self._auto:
            color = self._PHASE_COLORS.get(self._auto_phase, (0.9, 0.9, 0.9, 1.0))
            label = self._auto_phase.upper().replace("_", " ")
            imgui.text_colored(color, f"  {label}")
            _est = expected_duration()
            pct = min(1.0, self.sim_time / _est)
            imgui.progress_bar(pct, imgui.ImVec2(W - 16, 6), "")
            imgui.text(f"  t {self.sim_time:5.1f} / {_est:.0f} s")
        else:
            imgui.text_colored((0.6, 0.85, 1.0, 1.0), "  INTERACTIVE")
            imgui.text(f"  t {self.sim_time:5.1f} s")
        imgui.separator()

        # Applied forces / torques
        f_axial = self._descent_k * self._descent_mps
        t_rot   = self._rot_d * math.radians(self._rotation_dps)
        ax_sym  = "↓" if f_axial > 0 else ("↑" if f_axial < 0 else "·")
        rot_sym = "↻" if t_rot   > 0 else ("↺" if t_rot   < 0 else "·")
        imgui.text(f"  F axial  {ax_sym}  {abs(f_axial):5.1f} N")
        imgui.text(f"  T rot    {rot_sym}  {abs(t_rot * 1000):5.1f} mN·m")
        imgui.separator()

        # Shaft state
        imgui.text(f"  Shaft Z  {self._shaft_z * 1000:6.1f} mm")
        angle_mod = math.degrees(self._shaft_angle_rad) % 60.0
        engaged   = angle_mod < 5.0 or angle_mod > 55.0
        engaged_str = "  ★ ENGAGED" if engaged else ""
        imgui.text(f"  θ mod60  {angle_mod:5.1f}°{engaged_str}")

        imgui.end()

    # -----------------------------------------------------------------------
    # Simulation
    # -----------------------------------------------------------------------

    def _stop_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.stop()
            out = self._record_path
            try:
                self._recorder.save(out)
                print(f"[Record] Video saved -> {out}  ({self._recorder.frame_count} frames)")
            except Exception as e:
                print(f"[Record] Save skipped: {e}")
            self._recorder = None

    def _save_metrics(self) -> None:
        out = pathlib.Path(self._metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if self._metrics:
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._metrics[0].keys()))
                writer.writeheader()
                writer.writerows(self._metrics)
            print(f"[Auto] Metrics saved -> {out}  ({len(self._metrics)} rows)")

    def _save_recording(self) -> None:
        import json
        self._rec_file = HERE / f"{self._rec_name}.json"
        data = {"fps": self.fps, "solver": self._solver_type,
                "frames": self._rec_frames}
        self._rec_file.write_text(json.dumps(data))
        self._rec_saved = True
        print(f"[Rec] Saved {len(self._rec_frames)} frames "
              f"({len(self._rec_frames)/self.fps:.1f}s) -> {self._rec_file.name}")

    def _load_recording(self) -> None:
        import json
        self._rec_file = HERE / f"{self._rec_name}.json"
        if not self._rec_file.exists():
            print(f"[Rec] File not found: {self._rec_file.name}")
            return
        data = json.loads(self._rec_file.read_text())
        self._rec_frames = [tuple(f) for f in data["frames"]]
        self._rec_saved  = True
        print(f"[Rec] Loaded {len(self._rec_frames)} frames "
              f"({len(self._rec_frames)/self.fps:.1f}s) from {self._rec_file.name}")

    def _reset(self) -> None:
        zeros = np.zeros(self.state_0.body_qd.numpy().shape, dtype=np.float32)
        for state in (self.state_0, self.state_1):
            state.body_q.assign(self._initial_body_q)
            state.body_qd.assign(zeros)
        self.model.joint_q.assign(self._initial_joint_q)
        self._shaft_z         = self._shaft_z_max
        self._shaft_angle_rad = math.pi / Z
        self._rotation_dps    = 0.0
        self._descent_mps     = 0.0
        self._readback_tick   = 0
        self._paused          = False
        self.sim_time         = 0.0
        self._auto_wall_t0    = 0.0
        self._auto_phase      = "init"
        if self._auto:
            from scripted_trajectory import reset_trajectory
            reset_trajectory()
        if self._rec_mode == "recording":
            self._rec_frames = []
            self._rec_saved  = False
        self._rec_mode   = "idle"
        self._replay_idx = 0

    def simulate(self) -> None:
        target_omega_z = math.radians(self._rotation_dps)   # rad/s
        target_vel_z   = -self._descent_mps
        n_sub  = self._n_substeps
        sub_dt = self.frame_dt / n_sub

        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(n_sub):
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
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        frame_wall_start = time.perf_counter()
        self._step_count += 1

        # ── Start recorder after a few frames (viewer window now guaranteed visible)
        if self._record_path and self._recorder is None and self._step_count >= 5:
            self._recorder = _ScreenRecorder(
                target_fps=30,
                window_title="Newton Viewer",
            )
            self._recorder.start()
            print(f"[Record] Recording started -> {self._record_path}")

        # ── Scripted control ─────────────────────────────────────────────────
        if self._auto and not self._paused:
            from scripted_trajectory import get_controls
            if self._auto_wall_t0 == 0.0:
                self._auto_wall_t0 = frame_wall_start
            ctrl = get_controls(self.sim_time, shaft_z_mm=self._shaft_z * 1000.0)
            self._rotation_dps = ctrl.rotation_dps
            self._descent_mps  = ctrl.descent_mps
            self._auto_phase   = ctrl.phase
            if ctrl.done:
                self._paused = True
                if self._metrics_out:
                    self._save_metrics()
                self._stop_recording()
                print(f"[Auto] Trajectory complete ({self.sim_time:.1f}s sim time)")
                if self._record_path:
                    self.viewer.renderer.close()

        # ── Control recording / replay ───────────────────────────────────────
        if not self._paused:
            if self._rec_mode == "recording":
                self._rec_frames.append((self._rotation_dps, self._descent_mps))
            elif self._rec_mode == "replaying":
                if self._replay_idx < len(self._rec_frames):
                    rot, desc = self._rec_frames[self._replay_idx]
                    self._rotation_dps = rot
                    self._descent_mps  = desc
                    self._replay_idx  += 1
                else:
                    self._rec_mode = "idle"
                    self._paused   = True
                    print(f"[Rec] Replay complete  ({self._replay_idx} frames  "
                          f"{self._replay_idx/self.fps:.1f}s sim)")

        _track_forces = self._auto and self._metrics_out and not self._paused
        if _track_forces:
            _qd_pre = self.state_0.body_qd.numpy()[self._shaft_body].copy()

        if not self._paused:
            self.simulate()

        if _track_forces:
            qd_post = self.state_0.body_qd.numpy()[self._shaft_body]
            dv = (qd_post[:3] - _qd_pre[:3]) / self.frame_dt
            dw = (qd_post[3:] - _qd_pre[3:]) / self.frame_dt
            _Ixx = SHAFT_MASS * (3.0 * SHAFT_RADIUS**2 + SHAFT_LENGTH**2) / 12.0
            _Izz = 0.5 * SHAFT_MASS * SHAFT_RADIUS**2
            self._last_net_force = [
                float(SHAFT_MASS * dv[0]), float(SHAFT_MASS * dv[1]), float(SHAFT_MASS * dv[2]),
                float(_Ixx * dw[0]),       float(_Ixx * dw[1]),       float(_Izz * dw[2]),
            ]

        # Throttled GPU->CPU readback for ImGui display
        self._readback_tick += 1
        if self._readback_tick >= READBACK_INTERVAL:
            self._readback_tick = 0
            q = self.state_0.body_q.numpy()[self._shaft_body]
            self._shaft_z         = float(q[2])
            self._shaft_angle_rad = 2.0 * math.atan2(float(q[5]), float(q[6]))

        self.viz.update(self.contacts)

        # ── Metrics logging ──────────────────────────────────────────────────
        if self._auto and self._metrics_out:
            frame_ms = (time.perf_counter() - frame_wall_start) * 1000.0
            wall_elapsed = time.perf_counter() - self._auto_wall_t0 if self._auto_wall_t0 else 0.0
            self._metrics.append({
                "t":                    round(self.sim_time, 4),
                "wall_t":               round(wall_elapsed, 4),
                "phase":                self._auto_phase,
                "cmd_rot_dps":          round(self._rotation_dps, 2),
                "cmd_des_mmps":         round(self._descent_mps * 1000.0, 3),
                "shaft_z_mm":           round(self._shaft_z * 1000.0, 3),
                "shaft_angle_mod60":    round(math.degrees(self._shaft_angle_rad) % 60.0, 2),
                "contact_area_mm2":     round(self.viz.contact_area_mm2, 3),
                "n_contacts":           self.viz.reduced_contact_count,
                "max_depth_mm":         round(self.viz.max_depth_mm, 4),
                "frame_ms":             round(frame_ms, 2),
                # Net body force/torque from Newton's 2nd law: Δ(mv)/dt  (contact + applied)
                "net_fx_N":             round(self._last_net_force[0], 3),
                "net_fy_N":             round(self._last_net_force[1], 3),
                "net_fz_N":             round(self._last_net_force[2], 3),
                "net_tx_Nm":            round(self._last_net_force[3], 5),
                "net_ty_Nm":            round(self._last_net_force[4], 5),
                "net_tz_Nm":            round(self._last_net_force[5], 5),
                # Applied force/torque from trajectory commands (approx, ignores vel error)
                "applied_fz_N":         round(self._descent_k * self._descent_mps, 3),
                "applied_tz_Nm":        round(self._rot_d * math.radians(self._rotation_dps), 5),
            })

        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viz.render()
        self._apply_xray_gl()   # belt-and-suspenders: also set here for current frame
        self.viewer.end_frame()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, str(HERE.parent / "phase2_kinematic"))  # reuse contact_viz

    # Patch Newton's shape shaders BEFORE viewer creation to add mesh_alpha uniform.
    # Fragment: drives output alpha so geometry is semi-transparent in X-ray mode.
    # Vertex:   pushes clip-space depth to far plane when mesh_alpha < 0.5 so
    #           ALL contact/hydro viz (rendered after shapes) always passes depth test.
    import newton._src.viewer.gl.shaders as _nw_shaders
    _nw_shaders.shape_fragment_shader = (
        _nw_shaders.shape_fragment_shader
        .replace(
            "uniform float exposure;",
            "uniform float exposure;\nuniform float mesh_alpha;",
        )
        .replace(
            "FragColor = vec4(color, 1.0);",
            "FragColor = vec4(color, mesh_alpha);",
        )
    )
    _nw_shaders.shape_vertex_shader = (
        _nw_shaders.shape_vertex_shader
        .replace(                                           # declare in global scope
            "uniform mat4 light_space_matrix;",
            "uniform mat4 light_space_matrix;\nuniform float mesh_alpha;",
        )
        .replace(                                           # push depth to near-far in X-ray mode
            "    gl_Position = projection * view * worldPos;",
            "    gl_Position = projection * view * worldPos;\n"
            "    if (mesh_alpha < 0.5) {\n"
            "        // Bias by red channel so shapes get distinct depth layers:\n"
            "        // shaft (cobalt, r≈0.05) → 0.9997w (shallower, in front)\n"
            "        // hub   (magenta,r≈1.00) → 0.9999w (deeper,   behind shaft)\n"
            "        // contact viz at actual depths (0.8-0.93) always wins GL_LESS.\n"
            "        gl_Position.z = gl_Position.w * (0.9997 + aObjectColor.r * 0.0002);\n"
            "    }",
        )
    )

    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        choices=["vbd", "mujoco"],
        default="mujoco",
        help=(
            "Physics solver: 'mujoco' (default -- MuJoCo/MJWarp with Newton "
            "hydroelastic contacts) or 'vbd' (Variational Body Dynamics, "
            "primarily for deformables/cables)."
        ),
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        default=False,
        help="Run scripted insertion trajectory (scripted_trajectory.py) for solver comparison.",
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        metavar="PATH",
        help="CSV path for per-frame metrics (only used with --auto).",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=None,
        metavar="N",
        help="Override substeps per frame (default: 8 for VBD, 16 for MuJoCo).",
    )
    parser.add_argument(
        "--vbd-tune",
        action="store_true",
        default=False,
        help=(
            "Experimental VBD tuning for groove snap: tighter contact gap (0.05 mm), "
            "lower avbd_beta (2000), fewer iterations (10), 16 substeps, Dahl friction."
        ),
    )
    parser.add_argument(
        "--record",
        default=None,
        metavar="PATH",
        help=(
            "Record the Newton Viewer window to this MP4 path. "
            "Requires mss, opencv-python, pywin32. "
            "Best used with --auto so recording stops automatically."
        ),
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
