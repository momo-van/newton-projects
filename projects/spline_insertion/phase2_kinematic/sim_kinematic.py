# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Phase 2 -- interactive spline insertion with hydroelastic contact visualisation.

Loads the 6-tooth straight-key geometry from phase1_geometry/ (shaft.usd,
hub.usd) and lets the user interactively control:

  * Rotation angle -- direct position control (drag to search engagement angle)
  * Descent speed  -- push the shaft in or pull it back

Both controls start at zero.  The shaft is kinematic (zero-mass); position and
velocity are overridden every frame by the _drive_shaft kernel.  Contact forces
are computed by the hydroelastic pipeline and visualised as a pressure-coloured
wireframe patch.

Controls (ImGui side panel -- press H to toggle)
-------------------------------------------------
  Rotation (deg)   -45 ... +45   direct angle position
  Descent (mm/s)   -20 ... +20   +ve = insert, -ve = withdraw
  Pause / Resume
  Reset

Run
---
::

    python sim_kinematic.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.geometry import HydroelasticSDF

from contact_viz import SplineContactViz

HERE   = pathlib.Path(__file__).parent
PHASE1 = HERE.parent / "phase1_geometry"   # where the correct USDs live

# ── Parameters ────────────────────────────────────────────────────────────────

Z: int = 6                        # teeth -- must match phase1_geometry

INSERT_CLEARANCE: float = 0.030   # initial gap above hub entrance (m)
SHAPE_KH:         float = 5.0e7  # hydroelastic stiffness (Pa/m)
CONTACT_GAP:      float = 3.0e-4  # running clearance (m)
MESH_SDF_RES:     int   = 64

SIM_FPS: int = 60

_SDF_BAND = (-CONTACT_GAP * 4.0, CONTACT_GAP * 4.0)

SHAPE_CFG_HYDRO = newton.ModelBuilder.ShapeConfig(
    mu=0.15,
    kh=SHAPE_KH,
    ke=2.0e6,
    kd=2.0e3,
    gap=CONTACT_GAP * 4.0,
    is_hydroelastic=True,
)


# ── Warp kernel ───────────────────────────────────────────────────────────────

@wp.kernel
def _drive_shaft(
    body_q:    wp.array(dtype=wp.transform),
    body_qd:   wp.array(dtype=wp.spatial_vector),
    shaft_idx: int,
    target_z:  float,
    vel_z:     float,
    angle_z:   float,
    omega_z:   float,
):
    """Set shaft transform and velocity (kinematic drive)."""
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle_z)
    body_q[shaft_idx]  = wp.transform(wp.vec3(0.0, 0.0, target_z), rot)
    body_qd[shaft_idx] = wp.spatial_vector(0.0, 0.0, vel_z, 0.0, 0.0, omega_z)


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
    Phase 2 interactive spline insertion (kinematic).

    Step loop
    ---------
    1. ``_drive_shaft`` kernel writes body_q / body_qd on GPU.
    2. ``collide()`` computes the hydroelastic contact surface.
    3. ``viz.update()`` throttled GPU->CPU sync + ImGui metrics.
    """

    def __init__(self, viewer) -> None:
        self.fps      = SIM_FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer   = viewer
        self._paused  = False

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

        # Shaft starts above hub entrance, rotated half pitch (30 deg = misaligned)
        half_pitch_rad  = math.pi / Z
        shaft_start_z   = hub_half_z + shaft_half_z + INSERT_CLEARANCE
        self._shaft_z_min = -hub_half_z - shaft_half_z
        self._shaft_z_max = shaft_start_z

        self._shaft_z         = shaft_start_z
        self._shaft_angle_rad = half_pitch_rad
        self._hub_half_z      = hub_half_z
        self._shaft_half_z    = shaft_half_z

        # Controls: rotation = direct angle; descent = velocity (m/s)
        self._rotation_deg = math.degrees(half_pitch_rad)   # start misaligned
        self._descent_mps  = 0.0

        # ── Newton model ─────────────────────────────────────────────────
        builder = newton.ModelBuilder(gravity=0.0)

        builder.add_shape_mesh(
            -1,
            mesh=hub_mesh,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            cfg=SHAPE_CFG_HYDRO,
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
            cfg=SHAPE_CFG_HYDRO,
            label="shaft_shape",
        )
        builder.body_mass[self._shaft_body]        = 0.0
        builder.body_inv_mass[self._shaft_body]    = 0.0
        builder.body_inertia[self._shaft_body]     = wp.mat33(0.0)
        builder.body_inv_inertia[self._shaft_body] = wp.mat33(0.0)

        builder.color()
        self.model = builder.finalize()

        # ── Hydroelastic pipeline ────────────────────────────────────────
        sdf_hydro_cfg = HydroelasticSDF.Config(
            output_contact_surface=True,
            reduce_contacts=True,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            broad_phase="explicit",
            sdf_hydroelastic_config=sdf_hydro_cfg,
        )
        self.state_0  = self.model.state()
        self.contacts = self.collision_pipeline.contacts()

        self._initial_body_q = self.state_0.body_q.numpy().copy()

        # ── Viewer + SplineContactViz ────────────────────────────────────
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.10, -0.15, hub_half_z),
            pitch=15.0,
            yaw=45.0,
        )
        self.viz = SplineContactViz(self.collision_pipeline, self.model, viewer)
        self.viewer.register_ui_callback(self._controls_ui, position="side")

        n_hydro = int(
            (self.model.shape_flags.numpy()
             & newton.ShapeFlags.HYDROELASTIC).astype(bool).sum()
        )
        print("\n" + "=" * 60)
        print("  Phase 2 -- Spline Insertion (Kinematic + Hydroelastic)")
        print(f"  Geometry            : phase1_geometry/ (z={Z}, 6-tooth)")
        print(f"  Hydroelastic shapes : {n_hydro} / {self.model.shape_count}")
        print(f"  Shaft start Z       : {shaft_start_z * 1000:.1f} mm")
        print(f"  Hub bore span       : +/-{hub_half_z * 1000:.0f} mm")
        print(f"  Initial rotation    : {math.degrees(half_pitch_rad):.0f} deg (misaligned)")
        print("=" * 60)
        print("  H -- toggle ImGui panel   F -- frame camera")
        print("=" * 60 + "\n")

    # -----------------------------------------------------------------------
    # ImGui controls panel
    # -----------------------------------------------------------------------

    def _controls_ui(self, imgui) -> None:
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

        imgui.push_item_width(210)

        ch, val = imgui.slider_float(
            "Rotation (deg)##rot", self._rotation_deg, -45.0, 45.0, "%.1f"
        )
        if ch:
            self._rotation_deg = float(val)

        ch2, val2 = imgui.slider_float(
            "Descent (mm/s)##des", self._descent_mps * 1000.0, -20.0, 20.0, "%.1f"
        )
        if ch2:
            self._descent_mps = float(val2) / 1000.0

        imgui.pop_item_width()
        imgui.spacing()

        label = "Resume" if self._paused else "Pause"
        if imgui.button(label):
            self._paused = not self._paused
        imgui.same_line()
        if imgui.button("Reset"):
            self._reset()

        # Engagement hint
        if self.viz.contact_area_mm2 < 1.0 and depth_m > 0.001:
            imgui.spacing()
            imgui.text_colored(
                (0.2, 1.0, 0.3, 1.0),
                "  Keys aligned -- push down!",
            )

    # -----------------------------------------------------------------------
    # Simulation
    # -----------------------------------------------------------------------

    def _reset(self) -> None:
        self.state_0.body_q.assign(self._initial_body_q)
        zeros = np.zeros(self.state_0.body_qd.numpy().shape, dtype=np.float32)
        self.state_0.body_qd.assign(zeros)
        self._shaft_z         = self._shaft_z_max
        self._shaft_angle_rad = math.pi / Z
        self._rotation_deg    = math.degrees(math.pi / Z)
        self._descent_mps     = 0.0
        self._paused          = False

    def simulate(self) -> None:
        wp.launch(
            kernel=_drive_shaft,
            dim=1,
            inputs=(
                self.state_0.body_q,
                self.state_0.body_qd,
                self._shaft_body,
                float(self._shaft_z),
                float(-self._descent_mps),
                float(self._shaft_angle_rad),
                float(0.0),
            ),
            device=self.model.device,
        )
        self.collision_pipeline.collide(self.state_0, self.contacts)

    def step(self) -> None:
        if not self._paused:
            self._shaft_angle_rad = math.radians(self._rotation_deg)
            self._shaft_z = max(
                self._shaft_z_min,
                min(self._shaft_z_max, self._shaft_z - self._descent_mps * self.frame_dt),
            )
            self.simulate()

        self.viz.update(self.contacts)
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viz.render()
        self.viewer.end_frame()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example, args)
