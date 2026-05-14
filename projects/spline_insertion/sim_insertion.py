# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Splined shaft → hub insertion simulation.

Kinematically drives a DIN 5480 16-tooth splined shaft downward into its
mating hub along the Z-axis.  Newton mesh-SDF contacts fire as the teeth
engage the bore and are displayed in the viewer as contact arrows.

The shaft is treated as a zero-mass (kinematic) body: its position and
velocity are overridden every substep via a Warp kernel so the solver
cannot deflect it.  The hub is fixed to the world.

Controls
--------
  F      frame camera on model
  H      toggle ImGui side panel

  (Pause / Resume / Reset are in the ImGui side panel)

Run
---
::

    python sim_insertion.py

Pre-requisite
-------------
::

    python gen_geometry.py   # generates shaft.usd and hub.usd
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.solvers import SolverVBD

HERE = pathlib.Path(__file__).parent

# ── Parameters ────────────────────────────────────────────────────────────────

#: Shaft descent speed during insertion (m/s).
INSERT_SPEED: float = 0.005

#: Initial clearance above the hub top before insertion begins (m).
INSERT_CLEARANCE: float = 0.030

#: Simulation frames per second.
SIM_FPS: int = 60

#: Physics substeps per rendered frame (more → more stable contact).
SIM_SUBSTEPS: int = 8

#: SDF grid resolution (cells per longest axis).  Higher = sharper tooth
#: contact at the cost of longer build time.
MESH_SDF_RES: int = 128

#: Contact gap / SDF narrow-band half-width (m).
CONTACT_GAP: float = 0.001

SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    mu=0.15,
    ke=2.0e6,
    kd=2.0e3,
    gap=CONTACT_GAP,
    density=7800.0,
)

_SDF_BAND = (-2.0 * CONTACT_GAP, 2.0 * CONTACT_GAP)


# ── Warp kernel: kinematic shaft drive ───────────────────────────────────────

@wp.kernel
def _drive_shaft(
    body_q:    wp.array(dtype=wp.transform),
    body_qd:   wp.array(dtype=wp.spatial_vector),
    shaft_idx: int,
    target_z:  float,
    vel_z:     float,
):
    """
    Override shaft position and velocity every substep.

    Setting both body_q and body_qd prevents the VBD solver from
    drifting the kinematic body via contact impulses.
    """
    tf  = body_q[shaft_idx]
    rot = wp.transform_get_rotation(tf)
    pos = wp.transform_get_translation(tf)
    body_q[shaft_idx]  = wp.transform(wp.vec3(pos[0], pos[1], target_z), rot)
    body_qd[shaft_idx] = wp.spatial_vector(0.0, 0.0, vel_z, 0.0, 0.0, 0.0)


# ── USD helpers ───────────────────────────────────────────────────────────────

def _load_mesh(
    usd_path: pathlib.Path,
    prim_name: str,
    build_sdf: bool = False,
) -> tuple[newton.Mesh, np.ndarray]:
    """Load a Newton Mesh from a USD file; optionally build its SDF."""
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


# ── Example class ─────────────────────────────────────────────────────────────

class Example:
    """
    Splined shaft insertion simulation.

    Step loop
    ---------
    1. Advance target Z (CPU-side) at ``INSERT_SPEED``.
    2. ``collide()``  — broad + narrow phase, once per frame.
    3. Per substep: apply ``_drive_shaft`` kernel, then ``solver.step()``.
    4. ``render()``   — log bodies + contact arrows.
    """

    def __init__(self, viewer) -> None:
        self.fps          = SIM_FPS
        self.frame_dt     = 1.0 / self.fps
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt       = self.frame_dt / self.sim_substeps
        self.sim_time     = 0.0
        self.viewer       = viewer
        self._paused      = False
        self._inserting   = True

        shaft_usd = HERE / "shaft.usd"
        hub_usd   = HERE / "hub.usd"
        for p in (shaft_usd, hub_usd):
            if not p.exists():
                sys.exit(
                    f"ERROR: {p.name} not found.\n"
                    "Run  python gen_geometry.py  first."
                )

        # ── Load geometry and build SDFs ──────────────────────────────────
        print("Loading geometry and building SDFs (may take ~15 s)…")
        hub_mesh,   hub_verts   = _load_mesh(hub_usd,   "Hub",   build_sdf=True)
        shaft_mesh, shaft_verts = _load_mesh(shaft_usd, "Shaft", build_sdf=True)

        hub_half_z   = float(hub_verts[:, 2].max()   - hub_verts[:, 2].min()) / 2.0
        shaft_half_z = float(shaft_verts[:, 2].max() - shaft_verts[:, 2].min()) / 2.0

        # Hub is centred at z = 0 (spans −hub_half_z … +hub_half_z).
        # Shaft starts above the hub entrance; ends with its centre at hub centre.
        shaft_start_z = hub_half_z + shaft_half_z + INSERT_CLEARANCE
        shaft_end_z   = 0.0   # shaft centre aligned with hub centre = fully engaged

        self._shaft_start_z  = shaft_start_z
        self._shaft_end_z    = shaft_end_z
        self._shaft_z        = shaft_start_z
        self._shaft_half_z   = shaft_half_z
        self._hub_half_z     = hub_half_z

        # ── Newton model ─────────────────────────────────────────────────
        builder = newton.ModelBuilder(gravity=0.0)   # kinematic insertion — no gravity

        builder.add_shape_mesh(
            -1,   # world / static
            mesh=hub_mesh,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            cfg=SHAPE_CFG,
            label="hub",
        )

        self._shaft_body = builder.add_link(
            xform=wp.transform(
                wp.vec3(0.0, 0.0, shaft_start_z), wp.quat_identity()
            ),
            label="shaft",
        )
        builder.add_shape_mesh(
            self._shaft_body,
            mesh=shaft_mesh,
            cfg=SHAPE_CFG,
            label="shaft_shape",
        )
        # Zero mass → kinematic body; solver applies no net impulse to it.
        builder.body_mass[self._shaft_body]        = 0.0
        builder.body_inv_mass[self._shaft_body]    = 0.0
        builder.body_inertia[self._shaft_body]     = wp.mat33(0.0)
        builder.body_inv_inertia[self._shaft_body] = wp.mat33(0.0)

        builder.color()
        self.model = builder.finalize()

        # ── Collision pipeline + VBD solver ──────────────────────────────
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            broad_phase="explicit",
        )
        self.state_0  = self.model.state()
        self.state_1  = self.model.state()
        self.control  = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        self._initial_body_q = self.state_0.body_q.numpy().copy()

        self.solver = SolverVBD(
            self.model,
            iterations=12,
            friction_epsilon=0.1,
            rigid_contact_k_start=1.0e5,
            rigid_body_contact_buffer_size=512,
        )

        # ── Viewer ───────────────────────────────────────────────────────
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.10, -0.12, hub_half_z * 0.5),
            pitch=20.0,
            yaw=45.0,
        )
        self.viewer.register_ui_callback(self._controls_ui, position="side")

        print("\n" + "=" * 55)
        print("  Splined Shaft Insertion — Newton Simulation")
        print(f"  Insert speed  : {INSERT_SPEED * 1000:.0f} mm/s")
        print(f"  Shaft start Z : {shaft_start_z * 1000:.1f} mm")
        print(f"  Hub bore span : ±{hub_half_z * 1000:.0f} mm")
        print(f"  Travel        : {(shaft_start_z - shaft_end_z) * 1000:.0f} mm")
        print("=" * 55)
        print("  Pause / Resume / Reset → ImGui side panel (H)")
        print("  Frame camera           → F")
        print("=" * 55 + "\n")

    # ── ImGui panel ──────────────────────────────────────────────────────────

    def _controls_ui(self, imgui) -> None:
        imgui.separator()
        imgui.text("Insertion")
        imgui.separator()
        imgui.spacing()

        total   = self._shaft_start_z - self._shaft_end_z
        done    = self._shaft_start_z - self._shaft_z
        pct     = 100.0 * done / total if total > 0 else 0.0
        depth_m = self._hub_half_z - (self._shaft_z + self._shaft_half_z)

        status = ("paused"    if self._paused
                  else "done" if not self._inserting
                  else "inserting")
        imgui.text(f"Status   : {status}")
        imgui.text(f"Shaft Z  : {self._shaft_z * 1000:.1f} mm")
        imgui.text(f"Progress : {pct:.0f}%")
        imgui.text(f"Depth    : {max(depth_m, 0.0) * 1000:.1f} mm into bore")
        imgui.spacing()

        label = "Resume" if self._paused else "Pause"
        if imgui.button(label):
            self._paused = not self._paused
        imgui.same_line()
        if imgui.button("Reset"):
            self._reset()

    # ── Simulation ───────────────────────────────────────────────────────────

    def _reset(self) -> None:
        """Teleport shaft back to start position with zero velocity."""
        self.state_0.body_q.assign(self._initial_body_q)
        self.state_0.body_qd.assign(
            np.zeros(self.state_0.body_qd.numpy().shape, dtype=np.float32)
        )
        self._shaft_z   = self._shaft_start_z
        self._inserting = True
        self._paused    = False

    def simulate(self) -> None:
        """One physics frame: collide once, then SIM_SUBSTEPS solver steps."""
        self.collision_pipeline.collide(self.state_0, self.contacts)

        vel_z = -INSERT_SPEED if self._inserting else 0.0

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                kernel=_drive_shaft,
                dim=1,
                inputs=(
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self._shaft_body,
                    float(self._shaft_z),
                    float(vel_z),
                ),
                device=self.model.device,
            )
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        """Advance one rendered frame: move shaft target, then physics."""
        if not self._paused:
            if self._inserting:
                self._shaft_z -= INSERT_SPEED * self.frame_dt
                if self._shaft_z <= self._shaft_end_z:
                    self._shaft_z   = self._shaft_end_z
                    self._inserting = False
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        """Render bodies and contact arrows."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example, args)
