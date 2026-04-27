# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
RJ45 Plug-Socket Insertion — Hydroelastic Contact Demo

Demonstrates Newton's hydroelastic contact model on a realistic RJ45
connector geometry.  Hydroelastic contacts produce a full triangulated
contact patch rather than discrete contact points, enabling:

* Distributed pressure visualisation (handled by ``hydro_contact_viz.py``)
* Contact area estimation
* Dramatic contact reduction: ~23 000 hydro faces → ~205 solver contacts

File layout
-----------
rj45_hydro.py        — scene construction, physics kernels, simulation loop
hydro_contact_viz.py — reusable HydroContactViz (ImGui panel + patch rendering)

Controls
--------
  Gizmo arrows   drag the plug into the socket along the Y-axis
  Right-drag     spring-pick any body
  H              toggle the ImGui side-panel
  Space          pause / resume
  F              frame camera on the model

Run
---
::

    python rj45_hydro.py
"""

from __future__ import annotations

import dataclasses

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.usd
import newton.utils
from newton.geometry import HydroelasticSDF
from newton.math import quat_between_vectors_robust
from newton.solvers import SolverVBD

from hydro_contact_viz import HydroContactViz

# ---------------------------------------------------------------------------
# Shape / physics constants
# ---------------------------------------------------------------------------

#: Hydroelastic stiffness (Pa/m) — controls contact-surface compliance.
SHAPE_KH: float = 5.0e7

#: Base shape config used for cable segments and the ground plane.
SHAPE_CFG_BASE = newton.ModelBuilder.ShapeConfig(
    mu=0.1,
    ke=1.0e6,
    kd=1.0e3,
    gap=0.002,
    density=1.0e6,
    mu_torsional=0.0,
    mu_rolling=0.0,
)

#: Hydroelastic shape config used for the socket, plug body, and latch.
SHAPE_CFG_HYDRO = newton.ModelBuilder.ShapeConfig(
    mu=0.1,
    kh=SHAPE_KH,
    ke=1.0e6,
    kd=1.0e3,
    gap=0.002,
    density=1.0e6,
    mu_torsional=0.0,
    mu_rolling=0.0,
    is_hydroelastic=True,
)

MESH_SDF_MAX_RESOLUTION    = 128
MESH_SDF_NARROW_BAND_RANGE = (-2.0 * SHAPE_CFG_BASE.gap, 2.0 * SHAPE_CFG_BASE.gap)

PLUG_Y_OFFSET         = -0.025

CABLE_RADIUS          = 0.00325
CABLE_KINEMATIC_COUNT = 4
CABLE_KE              = 1.0e8
CABLE_KD              = 1.0e-3
CABLE_MU              = 2.0

LATCH_LIMIT_LOWER = -0.2
LATCH_LIMIT_UPPER =  0.3
LATCH_SPRING_KE   =  0.15
LATCH_SPRING_KD   =  0.2
LATCH_LIMIT_KD    =  1.0e-4


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _apply_gizmo_force(
    body_q:      wp.array(dtype=wp.transform),
    body_qd:     wp.array(dtype=wp.spatial_vector),
    body_f:      wp.array(dtype=wp.spatial_vector),
    body_mass:   wp.array(dtype=float),
    pick_target: wp.array(dtype=wp.vec3),
    stiffness:   float,
    damping:     float,
    pick_body:   wp.array(dtype=int),
    plug_idx:    int,
    latch_idx:   int,
    gravity:     wp.vec3,
):
    """Drive plug toward gizmo target; damp latch to follow; cancel gravity."""
    anti_g0 = -gravity * body_mass[plug_idx]
    anti_g1 = -gravity * body_mass[latch_idx]
    wp.atomic_add(body_f, plug_idx,  wp.spatial_vector(anti_g0, wp.vec3(0.0)))
    wp.atomic_add(body_f, latch_idx, wp.spatial_vector(anti_g1, wp.vec3(0.0)))

    target      = pick_target[0]
    picked_body = pick_body[0]

    if picked_body >= 0:
        if picked_body != plug_idx:
            vel0  = wp.spatial_top(body_qd[plug_idx])
            mass0 = body_mass[plug_idx]
            wp.atomic_add(body_f, plug_idx,
                          wp.spatial_vector(-(10.0 + mass0) * damping * vel0, wp.vec3(0.0)))
        if picked_body != latch_idx:
            vel1  = wp.spatial_top(body_qd[latch_idx])
            mass1 = body_mass[latch_idx]
            wp.atomic_add(body_f, latch_idx,
                          wp.spatial_vector(-(10.0 + mass1) * damping * vel1, wp.vec3(0.0)))
        return

    pos0  = wp.transform_get_translation(body_q[plug_idx])
    vel0  = wp.spatial_top(body_qd[plug_idx])
    mass0 = body_mass[plug_idx]
    mult0 = 10.0 + mass0
    f0    = mult0 * (stiffness * (target - pos0) - damping * vel0)
    wp.atomic_add(body_f, plug_idx, wp.spatial_vector(f0, wp.vec3(0.0)))

    vel1         = wp.spatial_top(body_qd[latch_idx])
    mass1        = body_mass[latch_idx]
    spring_accel = (target - pos0) * (mult0 * stiffness / mass0)
    f1           = spring_accel * mass1 - vel1 * ((10.0 + mass1) * damping)
    wp.atomic_add(body_f, latch_idx, wp.spatial_vector(f1, wp.vec3(0.0)))


@wp.kernel
def _sync_cable_anchors(
    body_q:           wp.array(dtype=wp.transform),
    body_qd:          wp.array(dtype=wp.spatial_vector),
    plug_idx:         int,
    anchor_indices:   wp.array(dtype=int),
    anchor_offsets:   wp.array(dtype=wp.vec3),
    anchor_rotations: wp.array(dtype=wp.quat),
):
    """Lock the first N cable segments to the plug body each substep."""
    tid          = wp.tid()
    plug_tf      = body_q[plug_idx]
    plug_pos     = wp.transform_get_translation(plug_tf)
    plug_rot     = wp.transform_get_rotation(plug_tf)
    idx          = anchor_indices[tid]
    anchor_world = plug_pos + wp.quat_rotate(plug_rot, anchor_offsets[tid])
    cable_rot    = wp.normalize(wp.mul(plug_rot, anchor_rotations[tid]))
    body_q[idx]  = wp.transform(anchor_world, cable_rot)
    body_qd[idx] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _align_cable_orientations(
    body_q:         wp.array(dtype=wp.transform),
    cable_body_idx: wp.array(dtype=int),
    cable_next_idx: wp.array(dtype=int),
):
    """Align each cable segment's Z-axis toward the next segment."""
    tid      = wp.tid()
    bi       = cable_body_idx[tid]
    bi_next  = cable_next_idx[tid]
    tf       = body_q[bi]
    pos      = wp.transform_get_translation(tf)
    rot      = wp.transform_get_rotation(tf)
    next_pos = wp.transform_get_translation(body_q[bi_next])
    seg      = next_pos - pos
    seg_len  = wp.length(seg)
    if seg_len < 1.0e-10:
        return
    d         = seg / seg_len
    z_current = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    q_swing   = quat_between_vectors_robust(z_current, d)
    rot_new   = wp.normalize(wp.mul(q_swing, rot))
    body_q[bi] = wp.transform(pos, rot_new)


# ---------------------------------------------------------------------------
# USD helpers
# ---------------------------------------------------------------------------

def _load_mesh(stage, prim_path: str) -> tuple[newton.Mesh, wp.vec3]:
    """Load a USD mesh prim and build its SDF."""
    prim     = stage.GetPrimAtPath(prim_path)
    usd_mesh = newton.usd.get_mesh(prim, load_normals=True)
    tf       = newton.usd.get_transform(prim, local=False)
    prim_pos = wp.transform_get_translation(tf)
    vertices = np.array(usd_mesh.vertices, dtype=np.float32)
    indices  = np.array(usd_mesh.indices,  dtype=np.int32)
    normals  = (np.array(usd_mesh.normals, dtype=np.float32)
                if usd_mesh.normals is not None else None)
    mesh = newton.Mesh(vertices, indices, normals=normals)
    mesh.build_sdf(
        max_resolution=MESH_SDF_MAX_RESOLUTION,
        narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
        margin=SHAPE_CFG_BASE.gap,
    )
    return mesh, prim_pos


def _load_cable_centerline(stage) -> tuple[wp.vec3, ...]:
    """Return world-space cable control points with the plug Y-offset applied."""
    prim       = stage.GetPrimAtPath("/World/CableCurve")
    all_points = UsdGeom.BasisCurves(prim).GetPointsAttr().Get()
    tf         = newton.usd.get_transform(prim, local=False)
    prim_pos   = wp.transform_get_translation(tf)
    return tuple(
        wp.vec3(
            float(p[0]) + float(prim_pos[0]),
            float(p[1]) + float(prim_pos[1]) + PLUG_Y_OFFSET,
            float(p[2]) + float(prim_pos[2]),
        )
        for p in all_points
    )


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

class Example:
    """
    RJ45 plug-insertion demo with hydroelastic contacts.

    Step loop
    ---------
    1. Sync gizmo / pick state
    2. ``simulate()`` — collide once, then VBD substeps
    3. ``viz.update()`` — read contact surface metrics from GPU
    4. ``render()`` — log bodies, contacts, and the coloured patch via ``viz``
    """

    def __init__(self, viewer) -> None:
        self.fps          = 60
        self.frame_dt     = 1.0 / self.fps
        self.sim_time     = 0.0
        self.sim_substeps = 6
        self.sim_dt       = self.frame_dt / self.sim_substeps

        self.viewer         = viewer
        self.pick_stiffness = 50.0
        self.pick_damping   = 10.0

        # ---- Load USD scene -------------------------------------------------
        usd_path = newton.examples.get_asset("rj45_plug.usd")
        stage    = Usd.Stage.Open(usd_path)

        socket_mesh, sc = _load_mesh(stage, "/World/Socket")
        plug_mesh,   pc = _load_mesh(stage, "/World/Plug")
        latch_mesh,  lc = _load_mesh(stage, "/World/Latch")

        # ---- Build Newton model ---------------------------------------------
        builder = newton.ModelBuilder(gravity=-9.81)
        builder.rigid_gap        = 0.005
        builder.default_shape_cfg = SHAPE_CFG_BASE
        builder.add_ground_plane()

        # Socket (static, body = -1) — hydroelastic
        self._socket_shape = builder.add_shape_mesh(
            -1,
            mesh=socket_mesh,
            xform=wp.transform(sc, wp.quat_identity()),
            cfg=SHAPE_CFG_HYDRO,
            label="socket",
        )

        # Plug (dynamic) — hydroelastic
        plug_pos        = wp.vec3(pc[0], pc[1] + PLUG_Y_OFFSET, pc[2])
        self._plug_body = builder.add_link(
            xform=wp.transform(plug_pos, wp.quat_identity()),
            label="plug",
        )
        plug_shape = builder.add_shape_mesh(
            self._plug_body, mesh=plug_mesh, cfg=SHAPE_CFG_HYDRO,
        )

        # Latch (dynamic, revolute off plug) — hydroelastic
        latch_pos        = wp.vec3(lc[0], lc[1] + PLUG_Y_OFFSET, lc[2])
        self._latch_body = builder.add_link(
            xform=wp.transform(latch_pos, wp.quat_identity()),
            label="latch",
        )
        latch_shape = builder.add_shape_mesh(
            self._latch_body, mesh=latch_mesh, cfg=SHAPE_CFG_HYDRO,
        )

        connector_shapes = (self._socket_shape, plug_shape, latch_shape)

        # D6 joint: world → plug (free translation, locked rotation)
        JointDof = newton.ModelBuilder.JointDofConfig
        d6_joint = builder.add_joint_d6(
            parent=-1,
            child=self._plug_body,
            linear_axes=(
                JointDof(axis=(1.0, 0.0, 0.0)),
                JointDof(axis=(0.0, 1.0, 0.0)),
                JointDof(axis=(0.0, 0.0, 1.0)),
            ),
            angular_axes=None,
            parent_xform=wp.transform(plug_pos, wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )

        # Revolute joint: plug → latch
        rev_joint = builder.add_joint_revolute(
            parent=self._plug_body,
            child=self._latch_body,
            axis=(-1.0, 0.0, 0.0),
            parent_xform=wp.transform(lc - pc, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            target_ke=LATCH_SPRING_KE,
            target_kd=LATCH_SPRING_KD,
            limit_lower=LATCH_LIMIT_LOWER,
            limit_upper=LATCH_LIMIT_UPPER,
            limit_kd=LATCH_LIMIT_KD,
            collision_filter_parent=True,
        )

        builder.add_articulation([d6_joint, rev_joint])

        # Cable (non-hydroelastic)
        cable_points = _load_cable_centerline(stage)
        cable_quats  = newton.utils.create_parallel_transport_cable_quaternions(cable_points)
        rod_bodies, _ = builder.add_rod(
            positions=cable_points,
            quaternions=cable_quats,
            radius=CABLE_RADIUS,
            cfg=dataclasses.replace(
                builder.default_shape_cfg,
                ke=CABLE_KE, kd=CABLE_KD, mu=CABLE_MU,
            ),
            bend_stiffness=1.0e-1,
            bend_damping=1.0e-1,
            stretch_stiffness=1.0e9,
            stretch_damping=1.0e-1,
            label="cable",
        )

        for body_idx in rod_bodies[:CABLE_KINEMATIC_COUNT]:
            for cable_shape in builder.body_shapes[body_idx]:
                for conn_shape in connector_shapes:
                    builder.add_shape_collision_filter_pair(cable_shape, conn_shape)

        for idx in (*rod_bodies[:CABLE_KINEMATIC_COUNT], rod_bodies[-1]):
            builder.body_mass[idx]        = 0.0
            builder.body_inv_mass[idx]    = 0.0
            builder.body_inertia[idx]     = wp.mat33(0.0)
            builder.body_inv_inertia[idx] = wp.mat33(0.0)

        anchor_body_ids = tuple(rod_bodies[:CABLE_KINEMATIC_COUNT])
        anchor_offsets  = tuple(cable_points[i] - plug_pos for i in range(CABLE_KINEMATIC_COUNT))
        anchor_rots     = tuple(cable_quats[i] for i in range(CABLE_KINEMATIC_COUNT))

        builder.color()
        self.model = builder.finalize()

        self._cable_anchor_indices   = wp.array(anchor_body_ids, dtype=int,     device=self.model.device)
        self._cable_anchor_offsets   = wp.array(anchor_offsets,  dtype=wp.vec3, device=self.model.device)
        self._cable_anchor_rotations = wp.array(anchor_rots,     dtype=wp.quat, device=self.model.device)

        align_start  = max(CABLE_KINEMATIC_COUNT - 1, 0)
        align_bodies = tuple(rod_bodies[align_start:-1])
        align_next   = tuple(rod_bodies[align_start + 1:])
        self._cable_align_indices = wp.array(align_bodies, dtype=int, device=self.model.device)
        self._cable_align_next    = wp.array(align_next,   dtype=int, device=self.model.device)
        self._cable_align_count   = len(align_bodies)

        # ---- Hydroelastic collision pipeline --------------------------------
        sdf_hydro_cfg = HydroelasticSDF.Config(
            output_contact_surface=True,   # required for patch visualisation
            reduce_contacts=True,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            broad_phase="explicit",
            sdf_hydroelastic_config=sdf_hydro_cfg,
        )

        # ---- Newton state ---------------------------------------------------
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
            rigid_body_contact_buffer_size=256,
        )

        # ---- Viewer setup ---------------------------------------------------
        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = True
        self.viewer.set_camera(
            pos=wp.vec3(0.125, plug_pos[1] - 0.025, 0.03),
            pitch=-10.0,
            yaw=180.0,
        )
        if hasattr(self.viewer, "_cam_speed"):
            self.viewer._cam_speed = 0.2

        # ---- Visualizer (ImGui panel + contact patch) -----------------------
        self.viz = HydroContactViz(self.collision_pipeline, self.model, viewer)

        # Register simulation control buttons in a separate UI callback
        self.viewer.register_ui_callback(self._controls_ui, position="side")

        # ---- Gizmo / pick ---------------------------------------------------
        self._rest_pos    = plug_pos
        self.gizmo_tf     = wp.transform(plug_pos, wp.quat_identity())
        self._pick_body   = wp.array([-1], dtype=int,    device=self.model.device)
        self._pick_target = wp.zeros(1,   dtype=wp.vec3, device=self.model.device)
        self._gravity     = wp.vec3(*self.model.gravity.numpy()[0])

        # No CUDA graph — contact surface read-back requires CPU sync
        self.graph = None

        hydro_flags = self.model.shape_flags.numpy()
        n_hydro = int((hydro_flags & newton.ShapeFlags.HYDROELASTIC).astype(bool).sum())
        print("\n" + "=" * 60)
        print("  Newton RJ45 Hydroelastic Contact Demo")
        print(f"  Hydroelastic shapes : {n_hydro} / {self.model.shape_count}")
        print("=" * 60)
        print("  Drag gizmo arrows to insert the plug into the socket.")
        print("  The contact patch iso-surface shows pressure distribution")
        print("  across the mating surfaces in real time.")
        print("  Press H to toggle the ImGui panel.")
        print("=" * 60 + "\n")

    # -----------------------------------------------------------------------
    # UI callbacks
    # -----------------------------------------------------------------------

    def _controls_ui(self, imgui) -> None:
        """Simulation control buttons (separate from the viz panel)."""
        imgui.separator()
        imgui.text("Controls")
        imgui.separator()
        imgui.spacing()

        if imgui.button("Reset position"):
            self._reset_plug()

        imgui.same_line()

        if imgui.button("Push into socket"):
            self.gizmo_tf = wp.transform(
                wp.vec3(
                    self._rest_pos[0],
                    self._rest_pos[1] + 0.06,
                    self._rest_pos[2],
                ),
                wp.quat_identity(),
            )

    # -----------------------------------------------------------------------
    # Simulation
    # -----------------------------------------------------------------------

    def _reset_plug(self) -> None:
        """Teleport plug back to start position with zero velocity."""
        self.state_0.body_q.assign(self._initial_body_q)
        zeros_qd = np.zeros(self.state_0.body_qd.numpy().shape, dtype=np.float32)
        self.state_0.body_qd.assign(zeros_qd)
        self.gizmo_tf = wp.transform(self._rest_pos, wp.quat_identity())

    def simulate(self) -> None:
        """
        Run one frame of physics.

        ``collide()`` is called once *before* the substep loop so that the
        contacts buffer is stable for the full frame.  ``viz.update()`` then
        reads the surface data after all substeps complete.
        """
        self.collision_pipeline.collide(self.state_0, self.contacts)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            wp.launch(
                kernel=_apply_gizmo_force,
                dim=1,
                inputs=(
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.model.body_mass,
                    self._pick_target,
                    self.pick_stiffness,
                    self.pick_damping,
                    self._pick_body,
                    self._plug_body,
                    self._latch_body,
                    self._gravity,
                ),
                device=self.model.device,
            )

            self.viewer.apply_forces(self.state_0)

            wp.launch(
                kernel=_sync_cable_anchors,
                dim=CABLE_KINEMATIC_COUNT,
                inputs=(
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self._plug_body,
                    self._cable_anchor_indices,
                    self._cable_anchor_offsets,
                    self._cable_anchor_rotations,
                ),
                device=self.model.device,
            )

            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

            wp.launch(
                kernel=_align_cable_orientations,
                dim=self._cable_align_count,
                inputs=(
                    self.state_0.body_q,
                    self._cable_align_indices,
                    self._cable_align_next,
                ),
                device=self.model.device,
            )

    def step(self) -> None:
        """Full frame: sync gizmo → physics → update viz metrics."""
        gp          = wp.transform_get_translation(self.gizmo_tf)
        picked_body = int(self.viewer.picking.pick_body.numpy()[0])
        self._pick_body.assign([picked_body])
        self._pick_target.assign([gp])

        self.simulate()
        self.sim_time += self.frame_dt

        # Update contact-surface metrics (throttled inside viz)
        self.viz.update(self.contacts)

        # Snap gizmo to plug when not actively dragging
        if not self.viewer.gizmo_is_using:
            plug_tf = self.state_0.body_q.numpy()[self._plug_body]
            snap_y  = plug_tf[1] if picked_body >= 0 else plug_tf[1]
            snap    = wp.vec3(self._rest_pos[0], plug_tf[1], self._rest_pos[2])
            self.gizmo_tf = wp.transform(snap, wp.quat_identity())

    def render(self) -> None:
        """Render bodies, contact arrows, and the hydroelastic patch."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_gizmo("plug", self.gizmo_tf, rotate=())
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viz.render()   # coloured iso-surface lines
        self.viewer.end_frame()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example, args)
