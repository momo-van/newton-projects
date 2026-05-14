# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Inspect phase 1 geometry in Rerun.

Shows the initial assembly configuration:
  - Hub at origin
  - Shaft rotated half a tooth pitch (misaligned) and positioned 30 mm above
    the hub entrance -- the starting pose for the kinematic search phase

This is the "operator holds shaft above hub, starting to rotate" moment.

Run
---
::

    python view_geometry.py
"""

from __future__ import annotations

import math
import pathlib
import subprocess
import sys

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh
import trimesh.transformations as tr

HERE = pathlib.Path(__file__).parent

RERUN_EXE = (
    r"C:\Users\mmohajerani\AppData\Local\Packages"
    r"\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0"
    r"\LocalCache\local-packages\Python313\Scripts\rerun.exe"
)

RRD_PATH = HERE / "phase1_geometry.rrd"


# ── Rerun helpers ─────────────────────────────────────────────────────────────

def log_solid(entity: str, mesh: trimesh.Trimesh, color: tuple) -> None:
    rr.log(
        entity,
        rr.Mesh3D(
            vertex_positions=mesh.vertices.astype(np.float32),
            triangle_indices=mesh.faces,
            vertex_normals=mesh.vertex_normals.astype(np.float32),
            vertex_colors=np.tile(color, (len(mesh.vertices), 1)).astype(np.uint8),
        ),
        static=True,
    )


def log_wireframe(entity: str, mesh: trimesh.Trimesh, color: tuple) -> None:
    strips = mesh.vertices.astype(np.float32)[mesh.edges_unique]
    rr.log(
        entity,
        rr.LineStrips3D(strips=strips, colors=[color] * len(strips)),
        static=True,
    )


def log_part(base: str, mesh: trimesh.Trimesh,
             solid_color: tuple, wire_color: tuple) -> None:
    log_solid(    f"{base}/solid",     mesh, solid_color)
    log_wireframe(f"{base}/wireframe", mesh, wire_color)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.path.insert(0, str(HERE))
    from gen_geometry import (
        straight_key_profile,
        hub_ring_polygon,
        profile_to_mesh,
        ring_to_mesh,
        Z, D_MAJOR, D_MINOR, KEY_WIDTH, LENGTH, HUB_OD, HUB_LEN_FACTOR,
    )

    print("Building meshes ...")
    shaft_profile = straight_key_profile()
    hub_ring      = hub_ring_polygon(shaft_profile)
    shaft_mesh    = profile_to_mesh(shaft_profile, LENGTH)
    hub_mesh      = ring_to_mesh(hub_ring, LENGTH * HUB_LEN_FACTOR)

    hub_len   = float(hub_mesh.bounds[1][2] - hub_mesh.bounds[0][2])
    shaft_len = float(shaft_mesh.bounds[1][2] - shaft_mesh.bounds[0][2])

    # Half-pitch rotation: keys land on hub lands (fully blocked = initial pose)
    half_pitch_deg = 180.0 / Z

    shaft = shaft_mesh.copy()
    rot   = tr.rotation_matrix(math.radians(half_pitch_deg), [0, 0, 1])
    shaft.apply_transform(rot)

    # Position: shaft bottom face sits 30 mm above hub top face
    shaft_z = hub_len / 2.0 + shaft_len / 2.0 + 0.030
    shaft.apply_translation([0.0, 0.0, shaft_z])

    # ── Rerun ────────────────────────────────────────────────────────────────
    rr.init("phase1_geometry")
    rr.save(str(RRD_PATH))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(name="Phase 1 -- Spline Geometry (initial pose)", origin="/world"),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)

    # Hub  (steel blue)
    log_part("world/hub", hub_mesh,
             solid_color=(150, 185, 215, 200),
             wire_color =(40,  110, 200, 255))

    # Shaft (warm steel -- misaligned initial pose)
    log_part("world/shaft", shaft,
             solid_color=(210, 175, 120, 215),
             wire_color =(180,  90,  20, 255))

    # Axes
    arrow = 0.035
    for lbl, vec, col in [
        ("X", [arrow, 0, 0], [220, 50,  50,  255]),
        ("Y", [0, arrow, 0], [50,  200, 50,  255]),
        ("Z", [0, 0, arrow], [50,  50,  220, 255]),
    ]:
        rr.log(
            f"world/axes/{lbl}",
            rr.Arrows3D(origins=[[0, 0, 0]], vectors=[vec],
                        colors=[col], labels=[lbl]),
            static=True,
        )

    print(f"Saved -> {RRD_PATH}")
    print(f"  z={Z}  D_maj={D_MAJOR*1000:.0f} mm  D_min={D_MINOR*1000:.0f} mm  key_w={KEY_WIDTH*1000:.0f} mm")
    print(f"  Shaft : {len(shaft_mesh.vertices):,} verts  L={shaft_len*1000:.0f} mm  watertight={shaft_mesh.is_watertight}")
    print(f"  Hub   : {len(hub_mesh.vertices):,} verts  L={hub_len*1000:.0f} mm    watertight={hub_mesh.is_watertight}")
    print(f"  Shaft Z offset : {shaft_z*1000:.1f} mm  (30 mm clearance above hub)")
    print(f"  Rotation       : {half_pitch_deg:.1f} deg (half pitch -- fully misaligned)")
    print("Launching Rerun ...")

    subprocess.Popen([RERUN_EXE, str(RRD_PATH)])


if __name__ == "__main__":
    main()
