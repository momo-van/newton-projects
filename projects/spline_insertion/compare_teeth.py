"""
Compare DIN 5480 spline cross-sections for z=6, z=8, z=16 side-by-side in Rerun.
Run:  python compare_teeth.py
"""
from __future__ import annotations

import math
import pathlib
import subprocess

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh
import trimesh.creation
from shapely.geometry import Polygon

from gen_geometry import (
    external_spline_profile,
    hub_ring_polygon,
    profile_to_mesh,
    ring_to_mesh,
    MODULE,
    ALPHA,
    HUB_OD,
    GAP,
    LENGTH,
)

HERE = pathlib.Path(__file__).parent

RERUN_EXE = (
    r"C:\Users\mmohajerani\AppData\Local\Packages"
    r"\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0"
    r"\LocalCache\local-packages\Python313\Scripts\rerun.exe"
)

VARIANTS = [
    {"z": 6,  "label": "z=6  (DIN min)",    "x_offset": -0.10},
    {"z": 8,  "label": "z=8  (smallest practical)", "x_offset":  0.00},
    {"z": 16, "label": "z=16 (current sim)", "x_offset":  0.10},
]

SHAFT_COLORS     = [(220, 140,  60, 210), (180, 210, 100, 210), (210, 170, 120, 210)]
SHAFT_WF_COLORS  = [(200,  80,  20, 255), (100, 180,  30, 255), (200, 100,  30, 255)]
HUB_COLORS       = [(140, 175, 210, 170), (140, 200, 175, 170), (160, 190, 210, 170)]
HUB_WF_COLORS    = [( 30, 100, 200, 255), ( 30, 170, 140, 255), ( 40, 120, 200, 255)]


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
    edges  = mesh.edges_unique
    verts  = mesh.vertices.astype(np.float32)
    strips = verts[edges]
    rr.log(
        entity,
        rr.LineStrips3D(strips=strips, colors=[color] * len(strips)),
        static=True,
    )


rr.init("spline_compare")
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

blueprint = rrb.Blueprint(
    rrb.Spatial3DView(name="Spline Tooth Comparison  z=6 | z=8 | z=16", origin="/world"),
    collapse_panels=False,
)
rr.send_blueprint(blueprint)

for i, v in enumerate(VARIANTS):
    z   = v["z"]
    lbl = v["label"]
    xo  = v["x_offset"]

    print(f"Generating {lbl} ...")

    # Scale hub OD to keep wall thickness reasonable for small z
    hub_od = max(HUB_OD, MODULE * z * 2.0 + 0.024)

    shaft_profile = external_spline_profile(z=z)
    shaft_mesh    = profile_to_mesh(shaft_profile, LENGTH)
    hub_ring      = hub_ring_polygon(shaft_profile, hub_od=hub_od)
    hub_mesh      = ring_to_mesh(hub_ring, LENGTH * 1.5)

    hub_half_z   = LENGTH * 1.5 / 2.0
    shaft_half_z = LENGTH / 2.0
    shaft_z      = hub_half_z + shaft_half_z + 0.030   # 30 mm clearance above hub top

    base = f"world/v{z}"

    # Hub at x_offset, centred at z=0
    rr.log(f"{base}/hub",   rr.Transform3D(translation=[xo, 0.0, 0.0]), static=True)
    log_solid(    f"{base}/hub/solid",     hub_mesh,   HUB_COLORS[i])
    log_wireframe(f"{base}/hub/wireframe", hub_mesh,   HUB_WF_COLORS[i])

    # Shaft 30 mm above hub
    rr.log(f"{base}/shaft", rr.Transform3D(translation=[xo, 0.0, shaft_z]), static=True)
    log_solid(    f"{base}/shaft/solid",     shaft_mesh, SHAFT_COLORS[i])
    log_wireframe(f"{base}/shaft/wireframe", shaft_mesh, SHAFT_WF_COLORS[i])

    # Label as a small axis arrow set at the hub centre-top
    rr.log(
        f"{base}/label",
        rr.Arrows3D(
            origins=[[xo, 0.0, hub_half_z + 0.005]],
            vectors=[[0.0, 0.0, 0.008]],
            colors=[[255, 255, 100, 255]],
            labels=[lbl],
        ),
        static=True,
    )

    print(f"  shaft watertight={shaft_mesh.is_watertight}  "
          f"hub watertight={hub_mesh.is_watertight}  "
          f"hub_od={hub_od*1000:.0f}mm")

RRD_PATH = HERE / "spline_compare.rrd"
rr.save(str(RRD_PATH))
print(f"\nSaved -> {RRD_PATH}")

subprocess.Popen([RERUN_EXE, str(RRD_PATH)])
print("Rerun launched.")
