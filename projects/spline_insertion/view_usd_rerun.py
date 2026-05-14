"""
View splined shaft + hub geometry in Rerun with solid and wireframe layers.

Each part is logged twice:
  world/<part>/solid      — shaded Mesh3D  (semi-transparent)
  world/<part>/wireframe  — LineStrips3D from unique mesh edges

Both layers are independently toggleable in the Rerun entity panel.

Run
---
    python view_usd_rerun.py
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import trimesh
import rerun as rr
import rerun.blueprint as rrb

HERE = pathlib.Path(__file__).parent

RERUN_EXE = (
    r"C:\Users\mmohajerani\AppData\Local\Packages"
    r"\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0"
    r"\LocalCache\local-packages\Python313\Scripts\rerun.exe"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_glb(path: pathlib.Path) -> trimesh.Trimesh:
    scene = trimesh.load(str(path), force="mesh")
    if isinstance(scene, trimesh.Scene):
        scene = trimesh.util.concatenate(list(scene.geometry.values()))
    return scene


def log_mesh_solid(
    entity: str,
    mesh: trimesh.Trimesh,
    color: tuple[int, int, int, int] = (180, 200, 220, 200),
) -> None:
    normals = mesh.vertex_normals.astype(np.float32)
    rr.log(
        entity,
        rr.Mesh3D(
            vertex_positions=mesh.vertices.astype(np.float32),
            triangle_indices=mesh.faces,
            vertex_normals=normals,
            vertex_colors=np.tile(color, (len(mesh.vertices), 1)).astype(np.uint8),
        ),
        static=True,
    )


def log_mesh_wireframe(
    entity: str,
    mesh: trimesh.Trimesh,
    color: tuple[int, int, int, int] = (60, 140, 220, 255),
) -> None:
    edges = mesh.edges_unique                          # (E, 2) unique edge indices
    verts = mesh.vertices.astype(np.float32)
    # Build list of 2-point line strips: one strip per edge
    strips = verts[edges]                              # (E, 2, 3)
    rr.log(
        entity,
        rr.LineStrips3D(
            strips=strips,
            colors=[color] * len(strips),
        ),
        static=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

shaft_glb = HERE / "shaft.glb"
hub_glb   = HERE / "hub.glb"

for p in (shaft_glb, hub_glb):
    if not p.exists():
        import sys
        sys.exit(f"ERROR: {p.name} not found. Run gen_geometry.py first.")

shaft_mesh = load_glb(shaft_glb)
hub_mesh   = load_glb(hub_glb)

RRD_PATH = HERE / "spline_geometry.rrd"

rr.init("spline_geometry")
rr.save(str(RRD_PATH))

rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

blueprint = rrb.Blueprint(
    rrb.Spatial3DView(name="Splined Shaft + Hub", origin="/world"),
    collapse_panels=False,
)
rr.send_blueprint(blueprint)

# Hub sits at origin (z: 0 -> hub_len)
hub_len   = float(hub_mesh.bounds[1][2] - hub_mesh.bounds[0][2])
shaft_len = float(shaft_mesh.bounds[1][2] - shaft_mesh.bounds[0][2])

log_mesh_solid(    "world/hub/solid",     hub_mesh,   color=(160, 190, 210, 180))
log_mesh_wireframe("world/hub/wireframe", hub_mesh,   color=(40, 120, 200, 255))

# Shaft positioned 30 mm above hub top — ready-to-insert pose
shaft_z = hub_len + 0.030
rr.log("world/shaft", rr.Transform3D(translation=[0.0, 0.0, shaft_z]), static=True)
log_mesh_solid(    "world/shaft/solid",     shaft_mesh, color=(210, 170, 120, 200))
log_mesh_wireframe("world/shaft/wireframe", shaft_mesh, color=(200, 100, 30, 255))

# Axis arrows scaled to part size
arrow_len = 0.025
for label, vec, col in [
    ("X", [arrow_len, 0, 0],         [220, 50,  50,  255]),
    ("Y", [0, arrow_len, 0],         [50,  200, 50,  255]),
    ("Z", [0, 0, arrow_len],         [50,  50,  220, 255]),
]:
    rr.log(
        f"world/axes/{label}",
        rr.Arrows3D(origins=[[0, 0, 0]], vectors=[vec], colors=[col], labels=[label]),
        static=True,
    )

print(f"\nSaved -> {RRD_PATH}")
print(f"  Hub   : {hub_glb.name}   ({len(hub_mesh.vertices):,} verts)  L={hub_len*1000:.1f}mm  OD={hub_mesh.bounds[1][0]*2000:.1f}mm")
print(f"  Shaft : {shaft_glb.name} ({len(shaft_mesh.vertices):,} verts)  L={shaft_len*1000:.1f}mm  tip_d={shaft_mesh.bounds[1][0]*2000:.1f}mm")
print(f"  Shaft-z offset: {shaft_z*1000:.1f} mm above origin (30 mm clearance above hub)")

import subprocess
subprocess.Popen([RERUN_EXE, str(RRD_PATH)])
