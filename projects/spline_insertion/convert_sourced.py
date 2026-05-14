"""
Convert downloaded STL spline pairs to GLB + USD (SI metres).

Source STL files are in millimetres from hexagon.de.
We scale ×0.001 (mm->m), centre each part on the Z-axis,
then export shaft.glb / hub.glb and shaft.usd / hub.usd
ready for Newton simulation and Rerun visualisation.

Two pairs are available:
  6-tooth  DIN ISO 14  straight-sided  (coarser, more visual)
  7-tooth  ANSI B92.1b involute 16/32  (finer, more industrial)

Set PAIR below to choose.
"""

from __future__ import annotations

import pathlib
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, Gf, Vt

HERE    = pathlib.Path(__file__).parent
SOURCED = HERE / "sourced"

# ── choose which pair to export ──────────────────────────────────────────────
PAIR = "7tooth"        # "6tooth" or "7tooth"
SCALE = 3.0            # visual scale-up (keeps proportions; 1 = true DIN size)
# ─────────────────────────────────────────────────────────────────────────────

MM_TO_M = 0.001


def stl_to_solid(stl_path: pathlib.Path, scale: float = 1.0) -> trimesh.Trimesh:
    """
    Build a watertight simulation-ready solid from a display STL:
      1. Load & scale (mm -> m)
      2. Slice at mid-height — avoids end-face degenerate triangles
      3. Keep the two largest-area section curves:
           shaft → one outer polygon (solid cross-section)
           hub   → outer circle minus inner spline bore (ring cross-section)
      4. Extrude the polygon over the original length
    The exact spline tooth profile from the STL is preserved.
    """
    from shapely.geometry import Polygon as SP
    from shapely.ops import unary_union
    from trimesh.creation import extrude_polygon

    mesh = trimesh.load(str(stl_path), force="mesh")
    mesh.apply_scale(MM_TO_M * scale)

    b = mesh.bounds
    cx = (b[0][0] + b[1][0]) / 2.0
    cy = (b[0][1] + b[1][1]) / 2.0
    mesh.apply_translation([-cx, -cy, -b[0][2]])

    z_min  = float(mesh.bounds[0][2])   # 0
    z_max  = float(mesh.bounds[1][2])
    height = z_max - z_min
    z_mid  = height / 2.0

    sec = mesh.section(plane_normal=[0, 0, 1], plane_origin=[0, 0, z_mid])
    if sec is None:
        raise RuntimeError(f"No section found for {stl_path.name}")
    p2d, _ = sec.to_2D()

    # Build candidate polygons from section entities; keep those with real area
    candidates = []
    for ent in p2d.entities:
        pts = p2d.vertices[ent.points]
        if len(pts) < 3:
            continue
        p = SP(pts)
        if not p.is_valid:
            p = p.buffer(0)
        if p.area > 1e-8:   # filter sub-mm² noise
            candidates.append(p)

    if not candidates:
        raise RuntimeError(f"No valid cross-section polygons in {stl_path.name}")

    # Sort by area descending
    candidates.sort(key=lambda p: p.area, reverse=True)

    if len(candidates) == 1:
        # Shaft: single solid cross-section
        cross = candidates[0]
    else:
        # Hub: outer ring (largest) minus inner bore (second largest that fits inside)
        outer = candidates[0]
        cross = outer
        for inner in candidates[1:]:
            if not inner.is_valid:
                inner = inner.buffer(0)
            # Subtract only if genuinely enclosed
            if outer.buffer(-1e-6).contains(inner.centroid):
                try:
                    cross = cross.difference(inner)
                except Exception:
                    pass
        if not cross.is_valid:
            cross = cross.buffer(0)

    solid = extrude_polygon(cross, height=height)
    solid.apply_translation([0.0, 0.0, z_min])
    solid.fix_normals()
    return solid


def load_and_prep(stl_path: pathlib.Path, scale: float = 1.0) -> trimesh.Trimesh:
    return stl_to_solid(stl_path, scale=scale)


def save_glb(mesh: trimesh.Trimesh, path: pathlib.Path) -> None:
    mesh.export(str(path))
    print(f"  GLB  -> {path.name}  ({len(mesh.vertices):,} verts, {len(mesh.faces):,} tris)")


def save_usd(mesh: trimesh.Trimesh, path: pathlib.Path, prim_name: str) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    mp = UsdGeom.Mesh.Define(stage, f"/World/{prim_name}")
    verts = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    mp.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts]))
    mp.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    mp.GetFaceVertexIndicesAttr().Set(Vt.IntArray([i for tri in faces for i in tri]))
    mp.GetSubdivisionSchemeAttr().Set("none")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.Save()
    print(f"  USD  -> {path.name}")


if __name__ == "__main__":
    shaft_stl = SOURCED / f"shaft_{PAIR}.stl"
    hub_stl   = SOURCED / f"hub_{PAIR}.stl"

    print(f"Loading {PAIR} pair  (scale ×{SCALE}) ...")
    shaft = load_and_prep(shaft_stl, scale=SCALE)
    hub   = load_and_prep(hub_stl,   scale=SCALE)

    shaft_len = shaft.bounds[1][2] - shaft.bounds[0][2]
    hub_len   = hub.bounds[1][2]   - hub.bounds[0][2]
    shaft_r   = max(abs(shaft.bounds[0][0]), abs(shaft.bounds[1][0]))
    hub_r     = max(abs(hub.bounds[0][0]),   abs(hub.bounds[1][0]))

    print(f"\n  Shaft : d={shaft_r*2*1000:.1f} mm  L={shaft_len*1000:.1f} mm")
    print(f"  Hub   : d={hub_r*2*1000:.1f} mm OD  L={hub_len*1000:.1f} mm")

    print("\nExporting ...")
    save_glb(shaft, HERE / "shaft.glb")
    save_usd(shaft, HERE / "shaft.usd", "Shaft")
    save_glb(hub,   HERE / "hub.glb")
    save_usd(hub,   HERE / "hub.usd",   "Hub")

    print("\nDone — shaft.glb / hub.glb / shaft.usd / hub.usd updated.")
