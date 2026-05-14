"""
Generate splined shaft + hub geometry as USD files.

DIN 5480 involute spline profile (30° pressure angle).
Exports shaft.usd and hub.usd into the same directory.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import trimesh
import trimesh.creation
from shapely.geometry import Polygon
from pxr import Usd, UsdGeom, Gf, Vt

HERE = pathlib.Path(__file__).parent


# ---------------------------------------------------------------------------
# Parameters  (all SI — metres)
# ---------------------------------------------------------------------------
Z      = 16          # number of teeth
MODULE = 0.002       # m  (2 mm module → 32 mm pitch diameter)
ALPHA  = math.radians(30.0)   # DIN 5480 pressure angle
LENGTH = 0.040       # shaft / spline engagement length (40 mm)
HUB_OD = 0.052       # hub outer diameter (52 mm)
GAP    = 0.0003      # diametral clearance shaft→hub (0.3 mm)

N_INV  = 30          # involute sample points per flank
N_ARC  = 12          # arc sample points


# ---------------------------------------------------------------------------
# 2-D involute spline profile
# ---------------------------------------------------------------------------

def _involute_xy(rb: float, t: float) -> tuple[float, float]:
    return rb * (math.cos(t) + t * math.sin(t)), rb * (math.sin(t) - t * math.cos(t))


def _t_at_r(rb: float, r: float) -> float:
    ratio = r / rb
    return math.sqrt(max(ratio * ratio - 1.0, 0.0))


def _arc_pts(r: float, a_start: float, a_end: float, n: int) -> np.ndarray:
    angles = np.linspace(a_start, a_end, n)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def external_spline_profile(z=Z, module=MODULE, alpha=ALPHA, n_inv=N_INV, n_arc=N_ARC) -> np.ndarray:
    """2-D closed polygon of an external (shaft) spline cross-section."""
    rp = module * z / 2.0
    rb = rp * math.cos(alpha)
    ra = rp + module                         # tip radius
    rf = max(rp - 1.25 * module, rb * 0.98) # root radius (never inside base circle)

    pitch = 2.0 * math.pi / z

    # Involute angle offset so the tooth is centred on θ=0
    t_p = _t_at_r(rb, rp)
    inv_alpha = math.tan(alpha) - alpha      # involute function at pressure angle
    half_tooth_pitch = math.pi / (2.0 * z)
    # Right flank: involute angle at pitch radius
    phi_p = math.atan2(
        rb * (math.sin(t_p) - t_p * math.cos(t_p)),
        rb * (math.cos(t_p) + t_p * math.sin(t_p))
    )
    offset = half_tooth_pitch - phi_p        # rotational offset to centre tooth

    t_root = _t_at_r(rb, rf)
    t_tip  = _t_at_r(rb, ra)

    all_pts: list[list[float]] = []

    for i in range(z):
        base = i * pitch

        # --- right flank (involute, inner→outer) ---
        for t in np.linspace(t_root, t_tip, n_inv):
            x, y = _involute_xy(rb, t)
            r = math.hypot(x, y)
            a = math.atan2(y, x) + base + offset
            all_pts.append([r * math.cos(a), r * math.sin(a)])

        # --- tip arc ---
        a_right_tip = math.atan2(*reversed(_involute_xy(rb, t_tip))) + base + offset
        a_left_tip  = base - (math.atan2(*reversed(_involute_xy(rb, t_tip))) + offset)
        if a_left_tip < a_right_tip:
            a_left_tip += 2.0 * math.pi
        for pt in _arc_pts(ra, a_right_tip, a_left_tip, n_arc)[1:]:
            all_pts.append(pt.tolist())

        # --- left flank (involute mirror, outer→inner) ---
        for t in np.linspace(t_tip, t_root, n_inv):
            x, y = _involute_xy(rb, t)
            r = math.hypot(x, y)
            a = base - (math.atan2(y, x) + offset)
            all_pts.append([r * math.cos(a), r * math.sin(a)])

        # --- root arc to next tooth ---
        a_this_root = base - (math.atan2(*reversed(_involute_xy(rb, t_root))) + offset)
        a_next_root = (i + 1) * pitch + math.atan2(*reversed(_involute_xy(rb, t_root))) + offset
        if a_next_root < a_this_root:
            a_next_root += 2.0 * math.pi
        for pt in _arc_pts(rf, a_this_root, a_next_root, n_arc)[1:]:
            all_pts.append(pt.tolist())

    return np.array(all_pts)


def hub_ring_polygon(
    shaft_profile: np.ndarray,
    hub_od: float = HUB_OD,
    gap: float = GAP,
    n_outer: int = 128,
) -> Polygon:
    """
    Build hub cross-section as: outer_circle MINUS enlarged_shaft_profile.
    The shaft is offset outward by gap/2 to create running clearance.
    """
    from shapely.geometry import LinearRing
    shaft_poly = Polygon(shaft_profile)
    if not shaft_poly.is_valid:
        shaft_poly = shaft_poly.buffer(0)
    # Expand shaft profile by gap/2 on all sides (clearance fit)
    pocket = shaft_poly.buffer(gap / 2.0, resolution=32)
    outer_pts = _arc_pts(hub_od / 2.0, 0.0, 2.0 * math.pi, n_outer)
    outer_ring = Polygon(outer_pts)
    ring = outer_ring.difference(pocket)
    if not ring.is_valid:
        ring = ring.buffer(0)
    return ring


# ---------------------------------------------------------------------------
# 3-D mesh via trimesh extrusion
# ---------------------------------------------------------------------------

def profile_to_mesh(profile_2d: np.ndarray, length: float) -> trimesh.Trimesh:
    poly = Polygon(profile_2d)
    if not poly.is_valid:
        poly = poly.buffer(0)
    mesh = trimesh.creation.extrude_polygon(poly, height=length)
    mesh.apply_translation([0.0, 0.0, -length / 2.0])
    return mesh


def ring_to_mesh(ring: Polygon, length: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.extrude_polygon(ring, height=length)
    mesh.apply_translation([0.0, 0.0, -length / 2.0])
    return mesh


# ---------------------------------------------------------------------------
# USD export
# ---------------------------------------------------------------------------

def mesh_to_usd(mesh: trimesh.Trimesh, out_path: pathlib.Path, prim_name: str) -> None:
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    xform = UsdGeom.Xform.Define(stage, "/World")
    mesh_prim = UsdGeom.Mesh.Define(stage, f"/World/{prim_name}")

    verts = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    face_counts = [3] * len(faces)
    face_indices = [idx for tri in faces for idx in tri]

    mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts]))
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
    mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_indices))
    mesh_prim.GetSubdivisionSchemeAttr().Set("none")

    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.Save()
    print(f"  Saved {out_path}  ({len(verts)} verts, {len(faces)} tris)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating splined shaft geometry...")
    shaft_profile = external_spline_profile()
    shaft_mesh    = profile_to_mesh(shaft_profile, LENGTH)
    shaft_path    = HERE / "shaft.usd"
    mesh_to_usd(shaft_mesh, shaft_path, "Shaft")
    shaft_mesh.export(str(HERE / "shaft.glb"))
    print(f"  Saved {HERE / 'shaft.glb'}")

    print("Generating internal spline hub geometry...")
    hub_ring = hub_ring_polygon(shaft_profile)
    hub_mesh = ring_to_mesh(hub_ring, LENGTH * 1.5)
    hub_path = HERE / "hub.usd"
    mesh_to_usd(hub_mesh, hub_path, "Hub")
    hub_mesh.export(str(HERE / "hub.glb"))
    print(f"  Saved {HERE / 'hub.glb'}")

    print("\nDone.")
    print(f"  Shaft tip radius : {MODULE * Z / 2 + MODULE:.4f} m")
    print(f"  Hub outer radius : {HUB_OD / 2:.4f} m")
    print(f"  Shaft bounds     : {shaft_mesh.bounds}")
    print(f"  Hub   bounds     : {hub_mesh.bounds}")
    print(f"  Shaft watertight : {shaft_mesh.is_watertight}")
    print(f"  Hub   watertight : {hub_mesh.is_watertight}")
