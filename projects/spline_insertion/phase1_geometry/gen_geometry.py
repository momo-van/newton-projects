# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Generate straight-key (parallel) spline shaft + hub as USD + GLB files.

Geometry estimated from reference video (industrial hand-assembly of a spline
shaft into a bolted hub):
  - 6 keys, ~50 mm major diameter, ~40 mm minor diameter, ~10 mm key width
  - Profile type: straight-key / parallel-side (ISO 14 / SAE B style)
  - NOT DIN 5480 involute -- the reference shows flat-flanked keys

Profile construction
--------------------
  The shaft cross-section is built point-by-point in CCW order:

    for each tooth i at angle theta_i = i * 2*pi/z:
      1. left  flank base  (r_min, tangential offset +hw)
      2. left  flank tip   (r_maj, tangential offset +hw)
      3. right flank tip   (r_maj, tangential offset -hw)  -- flat top
      4. right flank base  (r_min, tangential offset -hw)
      5. root arc from right base of tooth i to left base of tooth i+1
         (n_root interior sample points)

  This avoids Shapely boolean unions and produces a clean, well-conditioned
  polygon with uniform vertex spacing.

  Hub bore = outer circle - shaft profile expanded by gap/2.

Run
---
::

    python gen_geometry.py
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import trimesh
import trimesh.creation
from shapely.geometry import Polygon
from pxr import Gf, Usd, UsdGeom, Vt

HERE = pathlib.Path(__file__).parent

# ── Parameters (SI -- metres) ─────────────────────────────────────────────────

#: Number of keys/teeth.  6 visible in reference video.
Z: int = 6

#: Major (tip) diameter [m].  Estimated ~50 mm from hand-grip proportions.
D_MAJOR: float = 0.050

#: Minor (root) diameter [m].  Estimated ~40 mm -> tooth height = 5 mm.
D_MINOR: float = 0.040

#: Key (tooth) width [m].  Gap ~= key width visually -> ~10 mm.
KEY_WIDTH: float = 0.010

#: Shaft engagement length [m].  Shaft is noticeably longer than hub in video.
LENGTH: float = 0.100

#: Hub outer diameter [m].  Flange ~1.8x shaft OD in video -> ~90 mm.
HUB_OD: float = 0.090

#: Shaft-to-hub diametral clearance [m].
#: 0.4 mm (0.2 mm/side) — looks like a close running fit, and stays above the SDF
#: voxel size at MESH_SDF_RES=512 (~0.176 mm/voxel) so the SDF can distinguish
#: "tooth inside groove" from "tooth on land" (1.14× margin, same as res=256/GAP=0.8mm).
GAP: float = 0.0004

#: Hub length = LENGTH * factor.  Hub is shorter than shaft (matches video).
HUB_LEN_FACTOR: float = 0.45

#: Root arc sample points between adjacent teeth (interior only).
N_ROOT_ARC: int = 12


# ── 2-D shaft profile ─────────────────────────────────────────────────────────

def straight_key_profile(
    z: int = Z,
    d_major: float = D_MAJOR,
    d_minor: float = D_MINOR,
    key_width: float = KEY_WIDTH,
    n_root_arc: int = N_ROOT_ARC,
) -> np.ndarray:
    """
    2-D closed polygon of an external straight-key spline cross-section.

    Built explicitly point-by-point (CCW winding) without any Shapely boolean
    operations, giving a clean uniform-density polygon.

    For each of the z teeth:
      left-flank base -> left-flank tip -> flat top -> right-flank tip ->
      right-flank base -> root arc to next tooth

    Returns
    -------
    np.ndarray  shape (N, 2), float32
    """
    r_min = d_minor / 2.0
    r_maj = d_major / 2.0
    hw    = key_width / 2.0
    pitch = 2.0 * math.pi / z

    if hw >= r_min:
        raise ValueError(f"key half-width {hw*1e3:.1f} mm must be < root radius {r_min*1e3:.1f} mm")

    # Radial distance at which the flank (at tangential offset ±hw) intersects
    # the root circle: r_local² + hw² = r_min²  →  r_local = sqrt(r_min² - hw²).
    # Using r_min here instead would place the base outside the root circle and
    # create a polygon with near-zero-area notches that fail validity checks.
    r_base = math.sqrt(r_min**2 - hw**2)

    pts: list[list[float]] = []

    for i in range(z):
        th   = i * pitch
        c, s = math.cos(th), math.sin(th)

        # Correct CCW traversal order: enter tooth from the CW side (right flank),
        # cross the flat top in the CCW direction, exit on the CCW side (left flank).
        # This keeps the short gap arc between teeth (pitch - 2*alpha ~ 31 deg for
        # z=6) rather than the long arc (pitch + 2*alpha ~ 89 deg) which overlaps
        # the arcs of neighbouring teeth and causes self-intersections.

        # Right flank (CW side of tooth — first contact when approaching CCW)
        rx_base = r_base * c + hw * s;  ry_base = r_base * s - hw * c
        rx_tip  = r_maj  * c + hw * s;  ry_tip  = r_maj  * s - hw * c

        # Left flank (CCW side — exit toward next gap)
        lx_tip  = r_maj  * c - hw * s;  ly_tip  = r_maj  * s + hw * c
        lx_base = r_base * c - hw * s;  ly_base = r_base * s + hw * c

        pts.append([rx_base, ry_base])  # enter tooth
        pts.append([rx_tip,  ry_tip])   # up right flank
        pts.append([lx_tip,  ly_tip])   # across flat top (CCW)
        pts.append([lx_base, ly_base])  # down left flank

        # Short gap arc: from left_base_i to right_base_{i+1}
        # Span = pitch - 2*alpha  (the actual gap between adjacent teeth)
        th_next  = th + pitch
        c1, s1   = math.cos(th_next), math.sin(th_next)
        nrx_base = r_base * c1 + hw * s1
        nry_base = r_base * s1 - hw * c1

        phi1 = math.atan2(ly_base, lx_base)
        phi2 = math.atan2(nry_base, nrx_base)
        if phi2 <= phi1:
            phi2 += 2.0 * math.pi

        for phi in np.linspace(phi1, phi2, n_root_arc + 2)[1:-1]:
            pts.append([r_min * math.cos(phi), r_min * math.sin(phi)])

    return np.array(pts, dtype=np.float32)


# ── Hub bore polygon ──────────────────────────────────────────────────────────

def hub_ring_polygon(
    shaft_profile: np.ndarray,
    hub_od: float = HUB_OD,
    gap: float = GAP,
) -> Polygon:
    """
    Hub bore cross-section: outer circle minus shaft profile expanded by gap/2.

    The gap expansion creates the running clearance between key flanks and
    hub grooves.
    """
    shaft_poly = Polygon(shaft_profile)
    if not shaft_poly.is_valid:
        shaft_poly = shaft_poly.buffer(0)

    pocket = shaft_poly.buffer(gap / 2.0, quad_segs=32)
    r_hub  = hub_od / 2.0
    angles = np.linspace(0, 2.0 * math.pi, 256, endpoint=False)
    outer  = Polygon(np.column_stack([r_hub * np.cos(angles), r_hub * np.sin(angles)]))
    ring = outer.difference(pocket)
    if not ring.is_valid:
        ring = ring.buffer(0)

    # The gap buffer produces many inner-bore vertices vs 256 outer-circle vertices.
    # This imbalance causes earcut to create large spanning cap triangles (visible
    # as triangular artifacts on the hub top face).  Simplify ONLY the inner bore
    # hole ring at 0.1 mm tolerance — tooth shape is preserved while vertex count
    # drops to ~300, balancing outer (256) vs inner for earcut.
    # Note: tolerance must be smaller than the per-side clearance (GAP/2 = 0.75 mm)
    # so the bore groove profile is not rounded away.
    from shapely.geometry import LinearRing
    simplified_holes = [
        LinearRing(hole.coords).simplify(0.0001, preserve_topology=True)
        for hole in ring.interiors
    ]
    ring = Polygon(ring.exterior.coords, [list(h.coords) for h in simplified_holes])
    if not ring.is_valid:
        ring = ring.buffer(0)
    return ring


# ── 3-D mesh via extrusion ────────────────────────────────────────────────────

def profile_to_mesh(profile_2d: np.ndarray, length: float) -> trimesh.Trimesh:
    """Extrude a 2-D profile to a Z-centred 3-D mesh."""
    poly = Polygon(profile_2d)
    if not poly.is_valid:
        poly = poly.buffer(0)
    mesh = trimesh.creation.extrude_polygon(poly, height=length)
    mesh.apply_translation([0.0, 0.0, -length / 2.0])
    return mesh


def ring_to_mesh(ring: Polygon, length: float) -> trimesh.Trimesh:
    """Extrude a ring polygon to a Z-centred 3-D mesh."""
    mesh = trimesh.creation.extrude_polygon(ring, height=length)
    mesh.apply_translation([0.0, 0.0, -length / 2.0])
    return mesh


# ── USD export ────────────────────────────────────────────────────────────────

def mesh_to_usd(mesh: trimesh.Trimesh, out_path: pathlib.Path, prim_name: str) -> None:
    """Save a trimesh as a USD mesh prim (Z-up, metersPerUnit=1)."""
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    mp = UsdGeom.Mesh.Define(stage, f"/World/{prim_name}")

    verts = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    mp.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts]))
    mp.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    mp.GetFaceVertexIndicesAttr().Set(Vt.IntArray([idx for tri in faces for idx in tri]))
    mp.GetSubdivisionSchemeAttr().Set("none")

    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.Save()
    print(f"  Saved {out_path}  ({len(verts)} verts, {len(faces)} tris)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating straight-key spline shaft ...")
    shaft_profile = straight_key_profile()
    shaft_mesh    = profile_to_mesh(shaft_profile, LENGTH)
    mesh_to_usd(shaft_mesh, HERE / "shaft.usd", "Shaft")
    shaft_mesh.export(str(HERE / "shaft.glb"))
    print(f"  Saved shaft.glb  (watertight={shaft_mesh.is_watertight})")

    print("Generating hub ...")
    hub_ring = hub_ring_polygon(shaft_profile)
    hub_len  = LENGTH * HUB_LEN_FACTOR

    # ── Blind bore: bore section (ring) + solid bottom cap ────────────────────
    # The hub bore must be closed at the bottom so the shaft cannot pass through.
    # We split the hub into two individually-watertight pieces at the cap face:
    #   bore section : ring extrusion, (hub_len - CAP_THICKNESS) tall
    #                  → open bore where shaft teeth engage hub grooves
    #   cap section  : solid disc, CAP_THICKNESS tall, at the bore bottom
    #                  → mechanical stop; shaft bottom face seats here
    CAP_THICKNESS = 0.005  # 5 mm bottom plate
    bore_len      = hub_len - CAP_THICKNESS  # 40 mm open bore

    bore_mesh = ring_to_mesh(hub_ring, bore_len)
    # ring_to_mesh centres at z=0; shift UP so top stays at +hub_len/2
    bore_mesh.apply_translation([0.0, 0.0, CAP_THICKNESS / 2.0])
    # bore_mesh now spans [-(hub_len/2 - CAP_THICKNESS), +hub_len/2]
    #                   = [-17.5 mm, +22.5 mm]

    cap_angles  = np.linspace(0, 2.0 * math.pi, 128, endpoint=False)
    cap_profile = np.column_stack([
        (HUB_OD / 2.0) * np.cos(cap_angles),
        (HUB_OD / 2.0) * np.sin(cap_angles),
    ]).astype(np.float32)
    cap_mesh = profile_to_mesh(cap_profile, CAP_THICKNESS)
    # profile_to_mesh centres at z=0; shift DOWN so bottom is at -hub_len/2
    cap_mesh.apply_translation([0.0, 0.0, -(hub_len / 2.0 - CAP_THICKNESS / 2.0)])
    # cap_mesh now spans [-hub_len/2, -hub_len/2 + CAP_THICKNESS]
    #                  = [-22.5 mm, -17.5 mm]

    hub_mesh = trimesh.util.concatenate([bore_mesh, cap_mesh])

    mesh_to_usd(hub_mesh, HERE / "hub.usd", "Hub")
    hub_mesh.export(str(HERE / "hub.glb"))
    print(f"  Saved hub.glb  (watertight={hub_mesh.is_watertight})")
    seat_z_mm = (LENGTH / 2.0 - hub_len / 2.0 + CAP_THICKNESS) * 1000.0
    print(f"  Bore depth     : {bore_len*1000:.1f} mm  |  Cap thickness: {CAP_THICKNESS*1000:.1f} mm")
    print(f"  Shaft seats (centre) at world Z = +{seat_z_mm:.1f} mm  "
          f"({bore_len*1000:.0f} mm shaft inside hub)")

    print(
        f"\n  z={Z}  D_maj={D_MAJOR*1000:.0f} mm  D_min={D_MINOR*1000:.0f} mm  "
        f"key_w={KEY_WIDTH*1000:.0f} mm  shaft_len={LENGTH*1000:.0f} mm  "
        f"hub_len={hub_len*1000:.0f} mm  hub_od={HUB_OD*1000:.0f} mm"
    )
    print(f"  Tooth height : {(D_MAJOR - D_MINOR) / 2 * 1000:.1f} mm")
    print(f"  Shaft bounds : {shaft_mesh.bounds}")
    print(f"  Hub   bounds : {hub_mesh.bounds}")
