"""
Generate 6-groove splined shaft + hub, overwriting shaft.usd/hub.usd.

z=6, module=4mm → ~32 mm tip diameter (similar OD to the 16-tooth 2mm-module shaft).
Run:  python gen_6groove.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import gen_geometry as _g
import math

# Override parameters
_g.Z      = 6
_g.MODULE = 0.004          # 4 mm module → pitch dia = 24 mm, tip dia ≈ 32 mm
_g.ALPHA  = math.radians(30.0)
_g.LENGTH = 0.040
_g.HUB_OD = 0.064          # 64 mm OD → ~16 mm wall
_g.GAP    = 0.0005         # slightly larger clearance for chunkier teeth

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    print("Generating 6-groove shaft geometry  (z=6, module=4 mm)...")
    shaft_profile = _g.external_spline_profile(
        z=_g.Z, module=_g.MODULE, alpha=_g.ALPHA,
        n_inv=_g.N_INV, n_arc=_g.N_ARC,
    )
    shaft_mesh = _g.profile_to_mesh(shaft_profile, _g.LENGTH)
    _g.mesh_to_usd(shaft_mesh, HERE / "shaft.usd", "Shaft")
    shaft_mesh.export(str(HERE / "shaft.glb"))
    print(f"  Saved shaft.glb")

    print("Generating 6-groove hub geometry...")
    hub_ring = _g.hub_ring_polygon(
        shaft_profile, hub_od=_g.HUB_OD, gap=_g.GAP
    )
    hub_mesh = _g.ring_to_mesh(hub_ring, _g.LENGTH * 1.5)
    _g.mesh_to_usd(hub_mesh, HERE / "hub.usd", "Hub")
    hub_mesh.export(str(HERE / "hub.glb"))
    print(f"  Saved hub.glb")

    print("\nDone.")
    print(f"  Shaft tip dia  : {(_g.MODULE * _g.Z / 2 + _g.MODULE) * 2 * 1000:.1f} mm")
    print(f"  Hub outer dia  : {_g.HUB_OD * 1000:.0f} mm")
    print(f"  Shaft watertight : {shaft_mesh.is_watertight}")
    print(f"  Hub   watertight : {hub_mesh.is_watertight}")
    print(f"  Shaft bounds   : {shaft_mesh.bounds}")
    print(f"  Hub   bounds   : {hub_mesh.bounds}")
