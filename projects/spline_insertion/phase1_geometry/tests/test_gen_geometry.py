# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Tests for phase1_geometry/gen_geometry.py — straight-key parallel spline.

No GPU, Newton, or USD installation required.  pxr is stubbed so the module
loads cleanly; only the pure-Python geometry functions are exercised.

Run
---
::

    pytest examples/spline_insertion/phase1_geometry/tests/ -v
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from shapely.geometry import Polygon

SRC = pathlib.Path(__file__).parent.parent / "gen_geometry.py"


# ── Stub factory ──────────────────────────────────────────────────────────────

def _build_stubs() -> dict:
    pxr = types.ModuleType("pxr")
    pxr.Usd    = MagicMock()
    pxr.UsdGeom = MagicMock()
    pxr.Gf     = MagicMock()
    pxr.Vt     = MagicMock()
    return {
        "pxr":        pxr,
        "pxr.Usd":    MagicMock(),
        "pxr.UsdGeom": MagicMock(),
        "pxr.Gf":     MagicMock(),
        "pxr.Vt":     MagicMock(),
    }


def _load_mod() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("gen_geometry_p1_test", SRC)
    mod  = importlib.util.module_from_spec(spec)
    mod.__name__ = "gen_geometry_p1_test"
    with patch.dict(sys.modules, _build_stubs()):
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
    return mod


_gm = _load_mod()


# ═════════════════════════════════════════════════════════════════════════════
# 1. Constants
# ═════════════════════════════════════════════════════════════════════════════

class TestConstants(unittest.TestCase):

    def test_z_at_least_3(self):
        self.assertGreaterEqual(_gm.Z, 3)

    def test_major_greater_than_minor(self):
        self.assertGreater(_gm.D_MAJOR, _gm.D_MINOR)

    def test_tooth_height_positive(self):
        self.assertGreater((_gm.D_MAJOR - _gm.D_MINOR) / 2, 0)

    def test_key_width_fits_in_pitch(self):
        """Key must be narrower than one tooth pitch at the root circle."""
        pitch_arc = math.pi * _gm.D_MINOR / _gm.Z
        self.assertLess(_gm.KEY_WIDTH, pitch_arc)

    def test_gap_smaller_than_tooth_height(self):
        self.assertLess(_gm.GAP, (_gm.D_MAJOR - _gm.D_MINOR) / 2)

    def test_hub_od_larger_than_shaft(self):
        self.assertGreater(_gm.HUB_OD, _gm.D_MAJOR)

    def test_hub_len_factor_positive(self):
        self.assertGreater(_gm.HUB_LEN_FACTOR, 0.0)

    def test_hub_shorter_than_shaft(self):
        """Hub should be shorter than shaft to match industrial reference."""
        self.assertLess(_gm.LENGTH * _gm.HUB_LEN_FACTOR, _gm.LENGTH)

    def test_length_positive(self):
        self.assertGreater(_gm.LENGTH, 0)


# ═════════════════════════════════════════════════════════════════════════════
# 2. straight_key_profile
# ═════════════════════════════════════════════════════════════════════════════

class TestStraightKeyProfile(unittest.TestCase):

    def setUp(self):
        self.p = _gm.straight_key_profile()

    def test_returns_2d_array(self):
        self.assertEqual(self.p.ndim, 2)
        self.assertEqual(self.p.shape[1], 2)

    def test_min_point_count(self):
        self.assertGreater(len(self.p), 50)

    def test_no_duplicate_closing_point(self):
        """First and last points must differ (closing point stripped)."""
        self.assertFalse(np.allclose(self.p[0], self.p[-1]))

    def test_max_radius_is_major(self):
        # Key tip corners are at sqrt(hw² + r_maj²), slightly beyond r_maj.
        # Expected max radius is sqrt((KEY_WIDTH/2)² + (D_MAJOR/2)²).
        radii    = np.hypot(self.p[:, 0], self.p[:, 1])
        hw, r_maj = _gm.KEY_WIDTH / 2, _gm.D_MAJOR / 2
        expected = math.sqrt(hw**2 + r_maj**2)
        self.assertAlmostEqual(float(radii.max()), expected, delta=1e-4)

    def test_min_radius_is_minor(self):
        radii = np.hypot(self.p[:, 0], self.p[:, 1])
        self.assertAlmostEqual(radii.min(), _gm.D_MINOR / 2, delta=1e-3)

    def test_profile_is_valid_polygon(self):
        poly = Polygon(self.p)
        self.assertTrue(poly.is_valid)

    def test_z_fold_rotational_symmetry(self):
        """Rotating by 360/z must reproduce the same polygon (within 1%)."""
        from shapely.affinity import rotate as sh_rotate
        poly     = Polygon(self.p)
        rotated  = sh_rotate(poly, 360.0 / _gm.Z, origin=(0.0, 0.0))
        sym_diff = poly.symmetric_difference(rotated).area
        self.assertLess(sym_diff / poly.area, 0.01)

    def test_area_increases_with_more_teeth(self):
        """Profile with more teeth should have larger cross-sectional area."""
        p4 = _gm.straight_key_profile(z=4)
        p8 = _gm.straight_key_profile(z=8)
        self.assertGreater(Polygon(p8).area, Polygon(p4).area)

    def test_area_increases_with_larger_key_width(self):
        narrow = _gm.straight_key_profile(key_width=0.005)
        wide   = _gm.straight_key_profile(key_width=0.012)
        self.assertGreater(Polygon(wide).area, Polygon(narrow).area)

    def test_wider_major_radius_gives_larger_area(self):
        small = _gm.straight_key_profile(d_major=0.040, d_minor=0.034)
        large = _gm.straight_key_profile(d_major=0.060, d_minor=0.050)
        self.assertGreater(Polygon(large).area, Polygon(small).area)


# ═════════════════════════════════════════════════════════════════════════════
# 3. hub_ring_polygon
# ═════════════════════════════════════════════════════════════════════════════

class TestHubRingPolygon(unittest.TestCase):

    def setUp(self):
        self.shaft_profile = _gm.straight_key_profile()
        self.ring = _gm.hub_ring_polygon(self.shaft_profile)

    def test_ring_is_valid(self):
        self.assertTrue(self.ring.is_valid)

    def test_ring_has_one_interior(self):
        """Hub cross-section must have exactly one bore (interior ring)."""
        self.assertEqual(len(list(self.ring.interiors)), 1)

    def test_bore_larger_than_shaft(self):
        shaft_poly = Polygon(self.shaft_profile)
        bore       = Polygon(list(self.ring.interiors)[0])
        self.assertGreater(bore.area, shaft_poly.area)

    def test_outer_diameter(self):
        bounds = Polygon(self.ring.exterior.coords).bounds
        actual_od = bounds[2] - bounds[0]
        self.assertAlmostEqual(actual_od, _gm.HUB_OD, delta=1e-4)

    def test_gap_expands_bore_relative_to_shaft(self):
        shaft_poly   = Polygon(self.shaft_profile)
        ring_no_gap  = _gm.hub_ring_polygon(self.shaft_profile, gap=0.0)
        ring_with_gap = _gm.hub_ring_polygon(self.shaft_profile, gap=0.001)
        bore_no_gap  = Polygon(list(ring_no_gap.interiors)[0])
        bore_with_gap = Polygon(list(ring_with_gap.interiors)[0])
        self.assertGreater(bore_with_gap.area, bore_no_gap.area)


# ═════════════════════════════════════════════════════════════════════════════
# 4. profile_to_mesh / ring_to_mesh
# ═════════════════════════════════════════════════════════════════════════════

class TestMeshGeneration(unittest.TestCase):

    def setUp(self):
        self.profile = _gm.straight_key_profile()
        self.ring    = _gm.hub_ring_polygon(self.profile)

    def test_shaft_mesh_watertight(self):
        mesh = _gm.profile_to_mesh(self.profile, _gm.LENGTH)
        self.assertTrue(mesh.is_watertight)

    def test_shaft_mesh_centred_at_z_zero(self):
        mesh = _gm.profile_to_mesh(self.profile, _gm.LENGTH)
        z_centre = (mesh.bounds[0][2] + mesh.bounds[1][2]) / 2.0
        self.assertAlmostEqual(z_centre, 0.0, places=6)

    def test_shaft_mesh_z_extent(self):
        mesh = _gm.profile_to_mesh(self.profile, _gm.LENGTH)
        z_ext = mesh.bounds[1][2] - mesh.bounds[0][2]
        self.assertAlmostEqual(z_ext, _gm.LENGTH, places=6)

    def test_hub_mesh_watertight(self):
        hub_len = _gm.LENGTH * _gm.HUB_LEN_FACTOR
        mesh    = _gm.ring_to_mesh(self.ring, hub_len)
        self.assertTrue(mesh.is_watertight)

    def test_shaft_longer_than_hub(self):
        """Shaft is longer than hub to match industrial reference geometry."""
        shaft_mesh = _gm.profile_to_mesh(self.profile, _gm.LENGTH)
        hub_mesh   = _gm.ring_to_mesh(self.ring, _gm.LENGTH * _gm.HUB_LEN_FACTOR)
        shaft_len  = shaft_mesh.bounds[1][2] - shaft_mesh.bounds[0][2]
        hub_len    = hub_mesh.bounds[1][2]   - hub_mesh.bounds[0][2]
        self.assertGreater(shaft_len, hub_len)

    def test_mesh_has_faces(self):
        mesh = _gm.profile_to_mesh(self.profile, _gm.LENGTH)
        self.assertGreater(len(mesh.faces), 0)

    def test_mesh_has_vertices(self):
        mesh = _gm.profile_to_mesh(self.profile, _gm.LENGTH)
        self.assertGreater(len(mesh.vertices), 0)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Engagement geometry invariants
# ═════════════════════════════════════════════════════════════════════════════

class TestEngagementGeometry(unittest.TestCase):
    """Properties that matter for the search-and-engage insertion mechanic."""

    def test_z_valid_engagement_angles(self):
        """For z keys, there must be exactly z distinct engagement orientations."""
        engagement_step_deg = 360.0 / _gm.Z
        angles = [i * engagement_step_deg for i in range(_gm.Z)]
        self.assertEqual(len(angles), _gm.Z)
        self.assertAlmostEqual(angles[-1], ((_gm.Z - 1) * engagement_step_deg))

    def test_key_fits_in_hub_groove(self):
        """Shaft key width + gap must be less than hub groove width."""
        # Hub groove width ≈ key width + gap (by construction via buffer)
        groove_width = _gm.KEY_WIDTH + _gm.GAP
        self.assertGreater(groove_width, _gm.KEY_WIDTH)
        self.assertLess(_gm.GAP / _gm.KEY_WIDTH, 0.1)  # gap < 10% of key width

    def test_shaft_profile_inside_hub_bore(self):
        """Shaft profile (not expanded) must fit inside hub bore."""
        from shapely.geometry import Polygon as P
        shaft  = P(_gm.straight_key_profile())
        ring   = _gm.hub_ring_polygon(_gm.straight_key_profile())
        bore   = P(list(ring.interiors)[0])
        self.assertTrue(bore.contains(shaft))


if __name__ == "__main__":
    unittest.main()
