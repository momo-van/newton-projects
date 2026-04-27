# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Sanity-check tests for physics constants in rj45_hydro.py.

Verifies that constant values are physically plausible so that accidental
edits (e.g. wrong units, sign flip) are caught immediately.

Run
---
::

    pytest tests/test_constants.py -v
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Pre-import heavy packages so patch.dict does not evict them from sys.modules
# when it restores state (anything imported *inside* patch.dict that wasn't
# there before gets removed on exit, breaking subsequent imports).
import numpy  # noqa: F401
import warp   # noqa: F401


# ---------------------------------------------------------------------------
# Load rj45_hydro constants without executing __main__ or requiring a GPU
# ---------------------------------------------------------------------------

def _build_stubs() -> dict:
    """Return a dict of module-name → stub suitable for patch.dict(sys.modules)."""
    stubs: dict = {}

    # warp stub
    wp = types.ModuleType("warp")
    wp.array         = MagicMock(return_value=MagicMock())
    wp.vec3          = MagicMock(side_effect=lambda *a, **k: MagicMock())
    wp.mat33         = MagicMock(return_value=MagicMock())
    wp.quat_identity = MagicMock(return_value=MagicMock())
    wp.transform     = MagicMock(return_value=MagicMock())
    wp.kernel        = lambda f: f   # no-op decorator
    wp.float32       = float
    wp.launch        = MagicMock()
    stubs["warp"] = wp

    # Newton stub (minimal — just enough for ShapeConfig)
    newton = types.ModuleType("newton")
    newton.ModelBuilder = MagicMock()
    newton.ModelBuilder.ShapeConfig = MagicMock(
        side_effect=lambda **kw: types.SimpleNamespace(**kw)
    )
    newton.CollisionPipeline = MagicMock()
    newton.ShapeFlags        = MagicMock()
    newton.Mesh              = MagicMock()
    stubs["newton"] = newton

    for sub in ("newton.examples", "newton.usd", "newton.utils",
                "pxr", "pxr.Usd", "pxr.UsdGeom"):
        stubs[sub] = types.ModuleType(sub)

    # newton.geometry needs HydroelasticSDF
    ng = types.ModuleType("newton.geometry")
    ng.HydroelasticSDF = MagicMock()
    stubs["newton.geometry"] = ng

    # newton.math needs quat_between_vectors_robust
    nm = types.ModuleType("newton.math")
    nm.quat_between_vectors_robust = MagicMock()
    stubs["newton.math"] = nm

    # newton.solvers needs SolverVBD
    ns = types.ModuleType("newton.solvers")
    ns.SolverVBD = MagicMock()
    stubs["newton.solvers"] = ns

    # hydro_contact_viz stub — only needs HydroContactViz to be importable
    hcv = types.ModuleType("hydro_contact_viz")
    hcv.HydroContactViz      = MagicMock()
    hcv.DEFAULT_UPDATE_INTERVAL = 5
    stubs["hydro_contact_viz"] = hcv

    return stubs


def _load_constants():
    """
    Import rj45_hydro.py into a fresh module object using a temporary stub
    environment, then return that module.  Uses patch.dict so stubs are
    removed from sys.modules after loading — they do not affect other tests.
    """
    src  = pathlib.Path(__file__).parent.parent / "rj45_hydro.py"
    spec = importlib.util.spec_from_file_location("rj45_hydro_constants", src)
    mod  = importlib.util.module_from_spec(spec)
    mod.__name__ = "rj45_hydro_constants"   # avoid triggering __main__

    stubs = _build_stubs()
    with patch.dict(sys.modules, stubs):
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass   # partial load is fine — we only need the module-level constants

    return mod


_rj45 = _load_constants()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestShapeConstants(unittest.TestCase):

    def test_shape_kh_positive(self):
        self.assertGreater(_rj45.SHAPE_KH, 0,
                           "Hydroelastic stiffness must be positive")

    def test_shape_kh_reasonable_range(self):
        """kh should be between 1 MPa/m and 10 GPa/m for typical elastomers/metals."""
        self.assertGreaterEqual(_rj45.SHAPE_KH, 1.0e5)
        self.assertLessEqual(_rj45.SHAPE_KH,    1.0e10)

    def test_sdf_max_resolution_positive(self):
        self.assertGreater(_rj45.MESH_SDF_MAX_RESOLUTION, 0)

    def test_sdf_narrow_band_symmetric(self):
        lo, hi = _rj45.MESH_SDF_NARROW_BAND_RANGE
        self.assertLess(lo, 0,  "Narrow band lower bound must be negative")
        self.assertGreater(hi, 0, "Narrow band upper bound must be positive")
        self.assertAlmostEqual(abs(lo), abs(hi), places=9,
                               msg="Narrow band should be symmetric")


class TestCableConstants(unittest.TestCase):

    def test_cable_radius_positive(self):
        self.assertGreater(_rj45.CABLE_RADIUS, 0)

    def test_cable_radius_plausible_mm_range(self):
        """RJ45 cable is ~6.5 mm diameter → radius ~3.25 mm."""
        r_mm = _rj45.CABLE_RADIUS * 1000
        self.assertGreater(r_mm, 0.5)
        self.assertLess(r_mm, 20.0)

    def test_cable_kinematic_count_positive(self):
        self.assertGreater(_rj45.CABLE_KINEMATIC_COUNT, 0)

    def test_cable_stiffness_ordering(self):
        """Stretch stiffness should greatly exceed damping."""
        self.assertGreater(_rj45.CABLE_KE, _rj45.CABLE_KD)


class TestLatchConstants(unittest.TestCase):

    def test_latch_limits_ordered(self):
        self.assertLess(_rj45.LATCH_LIMIT_LOWER, _rj45.LATCH_LIMIT_UPPER,
                        "Lower joint limit must be less than upper limit")

    def test_latch_spring_stiffness_positive(self):
        self.assertGreater(_rj45.LATCH_SPRING_KE, 0)
        self.assertGreater(_rj45.LATCH_SPRING_KD, 0)


class TestPlugOffset(unittest.TestCase):

    def test_plug_y_offset_negative(self):
        """Plug starts above the socket (negative Y in this scene convention)."""
        self.assertLess(_rj45.PLUG_Y_OFFSET, 0,
                        "PLUG_Y_OFFSET should position plug above socket mouth")


if __name__ == "__main__":
    unittest.main()
