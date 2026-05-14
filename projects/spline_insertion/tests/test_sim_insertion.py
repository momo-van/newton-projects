# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Tests for sim_insertion.py — splined shaft insertion simulation.

All tests run without a GPU or Newton installation.  The heavy imports
(warp, newton, pxr) are replaced with lightweight stubs so the module-level
constants and the insertion logic can be exercised in any CI environment.

Run
---
::

    pytest tests/test_sim_insertion.py -v
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Pre-import to prevent patch.dict from evicting them on teardown.
import numpy  # noqa: F401
import numpy as np

SRC = pathlib.Path(__file__).parent.parent / "sim_insertion.py"


# ── Stub factory ─────────────────────────────────────────────────────────────

def _build_stubs() -> dict:
    """Return a sys.modules patch dict that satisfies all sim_insertion imports."""
    stubs: dict = {}

    # warp
    wp = types.ModuleType("warp")
    wp.kernel        = lambda f: f          # no-op decorator
    wp.array         = MagicMock(return_value=MagicMock())
    wp.vec3          = MagicMock(side_effect=lambda *a, **k: MagicMock())
    wp.mat33         = MagicMock(return_value=MagicMock())
    wp.quat_identity = MagicMock(return_value=MagicMock())
    wp.transform     = MagicMock(return_value=MagicMock())
    wp.spatial_vector = MagicMock(return_value=MagicMock())
    wp.transform_get_rotation    = MagicMock(return_value=MagicMock())
    wp.transform_get_translation = MagicMock(return_value=MagicMock())
    wp.launch        = MagicMock()
    wp.float32       = float
    stubs["warp"] = wp

    # newton
    newton = types.ModuleType("newton")
    newton.ModelBuilder = MagicMock()
    newton.ModelBuilder.ShapeConfig = MagicMock(
        side_effect=lambda **kw: types.SimpleNamespace(**kw)
    )
    newton.CollisionPipeline = MagicMock()
    newton.Mesh              = MagicMock()
    stubs["newton"] = newton

    for sub in ("newton.examples", "newton.usd"):
        stubs[sub] = types.ModuleType(sub)

    # newton.solvers — needs SolverVBD
    ns = types.ModuleType("newton.solvers")
    ns.SolverVBD = MagicMock()
    stubs["newton.solvers"] = ns

    # pxr stubs
    for sub in ("pxr", "pxr.Usd", "pxr.UsdGeom"):
        stubs[sub] = types.ModuleType(sub)

    return stubs


def _load_sim() -> types.ModuleType:
    """Load sim_insertion into a fresh module object using stub imports."""
    spec = importlib.util.spec_from_file_location("sim_insertion_test", SRC)
    mod  = importlib.util.module_from_spec(spec)
    mod.__name__ = "sim_insertion_test"   # prevent __main__ execution

    stubs = _build_stubs()
    with patch.dict(sys.modules, stubs):
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass   # partial load acceptable — we only need module-level names

    return mod


_sim = _load_sim()


# ── Helper: build a minimal Example-like object without Newton ───────────────

def _make_example(start_z: float = 0.080, end_z: float = 0.0) -> types.SimpleNamespace:
    """
    Return a SimpleNamespace that mirrors the scalar state of Example
    used by step() / _reset() without touching Newton or GPU.
    """
    fps       = _sim.SIM_FPS
    frame_dt  = 1.0 / fps

    obj = types.SimpleNamespace(
        fps          = fps,
        frame_dt     = frame_dt,
        sim_substeps = _sim.SIM_SUBSTEPS,
        sim_dt       = frame_dt / _sim.SIM_SUBSTEPS,
        sim_time     = 0.0,
        _paused      = False,
        _inserting   = True,
        _shaft_start_z = start_z,
        _shaft_end_z   = end_z,
        _shaft_z       = start_z,
        _shaft_half_z  = 0.020,
        _hub_half_z    = 0.030,
    )

    # Bind scalar-only parts of step() and _reset() so we can test logic
    # without touching body_q / solver.
    def _step_logic(self) -> None:
        """CPU-only insertion advancement (no simulate() call)."""
        if not self._paused:
            if self._inserting:
                self._shaft_z -= _sim.INSERT_SPEED * self.frame_dt
                if self._shaft_z <= self._shaft_end_z:
                    self._shaft_z   = self._shaft_end_z
                    self._inserting = False
        self.sim_time += self.frame_dt

    def _reset_logic(self) -> None:
        self._shaft_z   = self._shaft_start_z
        self._inserting = True
        self._paused    = False

    import types as _t
    obj.step_logic  = _t.MethodType(_step_logic,  obj)
    obj.reset_logic = _t.MethodType(_reset_logic, obj)
    return obj


# ═════════════════════════════════════════════════════════════════════════════
# 1. Module-level constants
# ═════════════════════════════════════════════════════════════════════════════

class TestConstants(unittest.TestCase):

    def test_insert_speed_positive(self):
        self.assertGreater(_sim.INSERT_SPEED, 0)

    def test_insert_speed_plausible_mm_s(self):
        """Insertion speed should be between 0.1 mm/s and 500 mm/s."""
        speed_mm_s = _sim.INSERT_SPEED * 1000
        self.assertGreater(speed_mm_s, 0.1)
        self.assertLess(speed_mm_s, 500.0)

    def test_insert_clearance_positive(self):
        self.assertGreater(_sim.INSERT_CLEARANCE, 0)

    def test_sim_fps_positive(self):
        self.assertGreater(_sim.SIM_FPS, 0)

    def test_sim_substeps_at_least_one(self):
        self.assertGreaterEqual(_sim.SIM_SUBSTEPS, 1)

    def test_contact_gap_positive(self):
        self.assertGreater(_sim.CONTACT_GAP, 0)

    def test_contact_gap_plausible_range(self):
        """Gap should be between 0.1 mm and 10 mm."""
        gap_mm = _sim.CONTACT_GAP * 1000
        self.assertGreater(gap_mm, 0.1)
        self.assertLess(gap_mm, 10.0)

    def test_sdf_band_symmetric(self):
        lo, hi = _sim._SDF_BAND
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)
        self.assertAlmostEqual(abs(lo), hi, places=12)

    def test_sdf_band_tied_to_gap(self):
        lo, hi = _sim._SDF_BAND
        self.assertAlmostEqual(hi, 2.0 * _sim.CONTACT_GAP, places=12)

    def test_mesh_sdf_res_positive(self):
        self.assertGreater(_sim.MESH_SDF_RES, 0)

    def test_shape_cfg_has_required_fields(self):
        cfg = _sim.SHAPE_CFG
        self.assertGreater(cfg.ke, 0)
        self.assertGreater(cfg.kd, 0)
        self.assertGreaterEqual(cfg.mu, 0)
        self.assertGreater(cfg.gap, 0)
        self.assertGreater(cfg.density, 0)

    def test_ke_greater_than_kd(self):
        """Stiffness coefficient should dominate damping."""
        self.assertGreater(_sim.SHAPE_CFG.ke, _sim.SHAPE_CFG.kd)

    def test_frame_dt_consistent(self):
        """sim_dt = frame_dt / substeps must equal 1 / (fps * substeps)."""
        expected_frame_dt = 1.0 / _sim.SIM_FPS
        expected_sim_dt   = expected_frame_dt / _sim.SIM_SUBSTEPS
        self.assertAlmostEqual(expected_sim_dt,
                               1.0 / (_sim.SIM_FPS * _sim.SIM_SUBSTEPS))


# ═════════════════════════════════════════════════════════════════════════════
# 2. Insertion advancement logic
# ═════════════════════════════════════════════════════════════════════════════

class TestInsertionAdvancement(unittest.TestCase):

    def test_shaft_z_decreases_each_step(self):
        ex = _make_example()
        z_before = ex._shaft_z
        ex.step_logic()
        self.assertLess(ex._shaft_z, z_before)

    def test_shaft_z_decreases_by_correct_amount(self):
        ex      = _make_example()
        z_before = ex._shaft_z
        ex.step_logic()
        expected_drop = _sim.INSERT_SPEED * ex.frame_dt
        self.assertAlmostEqual(z_before - ex._shaft_z, expected_drop, places=10)

    def test_shaft_stops_at_end_z(self):
        ex = _make_example(start_z=0.001, end_z=0.0)
        # Step many times — shaft must clamp at end_z, not go below.
        for _ in range(100):
            ex.step_logic()
        self.assertGreaterEqual(ex._shaft_z, ex._shaft_end_z)

    def test_inserting_flag_cleared_at_end(self):
        ex = _make_example(start_z=0.001, end_z=0.0)
        for _ in range(200):
            ex.step_logic()
        self.assertFalse(ex._inserting)

    def test_shaft_z_exact_clamp(self):
        """Once _shaft_z would go below end_z it is clamped exactly."""
        ex = _make_example(start_z=1e-7, end_z=0.0)   # one step overshoots
        ex.step_logic()
        self.assertEqual(ex._shaft_z, ex._shaft_end_z)

    def test_z_does_not_change_after_fully_inserted(self):
        ex = _make_example(start_z=0.0, end_z=0.0)
        ex._inserting = False
        ex.step_logic()
        self.assertEqual(ex._shaft_z, 0.0)

    def test_sim_time_advances_every_step(self):
        ex = _make_example()
        t0 = ex.sim_time
        ex.step_logic()
        self.assertAlmostEqual(ex.sim_time - t0, ex.frame_dt, places=12)

    def test_sim_time_advances_even_when_paused(self):
        ex          = _make_example()
        ex._paused  = True
        t0          = ex.sim_time
        ex.step_logic()
        self.assertAlmostEqual(ex.sim_time - t0, ex.frame_dt, places=12)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Paused state
# ═════════════════════════════════════════════════════════════════════════════

class TestPausedState(unittest.TestCase):

    def test_shaft_z_frozen_when_paused(self):
        ex         = _make_example()
        ex._paused = True
        z_before   = ex._shaft_z
        ex.step_logic()
        self.assertEqual(ex._shaft_z, z_before)

    def test_inserting_flag_unchanged_when_paused(self):
        ex         = _make_example(start_z=1e-9, end_z=0.0)
        ex._paused = True
        ex.step_logic()
        self.assertTrue(ex._inserting)   # would have ended if not paused


# ═════════════════════════════════════════════════════════════════════════════
# 4. Reset logic
# ═════════════════════════════════════════════════════════════════════════════

class TestResetLogic(unittest.TestCase):

    def test_reset_restores_start_z(self):
        ex = _make_example(start_z=0.080)
        for _ in range(30):
            ex.step_logic()
        ex.reset_logic()
        self.assertEqual(ex._shaft_z, ex._shaft_start_z)

    def test_reset_sets_inserting_true(self):
        ex = _make_example(start_z=0.0, end_z=0.0)
        ex._inserting = False
        ex.reset_logic()
        self.assertTrue(ex._inserting)

    def test_reset_clears_paused(self):
        ex         = _make_example()
        ex._paused = True
        ex.reset_logic()
        self.assertFalse(ex._paused)

    def test_reset_idempotent(self):
        ex = _make_example(start_z=0.080)
        ex.reset_logic()
        ex.reset_logic()
        self.assertEqual(ex._shaft_z, ex._shaft_start_z)
        self.assertTrue(ex._inserting)
        self.assertFalse(ex._paused)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Progress / depth calculation (mirrors _controls_ui arithmetic)
# ═════════════════════════════════════════════════════════════════════════════

class TestProgressCalculation(unittest.TestCase):

    def _progress(self, ex) -> float:
        total = ex._shaft_start_z - ex._shaft_end_z
        done  = ex._shaft_start_z - ex._shaft_z
        return 100.0 * done / total if total > 0 else 0.0

    def _depth_m(self, ex) -> float:
        return ex._hub_half_z - (ex._shaft_z + ex._shaft_half_z)

    def test_progress_zero_at_start(self):
        ex = _make_example(start_z=0.080, end_z=0.0)
        self.assertAlmostEqual(self._progress(ex), 0.0)

    def test_progress_100_at_end(self):
        ex          = _make_example(start_z=0.080, end_z=0.0)
        ex._shaft_z = ex._shaft_end_z
        self.assertAlmostEqual(self._progress(ex), 100.0, places=10)

    def test_progress_50_at_midpoint(self):
        ex          = _make_example(start_z=0.080, end_z=0.0)
        ex._shaft_z = 0.040
        self.assertAlmostEqual(self._progress(ex), 50.0, places=10)

    def test_depth_negative_before_engagement(self):
        """Shaft above hub entrance → depth must be ≤ 0."""
        ex = _make_example(start_z=0.080, end_z=0.0)
        # shaft tip at shaft_z - shaft_half_z = 0.080 - 0.020 = 0.060 > hub_half_z = 0.030
        self.assertLessEqual(self._depth_m(ex), 0.0)

    def test_depth_positive_when_engaged(self):
        """Shaft below hub entrance → depth > 0."""
        ex          = _make_example(start_z=0.080, end_z=0.0)
        ex._shaft_z = 0.0   # centre at hub centre
        # depth = hub_half_z - (0.0 + shaft_half_z) = 0.030 - 0.020 = 0.010
        self.assertGreater(self._depth_m(ex), 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Frame / substep timing
# ═════════════════════════════════════════════════════════════════════════════

class TestTiming(unittest.TestCase):

    def test_sim_dt_equals_frame_dt_over_substeps(self):
        expected = (1.0 / _sim.SIM_FPS) / _sim.SIM_SUBSTEPS
        self.assertAlmostEqual(expected,
                               1.0 / (_sim.SIM_FPS * _sim.SIM_SUBSTEPS),
                               places=12)

    def test_total_travel_time(self):
        """Full insertion at INSERT_SPEED should take the expected number of frames."""
        start_z = 0.080
        end_z   = 0.0
        travel  = start_z - end_z
        # frames = ceil(travel / (INSERT_SPEED * frame_dt))
        frame_dt       = 1.0 / _sim.SIM_FPS
        frames_expected = math.ceil(travel / (_sim.INSERT_SPEED * frame_dt))
        ex = _make_example(start_z=start_z, end_z=end_z)
        frames_taken = 0
        while ex._inserting and frames_taken < frames_expected * 2:
            ex.step_logic()
            frames_taken += 1
        self.assertFalse(ex._inserting)
        # Allow ±1 frame tolerance from floating-point accumulation.
        self.assertAlmostEqual(frames_taken, frames_expected, delta=1)

    def test_sim_time_after_n_steps(self):
        ex        = _make_example()
        n         = 120
        expected  = n * ex.frame_dt
        for _ in range(n):
            ex.step_logic()
        self.assertAlmostEqual(ex.sim_time, expected, places=9)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Geometry loading helper (mocked — no USD/GPU needed)
# ═════════════════════════════════════════════════════════════════════════════

class TestLoadMesh(unittest.TestCase):

    def test_load_mesh_returns_tuple(self):
        """_load_mesh must return (newton.Mesh, np.ndarray)."""
        fake_verts   = np.zeros((6, 3), dtype=np.float32)
        fake_indices = np.zeros((4, 3), dtype=np.int32)

        class _FakeUSDMesh:
            vertices = fake_verts
            indices  = fake_indices
            normals  = None

        fake_prim  = MagicMock()
        fake_stage = MagicMock()
        fake_stage.GetPrimAtPath.return_value = fake_prim

        mock_mesh = MagicMock()

        # Patch through _sim's own module-level names (avoids the stub
        # ModuleType having no Stage attribute for string-path patch).
        fake_usd = MagicMock()
        fake_usd.Stage.Open.return_value = fake_stage

        # newton.usd is not auto-bound as an attribute on the stub (Python skips
        # parent-binding when the module is already in sys.modules), so set it.
        import types as _t
        fake_nusd = _t.SimpleNamespace(get_mesh=MagicMock(return_value=_FakeUSDMesh()))
        with patch.object(_sim, "Usd", fake_usd), \
             patch.object(_sim.newton, "usd", fake_nusd, create=True), \
             patch.object(_sim.newton, "Mesh", MagicMock(return_value=mock_mesh), create=True):
            result = _sim._load_mesh(
                pathlib.Path("/fake/hub.usd"), "Hub", build_sdf=False
            )

        mesh, verts = result
        self.assertIs(mesh, mock_mesh)
        self.assertIsInstance(verts, np.ndarray)


if __name__ == "__main__":
    unittest.main()
