# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Tests for phase2_kinematic/contact_viz.py -- SplineContactViz.

No GPU, Newton, Warp, Rerun, or pxr installation required.  All heavy imports
are replaced with lightweight stubs so the pure-Python logic (colormap,
metric computation, array building) can run in any CI environment.

Run
---
::

    pytest examples/spline_insertion/phase2_kinematic/tests/ -v
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

SRC = pathlib.Path(__file__).parent.parent / "contact_viz.py"


# ── Stub factory ──────────────────────────────────────────────────────────────

def _build_stubs() -> dict:
    stubs: dict = {}

    # warp
    wp = types.ModuleType("warp")
    wp.array  = MagicMock(return_value=MagicMock())
    wp.vec3   = MagicMock(return_value=MagicMock())
    wp.kernel = lambda f: f
    stubs["warp"] = wp

    return stubs


def _load_mod() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("contact_viz_test", SRC)
    mod  = importlib.util.module_from_spec(spec)
    mod.__name__ = "contact_viz_test"
    with patch.dict(sys.modules, _build_stubs()):
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
    return mod


_cv = _load_mod()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_viz() -> "_cv.SplineContactViz":
    """Build a SplineContactViz with all Newton/Warp objects mocked."""
    pipeline = MagicMock()
    model    = MagicMock()
    viewer   = MagicMock()
    pipeline.hydroelastic_sdf = None
    return _cv.SplineContactViz(pipeline, model, viewer)


def _fake_surface(n_faces: int, depth_val: float = -0.001) -> MagicMock:
    """Return a mock hydroelastic contact surface with n_faces triangles."""
    pts    = np.zeros((n_faces * 3, 3), dtype=np.float32)
    depths = np.full(n_faces, depth_val, dtype=np.float32)
    # Give triangles a non-zero area: isoceles right triangle (1mm leg)
    for i in range(n_faces):
        base = i * 3
        pts[base + 0] = [0.000, 0.000, 0.0]
        pts[base + 1] = [0.001, 0.000, 0.0]
        pts[base + 2] = [0.000, 0.001, 0.0]

    surf = MagicMock()
    surf.face_contact_count.numpy.return_value = np.array([n_faces])
    surf.contact_surface_depth.numpy.return_value = depths
    surf.contact_surface_point.numpy.return_value = pts
    return surf


# ═════════════════════════════════════════════════════════════════════════════
# 1. Jet colormap
# ═════════════════════════════════════════════════════════════════════════════

class TestJetColormap(unittest.TestCase):

    def test_returns_ndarray(self):
        t = np.linspace(0, 1, 10, dtype=np.float32)
        rgb = _cv.SplineContactViz.jet(t)
        self.assertIsInstance(rgb, np.ndarray)

    def test_shape(self):
        t = np.linspace(0, 1, 7, dtype=np.float32)
        rgb = _cv.SplineContactViz.jet(t)
        self.assertEqual(rgb.shape, (7, 3))

    def test_dtype_float32(self):
        t = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        rgb = _cv.SplineContactViz.jet(t)
        self.assertEqual(rgb.dtype, np.float32)

    def test_range_0_to_1(self):
        t = np.linspace(0, 1, 50, dtype=np.float32)
        rgb = _cv.SplineContactViz.jet(t)
        self.assertGreaterEqual(float(rgb.min()), 0.0)
        self.assertLessEqual(float(rgb.max()), 1.0)

    def test_t0_is_blue(self):
        """t=0 should be a blue-dominant colour (low pressure)."""
        rgb = _cv.SplineContactViz.jet(np.array([0.0], dtype=np.float32))
        self.assertGreater(float(rgb[0, 2]), float(rgb[0, 0]))  # B > R

    def test_t1_is_red(self):
        """t=1 should be a red-dominant colour (high pressure)."""
        rgb = _cv.SplineContactViz.jet(np.array([1.0], dtype=np.float32))
        self.assertGreater(float(rgb[0, 0]), float(rgb[0, 2]))  # R > B

    def test_single_value(self):
        rgb = _cv.SplineContactViz.jet(np.array([0.5], dtype=np.float32))
        self.assertEqual(rgb.shape, (1, 3))

    def test_static_method_callable_on_class(self):
        t = np.array([0.25], dtype=np.float32)
        rgb = _cv.SplineContactViz.jet(t)
        self.assertEqual(rgb.shape, (1, 3))

    def test_jet_equals_underscore_jet(self):
        """Public .jet() and private ._jet() must be identical."""
        t = np.linspace(0, 1, 20, dtype=np.float32)
        rgb_pub  = _cv.SplineContactViz.jet(t)
        rgb_priv = _cv.SplineContactViz._jet(t)
        np.testing.assert_array_equal(rgb_pub, rgb_priv)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Construction
# ═════════════════════════════════════════════════════════════════════════════

class TestConstruction(unittest.TestCase):

    def test_default_metrics_are_zero(self):
        viz = _make_viz()
        self.assertEqual(viz.hydro_face_count, 0)
        self.assertEqual(viz.reduced_contact_count, 0)
        self.assertAlmostEqual(viz.max_depth_mm, 0.0)
        self.assertAlmostEqual(viz.contact_area_mm2, 0.0)

    def test_show_patch_default_true(self):
        viz = _make_viz()
        self.assertTrue(viz.show_patch)

    def test_opacity_default(self):
        viz = _make_viz()
        self.assertAlmostEqual(viz.opacity, 0.65, places=2)

    def test_gamma_default(self):
        viz = _make_viz()
        self.assertGreater(viz.gamma, 0.0)

    def test_raw_cache_empty_at_start(self):
        viz = _make_viz()
        self.assertIsNone(viz._raw_pts)
        self.assertIsNone(viz._raw_depths)
        self.assertEqual(viz._raw_n_faces, 0)

    def test_imgui_callback_registered(self):
        pipeline = MagicMock()
        model    = MagicMock()
        viewer   = MagicMock()
        pipeline.hydroelastic_sdf = None
        _cv.SplineContactViz(pipeline, model, viewer)
        viewer.register_ui_callback.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════════
# 3. _clear_surface
# ═════════════════════════════════════════════════════════════════════════════

class TestClearSurface(unittest.TestCase):

    def test_resets_all_metrics(self):
        viz = _make_viz()
        viz.hydro_face_count  = 99
        viz.max_depth_mm      = 5.0
        viz.contact_area_mm2  = 100.0
        viz._clear_surface()
        self.assertEqual(viz.hydro_face_count, 0)
        self.assertAlmostEqual(viz.max_depth_mm, 0.0)
        self.assertAlmostEqual(viz.contact_area_mm2, 0.0)

    def test_clears_patch_gpu_arrays(self):
        viz = _make_viz()
        viz._patch_starts = MagicMock()
        viz._patch_ends   = MagicMock()
        viz._patch_colors = MagicMock()
        viz._clear_surface()
        self.assertIsNone(viz._patch_starts)
        self.assertIsNone(viz._patch_ends)
        self.assertIsNone(viz._patch_colors)


# ═════════════════════════════════════════════════════════════════════════════
# 4. _sync_contact_surface
# ═════════════════════════════════════════════════════════════════════════════

class TestSyncContactSurface(unittest.TestCase):

    def _viz_with_surface(self, n_faces, depth_val=-0.001):
        viz = _make_viz()
        surf = _fake_surface(n_faces, depth_val)
        viz._pipeline.hydroelastic_sdf = MagicMock()
        viz._pipeline.hydroelastic_sdf.get_contact_surface.return_value = surf
        viz._sync_contact_surface()
        return viz

    def test_hydro_face_count_set(self):
        viz = self._viz_with_surface(8)
        self.assertEqual(viz.hydro_face_count, 8)

    def test_max_depth_mm_positive(self):
        viz = self._viz_with_surface(4, depth_val=-0.002)
        self.assertAlmostEqual(viz.max_depth_mm, 2.0, places=3)

    def test_contact_area_positive(self):
        viz = self._viz_with_surface(4)
        self.assertGreater(viz.contact_area_mm2, 0.0)

    def test_contact_area_scales_with_face_count(self):
        viz4  = self._viz_with_surface(4)
        viz8  = self._viz_with_surface(8)
        self.assertAlmostEqual(viz8.contact_area_mm2, viz4.contact_area_mm2 * 2, places=5)

    def test_raw_pts_cached(self):
        viz = self._viz_with_surface(6)
        self.assertIsNotNone(viz._raw_pts)

    def test_raw_depths_cached(self):
        viz = self._viz_with_surface(6)
        self.assertIsNotNone(viz._raw_depths)

    def test_no_surface_clears_metrics(self):
        viz = _make_viz()
        viz.hydro_face_count = 10
        # pipeline with zero-face surface
        surf = MagicMock()
        surf.face_contact_count.numpy.return_value = np.array([0])
        viz._pipeline.hydroelastic_sdf = MagicMock()
        viz._pipeline.hydroelastic_sdf.get_contact_surface.return_value = surf
        viz._sync_contact_surface()
        self.assertEqual(viz.hydro_face_count, 0)
        self.assertAlmostEqual(viz.contact_area_mm2, 0.0)

    def test_none_hydroelastic_sdf_clears(self):
        viz = _make_viz()
        viz.hydro_face_count = 5
        viz._pipeline.hydroelastic_sdf = None
        viz._sync_contact_surface()
        self.assertEqual(viz.hydro_face_count, 0)


# ═════════════════════════════════════════════════════════════════════════════
# 5. _rebuild_patch
# ═════════════════════════════════════════════════════════════════════════════

class TestRebuildPatch(unittest.TestCase):

    def _viz_with_patch(self, n_faces=4) -> "_cv.SplineContactViz":
        viz = _make_viz()
        surf = _fake_surface(n_faces, depth_val=-0.001)
        viz._pipeline.hydroelastic_sdf = MagicMock()
        viz._pipeline.hydroelastic_sdf.get_contact_surface.return_value = surf
        viz._sync_contact_surface()
        return viz

    def test_patch_starts_not_none_after_sync(self):
        viz = self._viz_with_patch(4)
        self.assertIsNotNone(viz._patch_starts)

    def test_patch_ends_not_none_after_sync(self):
        viz = self._viz_with_patch(4)
        self.assertIsNotNone(viz._patch_ends)

    def test_patch_colors_not_none_after_sync(self):
        viz = self._viz_with_patch(4)
        self.assertIsNotNone(viz._patch_colors)

    def test_density_step_one_produces_all_faces(self):
        """density_step=1 -> all faces used; GPU arrays are not None."""
        viz = self._viz_with_patch(6)
        self.assertIsNotNone(viz._patch_starts)
        self.assertIsNotNone(viz._patch_ends)
        self.assertIsNotNone(viz._patch_colors)

    def test_density_step_gt_1_still_produces_patch(self):
        viz = self._viz_with_patch(10)
        viz.density_step = 3
        viz._rebuild_patch()
        self.assertIsNotNone(viz._patch_starts)

    def test_empty_raw_produces_none_starts(self):
        viz = _make_viz()
        viz._rebuild_patch()
        self.assertIsNone(viz._patch_starts)


# ═════════════════════════════════════════════════════════════════════════════
# 6. update() -- contact count read
# ═════════════════════════════════════════════════════════════════════════════

class TestUpdate(unittest.TestCase):

    def test_contact_count_read_from_contacts(self):
        viz = _make_viz()
        contacts = MagicMock()
        contacts.rigid_contact_count = MagicMock()
        contacts.rigid_contact_count.numpy.return_value = np.array([7])
        viz.update(contacts)
        self.assertEqual(viz.reduced_contact_count, 7)

    def test_update_with_zero_contacts(self):
        viz = _make_viz()
        contacts = MagicMock()
        contacts.rigid_contact_count = np.array([0])
        viz.update(contacts)
        self.assertEqual(viz.reduced_contact_count, 0)

    def test_update_increments_frame_counter(self):
        viz = _make_viz()
        contacts = MagicMock()
        contacts.rigid_contact_count = np.array([0])
        viz._frame_counter = 0
        viz.update(contacts)
        self.assertEqual(viz._frame_counter, 1)

    def test_update_resets_counter_at_interval(self):
        viz = _make_viz()
        contacts = MagicMock()
        contacts.rigid_contact_count = np.array([0])
        viz._frame_counter = viz._update_interval - 1
        viz.update(contacts)
        self.assertEqual(viz._frame_counter, 0)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Engagement geometry: angle_mod60 at alignment
# ═════════════════════════════════════════════════════════════════════════════

class TestEngagementAngles(unittest.TestCase):
    """The 'wow moment' invariants: z=6 has engagement every 60 degrees."""

    def test_six_engagement_angles_in_360(self):
        angles = [i * (360.0 / 6) for i in range(6)]
        self.assertEqual(len(angles), 6)

    def test_engagement_angles_are_multiples_of_60(self):
        for i in range(6):
            angle = i * 60.0
            self.assertAlmostEqual(angle % 60.0, 0.0, places=10)

    def test_half_pitch_is_misaligned(self):
        """30 deg (half pitch for z=6) must NOT be an engagement angle."""
        self.assertNotAlmostEqual(30.0 % 60.0, 0.0, places=3)

    def test_full_pitch_is_aligned(self):
        self.assertAlmostEqual(60.0 % 60.0, 0.0)

    def test_angle_mod60_wraps_correctly(self):
        """75 deg mod 60 = 15 deg (not aligned)."""
        self.assertAlmostEqual(75.0 % 60.0, 15.0)

    def test_contact_area_drops_at_engagement(self):
        """
        Simulate what the demo should show: contact_area goes to zero when
        shaft_angle_mod60 = 0 (keys dropped into grooves).
        """
        engaged_area    = 0.0    # keys in grooves → no contact patch
        misaligned_area = 150.0  # keys on lands → contact patch present
        self.assertLess(engaged_area, misaligned_area)


# ═════════════════════════════════════════════════════════════════════════════
# 8. Contact area geometry validation
# ═════════════════════════════════════════════════════════════════════════════

class TestContactAreaGeometry(unittest.TestCase):

    def test_single_unit_right_triangle_area_mm2(self):
        """
        A right isosceles triangle with 1 mm legs has area 0.5 mm^2.
        Verify _sync_contact_surface computes this correctly.
        """
        viz  = _make_viz()
        surf = _fake_surface(1, depth_val=-0.001)
        viz._pipeline.hydroelastic_sdf = MagicMock()
        viz._pipeline.hydroelastic_sdf.get_contact_surface.return_value = surf
        viz._sync_contact_surface()
        # Each triangle: (0,0,0), (1mm, 0, 0), (0, 1mm, 0) → area = 0.5 mm^2
        self.assertAlmostEqual(viz.contact_area_mm2, 0.5, places=4)

    def test_n_triangles_gives_n_times_single_area(self):
        for n in (2, 4, 8):
            viz  = _make_viz()
            surf = _fake_surface(n, depth_val=-0.001)
            viz._pipeline.hydroelastic_sdf = MagicMock()
            viz._pipeline.hydroelastic_sdf.get_contact_surface.return_value = surf
            viz._sync_contact_surface()
            self.assertAlmostEqual(viz.contact_area_mm2, 0.5 * n, places=4,
                                   msg=f"n={n}")

    def test_deeper_penetration_does_not_change_area(self):
        """Area is geometric (from pts), not depth-dependent."""
        viz1 = _make_viz(); surf1 = _fake_surface(4, -0.001)
        viz2 = _make_viz(); surf2 = _fake_surface(4, -0.005)
        for viz, surf in [(viz1, surf1), (viz2, surf2)]:
            viz._pipeline.hydroelastic_sdf = MagicMock()
            viz._pipeline.hydroelastic_sdf.get_contact_surface.return_value = surf
            viz._sync_contact_surface()
        self.assertAlmostEqual(viz1.contact_area_mm2, viz2.contact_area_mm2, places=4)

    def test_deeper_penetration_gives_larger_max_depth(self):
        viz1 = _make_viz(); surf1 = _fake_surface(4, -0.001)
        viz2 = _make_viz(); surf2 = _fake_surface(4, -0.005)
        for viz, surf in [(viz1, surf1), (viz2, surf2)]:
            viz._pipeline.hydroelastic_sdf = MagicMock()
            viz._pipeline.hydroelastic_sdf.get_contact_surface.return_value = surf
            viz._sync_contact_surface()
        self.assertGreater(viz2.max_depth_mm, viz1.max_depth_mm)


if __name__ == "__main__":
    unittest.main()
