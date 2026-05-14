# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for HydroContactViz.

These tests cover the pure-Python / NumPy logic that does not require a GPU,
a Newton model, or a live viewer.  GPU-dependent paths (``update()``,
``render()``, ``_sync_contact_surface()``) are exercised via lightweight mocks
so that the test suite runs on any machine with NumPy installed.

Run
---
::

    pytest tests/test_hydro_contact_viz.py -v
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without a full Newton install
# ---------------------------------------------------------------------------

def _make_warp_stub() -> types.ModuleType:
    wp = types.ModuleType("warp")
    wp.array   = MagicMock(return_value=MagicMock())
    wp.vec3    = MagicMock()
    wp.float32 = float
    return wp


def _install_stubs() -> None:
    """Insert lightweight stubs for warp so the import succeeds without GPU."""
    if "warp" not in sys.modules:
        sys.modules["warp"] = _make_warp_stub()


_install_stubs()

# Now we can import the module under test
from hydro_contact_viz import HydroContactViz, DEFAULT_UPDATE_INTERVAL  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_viz() -> HydroContactViz:
    """Return a HydroContactViz wired to mock dependencies."""
    pipeline = MagicMock()
    pipeline.hydroelastic_sdf = None   # no surface by default

    model = MagicMock()
    model.device = "cpu"

    viewer = MagicMock()
    viewer.show_hydro_contact_surface = True   # should be set to False by __init__

    viz = HydroContactViz(pipeline, model, viewer)
    return viz


def _triangle_pts(n: int) -> np.ndarray:
    """
    Generate ``n`` unit triangles in the XY-plane as (n*3, 3) float32.

    Layout is interleaved: [v0_tri0, v1_tri0, v2_tri0, v0_tri1, ...],
    matching Newton's contact_surface_point array format.
    Each triangle has vertices (0,0,0), (1,0,0), (0,1,0) → area = 0.5 m².
    """
    one_tri = np.array([[0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0]], dtype=np.float32)
    return np.tile(one_tri, (n, 1))


# ---------------------------------------------------------------------------
# Colormap tests
# ---------------------------------------------------------------------------

class TestJetColormap(unittest.TestCase):

    def test_output_shape(self):
        t = np.linspace(0.0, 1.0, 50, dtype=np.float32)
        rgb = HydroContactViz.jet(t)
        self.assertEqual(rgb.shape, (50, 3))

    def test_output_dtype(self):
        t = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        rgb = HydroContactViz.jet(t)
        self.assertEqual(rgb.dtype, np.float32)

    def test_values_in_range(self):
        t = np.linspace(0.0, 1.0, 200, dtype=np.float32)
        rgb = HydroContactViz.jet(t)
        self.assertTrue(np.all(rgb >= 0.0), "RGB values must be >= 0")
        self.assertTrue(np.all(rgb <= 1.0), "RGB values must be <= 1")

    def test_blue_at_zero(self):
        """t=0 should map to blue (high blue, low red)."""
        rgb = HydroContactViz.jet(np.array([0.0], dtype=np.float32))
        r, g, b = rgb[0]
        self.assertGreater(float(b), float(r),
                           "t=0 should be blue-dominant (low pressure)")

    def test_red_at_one(self):
        """t=1 should map to red (high red, low blue)."""
        rgb = HydroContactViz.jet(np.array([1.0], dtype=np.float32))
        r, g, b = rgb[0]
        self.assertGreater(float(r), float(b),
                           "t=1 should be red-dominant (high pressure)")

    def test_single_value(self):
        """Scalar array should not raise."""
        rgb = HydroContactViz.jet(np.array([0.5], dtype=np.float32))
        self.assertEqual(rgb.shape, (1, 3))

    def test_monotone_red_channel(self):
        """Red channel should be monotonically non-decreasing for t in [0.5, 1]."""
        t   = np.linspace(0.5, 1.0, 100, dtype=np.float32)
        rgb = HydroContactViz.jet(t)
        red = rgb[:, 0]
        self.assertTrue(np.all(np.diff(red) >= -1e-6),
                        "Red channel should increase as t approaches 1")


# ---------------------------------------------------------------------------
# Pressure polarity tests
# ---------------------------------------------------------------------------

class TestPressurePolarity(unittest.TestCase):
    """
    Verify the depth→colour inversion: small abs_depth = high pressure = red.

    Newton's contact surface assigns small absolute depth values to triangles
    at the centre of the patch (highest pressure) and large values at the edges
    (lowest pressure).  The visualizer must map this correctly.
    """

    def _face_colors_for_depths(self, depths: np.ndarray) -> np.ndarray:
        """
        Compute the jet-mapped face colours for the given depth array directly,
        mirroring the inversion logic in HydroContactViz._rebuild_patch.
        """
        d_sub = np.abs(depths)
        d_min = float(d_sub.min())
        d_max = float(d_sub.max())
        t = np.clip(
            (d_max - d_sub) / (d_max - d_min + 1e-12), 0.0, 1.0
        ).astype(np.float32)
        t = np.power(t, 1.0)   # gamma=1 (linear)
        return HydroContactViz.jet(t)

    def test_high_pressure_is_red(self):
        """The triangle with smallest abs_depth should be the most red."""
        depths = np.array([0.001, 0.010], dtype=np.float32)
        face_colors = self._face_colors_for_depths(depths)
        r_high_p = float(face_colors[0, 0])   # red channel of high-pressure face
        r_low_p  = float(face_colors[1, 0])   # red channel of low-pressure face
        self.assertGreater(r_high_p, r_low_p,
                           "High-pressure face (small depth) must be more red")

    def test_low_pressure_is_blue(self):
        """The triangle with largest abs_depth should be the most blue."""
        depths = np.array([0.001, 0.010], dtype=np.float32)
        face_colors = self._face_colors_for_depths(depths)
        b_high_p = float(face_colors[0, 2])
        b_low_p  = float(face_colors[1, 2])
        self.assertGreater(b_low_p, b_high_p,
                           "Low-pressure face (large depth) must be more blue")


# ---------------------------------------------------------------------------
# Contact area computation tests
# ---------------------------------------------------------------------------

class TestContactAreaComputation(unittest.TestCase):
    """
    Verify the triangle-area formula used inside _sync_contact_surface.

    Each unit triangle (vertices at (0,0,0), (1,0,0), (0,1,0)) has area 0.5 m²
    = 0.5e6 mm².
    """

    def test_unit_triangle_area(self):
        n   = 4
        pts = _triangle_pts(n)
        v0  = pts[0::3]; v1 = pts[1::3]; v2 = pts[2::3]
        cross     = np.cross(v1 - v0, v2 - v0)
        tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
        area_mm2  = float(np.sum(tri_areas)) * 1.0e6
        expected  = n * 0.5 * 1.0e6
        self.assertAlmostEqual(area_mm2, expected, places=3)


# ---------------------------------------------------------------------------
# Defaults and initialisation tests
# ---------------------------------------------------------------------------

class TestDefaults(unittest.TestCase):

    def setUp(self):
        self.viz = _make_viz()

    def test_default_update_interval(self):
        self.assertEqual(DEFAULT_UPDATE_INTERVAL, 5)

    def test_show_patch_default_true(self):
        self.assertTrue(self.viz.show_patch)

    def test_opacity_default(self):
        self.assertAlmostEqual(self.viz.opacity, 0.65, places=2)

    def test_density_step_default(self):
        self.assertEqual(self.viz.density_step, 1)

    def test_gamma_default(self):
        self.assertAlmostEqual(self.viz.gamma, 0.5, places=2)

    def test_metrics_start_at_zero(self):
        self.assertEqual(self.viz.hydro_face_count,      0)
        self.assertEqual(self.viz.reduced_contact_count, 0)
        self.assertAlmostEqual(self.viz.max_depth_mm,     0.0)
        self.assertAlmostEqual(self.viz.contact_area_mm2, 0.0)

    def test_built_in_renderer_disabled(self):
        """__init__ must set viewer.show_hydro_contact_surface = False."""
        self.assertFalse(self.viz._viewer.show_hydro_contact_surface)

    def test_ui_callback_registered(self):
        self.viz._viewer.register_ui_callback.assert_called_once()


# ---------------------------------------------------------------------------
# Opacity scaling tests
# ---------------------------------------------------------------------------

class TestOpacityScaling(unittest.TestCase):
    """Verify that opacity scales RGB linearly (no GPU arrays needed)."""

    def _colors_at_opacity(self, opacity: float) -> np.ndarray:
        """Compute face colours at given opacity using the pure-numpy path."""
        depths = np.array([0.001, 0.005, 0.010], dtype=np.float32)
        d_sub  = np.abs(depths)
        d_min, d_max = float(d_sub.min()), float(d_sub.max())
        t = np.clip((d_max - d_sub) / (d_max - d_min + 1e-12), 0.0, 1.0
                    ).astype(np.float32)
        return HydroContactViz.jet(t) * np.float32(opacity)

    def test_zero_opacity_gives_black(self):
        colors = self._colors_at_opacity(0.0)
        self.assertTrue(np.allclose(colors, 0.0, atol=1e-6),
                        "Opacity 0 should yield all-black lines")

    def test_full_opacity_max_brightness(self):
        colors_full = self._colors_at_opacity(1.0)
        colors_half = self._colors_at_opacity(0.5)
        self.assertTrue(np.all(colors_full >= colors_half - 1e-6),
                        "Full opacity must be at least as bright as half opacity")


# ---------------------------------------------------------------------------
# _clear_surface tests
# ---------------------------------------------------------------------------

class TestClearSurface(unittest.TestCase):

    def setUp(self):
        self.viz = _make_viz()

    def test_clears_all_state(self):
        # Populate state artificially
        self.viz.hydro_face_count  = 999
        self.viz.max_depth_mm      = 3.5
        self.viz.contact_area_mm2  = 12.0
        self.viz._raw_pts          = np.zeros((6, 3))
        self.viz._raw_n_faces      = 2
        self.viz._patch_starts     = MagicMock()

        self.viz._clear_surface()

        self.assertEqual(self.viz.hydro_face_count,  0)
        self.assertAlmostEqual(self.viz.max_depth_mm, 0.0)
        self.assertAlmostEqual(self.viz.contact_area_mm2, 0.0)
        self.assertIsNone(self.viz._raw_pts)
        self.assertEqual(self.viz._raw_n_faces, 0)
        self.assertIsNone(self.viz._patch_starts)
        self.assertIsNone(self.viz._patch_ends)
        self.assertIsNone(self.viz._patch_colors)


# ---------------------------------------------------------------------------
# update() throttle tests
# ---------------------------------------------------------------------------

class TestUpdateThrottle(unittest.TestCase):

    def setUp(self):
        self.viz = _make_viz()
        self.viz._update_interval = 3

    def test_surface_sync_only_at_interval(self):
        contacts = MagicMock()
        contacts.rigid_contact_count = MagicMock()
        contacts.rigid_contact_count.numpy = MagicMock(return_value=np.array([42]))

        with patch.object(self.viz, "_sync_contact_surface") as mock_sync:
            self.viz.update(contacts)   # frame 1 — no sync
            self.viz.update(contacts)   # frame 2 — no sync
            self.viz.update(contacts)   # frame 3 — sync fires
            self.assertEqual(mock_sync.call_count, 1)

    def test_contact_count_read_every_frame(self):
        contacts = MagicMock()
        contacts.rigid_contact_count = MagicMock()
        contacts.rigid_contact_count.numpy = MagicMock(return_value=np.array([7]))

        with patch.object(self.viz, "_sync_contact_surface"):
            for _ in range(5):
                self.viz.update(contacts)

        self.assertEqual(self.viz.reduced_contact_count, 7)


# ---------------------------------------------------------------------------
# render() tests
# ---------------------------------------------------------------------------

class TestRender(unittest.TestCase):

    def setUp(self):
        self.viz = _make_viz()

    def test_clears_lines_when_no_patch(self):
        self.viz.show_patch     = True
        self.viz._patch_starts  = None
        self.viz.render()
        self.viz._viewer.log_lines.assert_called_with(
            "/hydro_contact_patch", None, None, None
        )

    def test_clears_lines_when_hidden(self):
        self.viz.show_patch    = False
        self.viz._patch_starts = MagicMock()   # has data but hidden
        self.viz.render()
        self.viz._viewer.log_lines.assert_called_with(
            "/hydro_contact_patch", None, None, None
        )

    def test_sends_lines_when_visible(self):
        self.viz.show_patch    = True
        self.viz._patch_starts = MagicMock()
        self.viz._patch_ends   = MagicMock()
        self.viz._patch_colors = MagicMock()
        self.viz.render()
        call_args = self.viz._viewer.log_lines.call_args
        self.assertEqual(call_args[0][0], "/hydro_contact_patch")
        self.assertIsNotNone(call_args[0][1])


if __name__ == "__main__":
    unittest.main()
