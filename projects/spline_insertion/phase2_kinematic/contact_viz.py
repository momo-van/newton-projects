# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
SplineContactViz -- hydroelastic contact patch visualizer for Newton ViewerGL.

Adapted from examples/rj45_hydro/hydro_contact_viz.py for the spline
insertion demo.  Rendering stays entirely inside Newton ViewerGL:

* Pressure-coloured wireframe patch via ``viewer.log_lines``
* Live ImGui metrics panel (face count, contact area, max depth)
* Tweakable parameters: opacity, density step, gamma

Jet colormap convention: red = high pressure (small depth), blue = low pressure.
"""

from __future__ import annotations

import numpy as np
import warp as wp

DEFAULT_UPDATE_INTERVAL: int = 5


class SplineContactViz:
    """
    Pressure-coloured hydroelastic contact patch visualizer for Newton ViewerGL.

    Parameters
    ----------
    collision_pipeline:
        A ``newton.CollisionPipeline`` built with
        ``HydroelasticSDF.Config(output_contact_surface=True)``.
    model:
        The ``newton.Model`` -- used for GPU device placement.
    viewer:
        A ``newton.ViewerGL`` instance.
    update_interval:
        Frames between GPU->CPU readbacks.
    """

    def __init__(
        self,
        collision_pipeline,
        model,
        viewer,
        update_interval: int = DEFAULT_UPDATE_INTERVAL,
    ) -> None:
        self._pipeline        = collision_pipeline
        self._model           = model
        self._viewer          = viewer
        self._update_interval = update_interval
        self._frame_counter   = 0

        # ---- Public metrics ----
        self.hydro_face_count:      int   = 0
        self.reduced_contact_count: int   = 0
        self.max_depth_mm:          float = 0.0
        self.contact_area_mm2:      float = 0.0

        # ---- Raw surface cache ----
        self._raw_pts:     np.ndarray | None = None
        self._raw_depths:  np.ndarray | None = None
        self._raw_n_faces: int = 0

        # ---- GPU line arrays ----
        self._patch_starts: wp.array | None = None
        self._patch_ends:   wp.array | None = None
        self._patch_colors: wp.array | None = None

        # ---- Visualization parameters ----
        self.show_patch:   bool  = True
        self.opacity:      float = 0.65
        self.density_step: int   = 1
        self.gamma:        float = 0.5

        if hasattr(viewer, "show_hydro_contact_surface"):
            viewer.show_hydro_contact_surface = False

        viewer.register_ui_callback(self._imgui_panel, position="side")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def update(self, contacts) -> None:
        """
        Refresh metrics and rebuild patch arrays.

        Call once per frame after ``collision_pipeline.collide()`` and
        before ``viewer.begin_frame()``.
        """
        try:
            rcc = contacts.rigid_contact_count
            self.reduced_contact_count = (
                int(rcc.numpy()[0]) if hasattr(rcc, "numpy") else int(rcc)
            )
        except Exception:
            self.reduced_contact_count = 0

        self._frame_counter += 1
        if self._frame_counter >= self._update_interval:
            self._frame_counter = 0
            self._sync_contact_surface()

    def render(self) -> None:
        """
        Emit the contact patch line segments to the viewer.

        Must be called inside a ``viewer.begin_frame()`` / ``viewer.end_frame()`` block.
        """
        if self.show_patch and self._patch_starts is not None:
            self._viewer.log_lines(
                "/hydro_contact_patch",
                self._patch_starts,
                self._patch_ends,
                self._patch_colors,
            )
        else:
            self._viewer.log_lines("/hydro_contact_patch", None, None, None)

    # -----------------------------------------------------------------------
    # Internal sync
    # -----------------------------------------------------------------------

    def _sync_contact_surface(self) -> None:
        hydro = self._pipeline.hydroelastic_sdf
        if hydro is None:
            self._clear_surface()
            return
        try:
            surface = hydro.get_contact_surface()
            n_faces = int(surface.face_contact_count.numpy()[0])
            self.hydro_face_count = n_faces

            if n_faces > 0:
                depths = surface.contact_surface_depth.numpy()[:n_faces]
                pts    = surface.contact_surface_point.numpy()[:n_faces * 3]

                self.max_depth_mm = float(np.max(np.abs(depths))) * 1000.0

                v0 = pts[0::3]; v1 = pts[1::3]; v2 = pts[2::3]
                cross = np.cross(v1 - v0, v2 - v0)
                self.contact_area_mm2 = float(
                    np.sum(0.5 * np.linalg.norm(cross, axis=1))
                ) * 1.0e6

                self._raw_pts     = pts
                self._raw_depths  = depths
                self._raw_n_faces = n_faces
                self._rebuild_patch()
            else:
                self._clear_surface()

        except Exception as exc:
            print(f"[SplineContactViz] surface read error: {exc}")
            self._clear_surface()

    def _clear_surface(self) -> None:
        self.hydro_face_count  = 0
        self.max_depth_mm      = 0.0
        self.contact_area_mm2  = 0.0
        self._raw_pts          = None
        self._raw_depths       = None
        self._raw_n_faces      = 0
        self._patch_starts = self._patch_ends = self._patch_colors = None

    def _rebuild_patch(self) -> None:
        if self._raw_pts is None or self._raw_n_faces == 0:
            self._patch_starts = self._patch_ends = self._patch_colors = None
            return

        pts  = self._raw_pts
        deps = self._raw_depths
        n    = self._raw_n_faces
        step = max(1, self.density_step)

        pts_3d  = pts.reshape(n, 3, 3)
        pts_sub = pts_3d[::step].reshape(-1, 3).astype(np.float32)
        d_sub   = np.abs(deps[::step])
        n_sub   = len(d_sub)

        if n_sub == 0:
            self._patch_starts = self._patch_ends = self._patch_colors = None
            return

        d_min = float(d_sub.min())
        d_max = float(d_sub.max())
        t = np.clip(
            (d_max - d_sub) / (d_max - d_min + 1e-12), 0.0, 1.0
        ).astype(np.float32)
        t = np.power(t, self.gamma)

        face_colors = self._jet(t) * np.float32(self.opacity)
        edge_colors = np.tile(face_colors, (3, 1))

        v0, v1, v2 = pts_sub[0::3], pts_sub[1::3], pts_sub[2::3]
        starts = np.vstack([v0, v1, v2]).astype(np.float32)
        ends   = np.vstack([v1, v2, v0]).astype(np.float32)

        dev = self._model.device
        self._patch_starts = wp.array(starts,                          dtype=wp.vec3, device=dev)
        self._patch_ends   = wp.array(ends,                            dtype=wp.vec3, device=dev)
        self._patch_colors = wp.array(edge_colors.astype(np.float32), dtype=wp.vec3, device=dev)

    # -----------------------------------------------------------------------
    # Jet colormap
    # -----------------------------------------------------------------------

    @staticmethod
    def jet(t: np.ndarray) -> np.ndarray:
        """Jet colormap: blue (t=0, low pressure) to red (t=1, high pressure)."""
        ramp_t = np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=np.float32)
        ramp_r = np.array([0.00, 0.00, 0.00, 1.00, 1.00], dtype=np.float32)
        ramp_g = np.array([0.00, 0.50, 1.00, 1.00, 0.00], dtype=np.float32)
        ramp_b = np.array([0.60, 1.00, 0.50, 0.00, 0.00], dtype=np.float32)
        r = np.interp(t, ramp_t, ramp_r).astype(np.float32)
        g = np.interp(t, ramp_t, ramp_g).astype(np.float32)
        b = np.interp(t, ramp_t, ramp_b).astype(np.float32)
        return np.stack([r, g, b], axis=1)

    @staticmethod
    def _jet(t: np.ndarray) -> np.ndarray:
        return SplineContactViz.jet(t)

    # -----------------------------------------------------------------------
    # ImGui panel
    # -----------------------------------------------------------------------

    def _imgui_panel(self, imgui) -> None:
        imgui.separator()
        imgui.text("Hydro Contact Patch")
        imgui.separator()
        imgui.spacing()

        ch, show = imgui.checkbox("Show contact patch", self.show_patch)
        if ch:
            self.show_patch = show

        if self.show_patch:
            imgui.spacing()
            imgui.push_item_width(180)

            ch_o, val_o = imgui.slider_float(
                "Opacity##patch", self.opacity, 0.0, 1.0, "%.2f"
            )
            if ch_o:
                self.opacity = float(val_o)
                self._rebuild_patch()

            ch_s, val_s = imgui.slider_int(
                "Density (step)##patch", self.density_step, 1, 20
            )
            if ch_s:
                self.density_step = max(1, int(val_s))
                self._rebuild_patch()

            ch_g, val_g = imgui.slider_float(
                "Gamma##patch", self.gamma, 0.1, 3.0, "%.2f"
            )
            if ch_g:
                self.gamma = float(val_g)
                self._rebuild_patch()

            imgui.pop_item_width()

        imgui.spacing()
        imgui.separator()
        imgui.text("Contact Metrics")
        imgui.separator()
        imgui.spacing()

        if self.reduced_contact_count > 0 or self.hydro_face_count > 0:
            imgui.text(f"  Contacts -> solver : {self.reduced_contact_count}")
            if self.hydro_face_count > 0:
                imgui.text_colored(
                    (0.3, 1.0, 0.4, 1.0),
                    f"  Hydro faces        : {self.hydro_face_count}",
                )
                imgui.text(f"  Max depth          : {self.max_depth_mm:.3f} mm")
                imgui.text(f"  Contact area       : {self.contact_area_mm2:.2f} mm^2")
                imgui.spacing()
                reduction = 100.0 * (
                    1.0 - self.reduced_contact_count / max(self.hydro_face_count, 1)
                )
                imgui.text_colored(
                    (1.0, 0.85, 0.2, 1.0),
                    f"  Reduction          : {reduction:.0f}%",
                )
            else:
                imgui.text_colored(
                    (1.0, 0.75, 0.2, 1.0),
                    "  Rotate shaft to engage",
                )
        else:
            imgui.text_colored(
                (0.55, 0.55, 0.55, 1.0),
                "  No contact detected",
            )
