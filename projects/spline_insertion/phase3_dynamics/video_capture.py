# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Window-level screen recorder for the Newton Viewer.

Captures a named Win32 window using PrintWindow(PW_RENDERFULLCONTENT), which
reads directly from the DWM compositor framebuffer.  The window does NOT need
to be active or on top -- it can be fully occluded by other applications.

Records to MP4 using OpenCV.

Dependencies: pywin32, opencv-python  (pip install pywin32 opencv-python)

Usage
-----
::

    from video_capture import ScreenRecorder

    rec = ScreenRecorder(target_fps=30, window_title="Newton Viewer")
    rec.start()
    # ... run simulation loop ...
    rec.stop()
    rec.save("output.mp4")
"""

from __future__ import annotations

import ctypes
import pathlib
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import win32gui
import win32ui

# PW_RENDERFULLCONTENT  (Windows 8.1+)
# Forces DWM to composite hardware-accelerated (OpenGL/D3D) window content
# into the target DC.  Without this flag, PrintWindow only captures the GDI
# layer and OpenGL windows appear black.
_PW_RENDERFULLCONTENT: int = 0x00000002


# ── Low-level helpers ─────────────────────────────────────────────────────────

def find_window_hwnd(title_substr: str) -> Optional[int]:
    """
    Return the HWND of the first visible window whose title contains
    *title_substr* (case-insensitive).  Returns None if not found.
    """
    results: list[int] = []

    def _cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title_substr.lower() in title.lower():
                results.append(hwnd)

    win32gui.EnumWindows(_cb, None)
    return results[0] if results else None


def find_window_rect(title_substr: str) -> Optional[Tuple[int, int, int, int]]:
    """Return (left, top, width, height) for the first matching window."""
    hwnd = find_window_hwnd(title_substr)
    if hwnd is None:
        return None
    l, t, r, b = win32gui.GetWindowRect(hwnd)
    return (l, t, r - l, b - t)


def capture_hwnd(hwnd: int) -> Optional[np.ndarray]:
    """
    Capture *hwnd* into a BGR numpy array using PrintWindow.

    Works even when the window is minimised, hidden behind other windows, or
    on a non-primary monitor.  Returns None on failure.
    """
    try:
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        w, h = r - l, b - t
        if w <= 0 or h <= 0:
            return None

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc  = win32ui.CreateDCFromHandle(hwnd_dc)
        mem_dc  = mfc_dc.CreateCompatibleDC()
        bmp     = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc_dc, w, h)
        mem_dc.SelectObject(bmp)

        result = ctypes.windll.user32.PrintWindow(
            hwnd, mem_dc.GetSafeHdc(), _PW_RENDERFULLCONTENT
        )
        if not result:
            # Fallback: retry without the full-content flag (GDI-only windows)
            ctypes.windll.user32.PrintWindow(hwnd, mem_dc.GetSafeHdc(), 0)

        bits = bmp.GetBitmapBits(True)
        img  = np.frombuffer(bits, dtype=np.uint8).reshape(h, w, 4)  # BGRA

        win32gui.DeleteObject(bmp.GetHandle())
        mem_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        return img[:, :, :3].copy()   # BGR, owned buffer

    except Exception:
        return None


# ── Recorder ─────────────────────────────────────────────────────────────────

class ScreenRecorder:
    """
    Background-thread window recorder.

    Captures the target window by title using PrintWindow -- the window does
    not need to be the active/foreground window.

    Parameters
    ----------
    target_fps:
        Capture rate.  Frames are dropped (not duplicated) when the machine
        cannot keep up.
    window_title:
        Substring of the target window title (case-insensitive).  The
        recorder retries every frame until the window appears, so it is safe
        to call start() before the window opens.
    """

    def __init__(
        self,
        target_fps: int = 30,
        window_title: str = "Newton Viewer",
        region: Optional[Tuple[int, int, int, int]] = None,  # kept for API compat
    ) -> None:
        self._fps    = target_fps
        self._title  = window_title
        self._hwnd:  Optional[int] = None

        self._frames:  list[np.ndarray] = []
        self._running  = False
        self._thread:  Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin capturing frames in a background thread."""
        if self._running:
            return
        self._frames.clear()
        self._hwnd    = None
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing and join the background thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def save(self, path: str | pathlib.Path, codec: str = "mp4v") -> pathlib.Path:
        """Write captured frames to an MP4 file at target_fps.  Returns the output path."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        n = len(self._frames)
        if n == 0:
            raise RuntimeError("No frames captured -- did you call start()/stop()?")

        h, w = self._frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, self._fps, (w, h))
        for frame in self._frames:
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
        writer.release()
        duration = n / self._fps
        print(f"[VideoCapture] saved {n} frames @ {self._fps} fps -> {path}  ({duration:.1f} s)")
        return path

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        interval   = 1.0 / self._fps
        next_tick  = time.perf_counter()

        while self._running:
            # Refresh HWND every iteration -- handles window (re)creation
            if self._hwnd is None or not win32gui.IsWindow(self._hwnd):
                self._hwnd = find_window_hwnd(self._title)

            now = time.perf_counter()
            if now >= next_tick:
                if self._hwnd is not None:
                    frame = capture_hwnd(self._hwnd)
                    if frame is not None:
                        self._frames.append(frame)
                next_tick += interval
                # Resync if we've fallen behind (drop rather than duplicate)
                if time.perf_counter() > next_tick:
                    next_tick = time.perf_counter()
            else:
                time.sleep(max(0.0, next_tick - now - 0.001))
