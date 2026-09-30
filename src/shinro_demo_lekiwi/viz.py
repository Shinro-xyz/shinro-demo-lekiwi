"""Shared visualization for the LeKiwi demos: a bird's-eye tracking view.

Produces a composite frame per tick:

    +--------------------------+-----------------+
    |  MuJoCo scene (top-down) |  bird's-eye x-y |
    |                          |  reference vs   |
    |                          |  base path      |
    +--------------------------+-----------------+
                               |  tracking error |
                               |  vs time        |
                               +-----------------+

The x-y panel plots the B-spline **control polygon** (what the curve was
dictated by), the sampled reference curve, and the base's actual path, so the
tracking quality is obvious at a glance.
"""

from __future__ import annotations

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (use() must run before pyplot imports)


def birdseye_camera(camera, lookat=(0.0, 0.0, 0.0), distance=3.4):
    """Point a MuJoCo camera straight down for a top-down (bird's-eye) view."""
    camera.azimuth = 0.0
    camera.elevation = -89.0
    camera.distance = distance
    camera.lookat[:] = lookat
    return camera


def _resize_to_height(rgb: np.ndarray, height: int) -> np.ndarray:
    if rgb.shape[0] == height:
        return rgb
    img = Image.fromarray(rgb)
    width = max(1, round(img.width * height / img.height))
    return np.asarray(img.resize((width, height), Image.BILINEAR))


def compose_h(panels, height: int = 720) -> np.ndarray:
    """Horizontally stack RGB panels, each resized to a common height."""
    return np.concatenate([_resize_to_height(np.asarray(p)[..., :3], height) for p in panels], axis=1)


class TrackingPanels:
    """A matplotlib figure with a bird's-eye path panel and a tracking-error panel.

    Call :meth:`update` each tick and :meth:`frame` to grab the panel as an RGB
    array (to be composed with the 3D scene).
    """

    def __init__(
        self,
        reference_xy,
        duration: float,
        control_points=None,
        path_wh: tuple[float, float] = (7.2, 7.2),
        err_wh: tuple[float, float] = (3.6, 7.2),
        dpi: int = 100,
    ):
        ref = np.asarray(reference_xy, dtype=np.float64)
        self.path_wh, self.err_wh, self.dpi = path_wh, err_wh, dpi
        self._t: list[float] = []
        self._err: list[float] = []
        self._base: list[np.ndarray] = []

        self.fig = plt.figure(figsize=(path_wh[0] + err_wh[0], path_wh[1]), dpi=dpi)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[path_wh[0], err_wh[0]], wspace=0.28)
        self.ax_path = self.fig.add_subplot(gs[0, 0])
        self.ax_err = self.fig.add_subplot(gs[0, 1])

        # ── bird's-eye x-y panel ──────────────────────────────────────────
        ax = self.ax_path
        if control_points is not None:
            cp = np.asarray(control_points, dtype=np.float64)
            ax.plot(cp[:, 0], cp[:, 1], "--", color="0.55", lw=1.0, zorder=1, label="control polygon")
            ax.plot(cp[:, 0], cp[:, 1], "x", color="0.45", ms=5, zorder=2)
        ax.plot(ref[:, 0], ref[:, 1], "--", color="tab:blue", lw=1.6, zorder=3, label="reference (B-spline)")
        (self.line_base,) = ax.plot([], [], "-", color="tab:orange", lw=2.0, zorder=4, label="base path")
        (self.pt_base,) = ax.plot([], [], "o", color="tab:orange", ms=8, zorder=5)
        (self.pt_ref,) = ax.plot([], [], "o", color="tab:blue", ms=7, zorder=5)
        all_xy = ref[:, :2] if control_points is None else np.vstack([ref[:, :2], np.asarray(control_points)[:, :2]])
        lo = all_xy.min(axis=0) - 0.25
        hi = all_xy.max(axis=0) + 0.25
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("bird's-eye: reference vs base", fontsize=11)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)

        # ── tracking-error panel ──────────────────────────────────────────
        eax = self.ax_err
        (self.line_err,) = eax.plot([], [], "-", color="tab:red", lw=1.6)
        eax.set_xlim(0.0, duration)
        eax.set_ylim(bottom=0.0)
        eax.set_xlabel("t [s]")
        eax.set_ylabel("|base − ref| [m]")
        eax.set_title("tracking error", fontsize=11)
        eax.grid(True, alpha=0.3)

    def update(self, base_xy, reference_xy, t: float, error: float) -> None:
        self._base.append(np.asarray(base_xy, dtype=np.float64)[:2])
        self._t.append(t)
        self._err.append(error)

        base = np.array(self._base)
        self.line_base.set_data(base[:, 0], base[:, 1])
        self.pt_base.set_data([base[-1, 0]], [base[-1, 1]])
        self.pt_ref.set_data([np.asarray(reference_xy)[0]], [np.asarray(reference_xy)[1]])

        self.line_err.set_data(self._t, self._err)
        self.ax_err.set_ylim(0.0, max(0.02, max(self._err) * 1.15))

    def frame(self) -> np.ndarray:
        """Render the two panels to an RGB array (height = path panel height)."""
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return rgba[..., :3]
