"""Geometry helpers for converting image pixels to OAK camera coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float


def pixel_to_camera(
    u: float,
    v: float,
    depth_mm: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float] | None:
    """Convert one RGB pixel and depth value into camera coordinates.

    Camera optical frame:
        X -> image right
        Y -> image down
        Z -> forward from camera

    Returns:
        (x_c, y_c, z_c) in metres.
    """

    if depth_mm <= 0:
        return None

    z_c = depth_mm / 1000.0

    x_c = (u - intrinsics.cx) * z_c / intrinsics.fx
    y_c = (v - intrinsics.cy) * z_c / intrinsics.fy

    return x_c, y_c, z_c


def bbox_center(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[int, int]:
    """Return the centre pixel of a YOLO bounding box."""

    u = int(round((x1 + x2) / 2.0))
    v = int(round((y1 + y2) / 2.0))

    return u, v


def median_depth_at(
    depth_mm: np.ndarray,
    u: int,
    v: int,
    radius: int = 2,
) -> float | None:
    """Return robust median depth around a pixel.

    Zero depth values are ignored because they mean invalid/no measurement.
    """

    h, w = depth_mm.shape[:2]

    x1 = max(0, u - radius)
    x2 = min(w, u + radius + 1)
    y1 = max(0, v - radius)
    y2 = min(h, v + radius + 1)

    patch = depth_mm[y1:y2, x1:x2]

    valid = patch[patch > 0]

    if valid.size == 0:
        return None

    return float(np.median(valid))

def bbox_to_camera(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float] | None:
    """Convert a detected bounding box to a 3D camera-frame position."""

    u, v = bbox_center(x1, y1, x2, y2)

    depth = median_depth_at(
        depth_mm,
        u,
        v,
        radius=2,
    )

    if depth is None:
        return None

    return pixel_to_camera(
        u,
        v,
        depth,
        intrinsics,
    )
