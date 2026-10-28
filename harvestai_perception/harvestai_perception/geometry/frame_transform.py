"""Coordinate transforms between the OAK camera and HarvestAI frame."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameConfig:
    """Physical HarvestAI frame dimensions in metres."""

    height: float = 0.22

    # Inner usable square is 16 cm x 16 cm.
    inner_half_width: float = 0.08
    inner_half_length: float = 0.08

    # Temporary assumption:
    # camera optical centre is at the top-centre of the frame.
    camera_x: float = 0.0
    camera_y: float = 0.0
    camera_z: float = 0.22


DEFAULT_FRAME = FrameConfig()


def camera_to_frame(
    x_c: float,
    y_c: float,
    z_c: float,
    config: FrameConfig = DEFAULT_FRAME,
) -> tuple[float, float, float]:
    """Convert OAK optical-frame coordinates to HarvestAI frame coordinates.

    Camera optical frame:
        +X -> image right
        +Y -> image down
        +Z -> forward from camera (downwards when mounted above frame)

    HarvestAI frame:
        +X -> right
        +Y -> forward
        +Z -> upwards from frame base

    Assumes the camera is centred at the top and points vertically down.
    """

    x_f = config.camera_x + x_c
    y_f = config.camera_y - y_c
    z_f = config.camera_z - z_c

    return x_f, y_f, z_f


def inside_working_volume(
    x_f: float,
    y_f: float,
    z_f: float,
    config: FrameConfig = DEFAULT_FRAME,
) -> bool:
    """Return True when the point lies inside the inner frame volume."""

    return (
        -config.inner_half_width <= x_f <= config.inner_half_width
        and -config.inner_half_length <= y_f <= config.inner_half_length
        and 0.0 <= z_f <= config.height
    )
