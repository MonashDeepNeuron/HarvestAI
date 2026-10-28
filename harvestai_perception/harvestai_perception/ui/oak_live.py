"""Live strawberry detection using an OAK-D Pro camera."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

# Allow this script to import oak_camera.py from the parent package directory.
PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from oak_camera import OakCamera  # noqa: E402
from detector import DEFAULT_MODEL_PATH, get_detector  # noqa: E402

from geometry.camera_geometry import (
    CameraIntrinsics,
    bbox_to_camera,
)

from geometry.frame_transform import (
    camera_to_frame,
    inside_working_volume,
)

def main() -> None:
    print(f"Loading model: {DEFAULT_MODEL_PATH.name}")

    detector = get_detector(DEFAULT_MODEL_PATH)

    print("Starting OAK-D Pro...")
    print("Press q to quit.")

    with OakCamera(
        width=640,
        height=480,
        fps=15,
        enable_depth=True,
    ) as camera:

        fx, fy, cx, cy = camera.get_intrinsics()

        intrinsics = CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

        print(
            f"Camera intrinsics: "
            f"fx={fx:.2f}, fy={fy:.2f}, "
            f"cx={cx:.2f}, cy={cy:.2f}"
        )

        while True:
            rgbd = camera.read_rgbd()

            if rgbd is None:
                continue

            rgb, depth_mm = rgbd

            result = detector.detect(
                rgb,
                conf=0.5,
                ripeness_engine=None,
            )

            positions = []

            for detection in result.detections:
                position = bbox_to_camera(
                    detection.x1,
                    detection.y1,
                    detection.x2,
                    detection.y2,
                    depth_mm,
                    intrinsics,
                )

                positions.append(position)

            # detector returns RGB; OpenCV display expects BGR.
            display = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)

            for detection, position in zip(result.detections, positions):
                if position is None:
                    continue

                x_c, y_c, z_c = position

                x_f, y_f, z_f = camera_to_frame(
                    x_c,
                    y_c,
                    z_c,
                )

                inside = inside_working_volume(
                    x_f,
                    y_f,
                    z_f,
                )

                label = (
                    f"C({x_c:.3f},{y_c:.3f},{z_c:.3f}) "
                    f"F({x_f:.3f},{y_f:.3f},{z_f:.3f}) "
                    f"{'IN' if inside else 'OUT'}"
                )

                x = int(detection.x1)
                y = max(20, int(detection.y2) + 20)

                cv2.putText(
                    display,
                    label,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                display,
                f"Strawberries: {result.count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("HarvestAI - OAK Strawberry Detection", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    