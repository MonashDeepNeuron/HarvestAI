"""OAK-D Pro RGB camera interface.

This module is intentionally independent of ROS so it can run directly on
the Jetson. It returns RGB numpy arrays that can be passed straight into the
HarvestAI strawberry detector.
"""

from __future__ import annotations

import cv2
import depthai as dai
import numpy as np


class OakCamera:
    """Small wrapper around the OAK-D Pro colour camera."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 15.0,
        enable_depth: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth

        self._pipeline = self._build_pipeline()
        self._device: dai.Device | None = None
        self._rgb_queue = None
        self._depth_queue = None

    def _build_pipeline(self) -> dai.Pipeline:
        """Create the DepthAI RGB pipeline."""
        pipeline = dai.Pipeline()

        camera = pipeline.create(dai.node.ColorCamera)
        rgb_out = pipeline.create(dai.node.XLinkOut)

        rgb_out.setStreamName("rgb")

        camera.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camera.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1080_P
        )
        camera.setPreviewSize(self.width, self.height)
        camera.setFps(self.fps)
        camera.setInterleaved(False)
        camera.setColorOrder(
            dai.ColorCameraProperties.ColorOrder.BGR
        )

        camera.preview.link(rgb_out.input)

        if self.enable_depth:
            left = pipeline.create(dai.node.MonoCamera)
            right = pipeline.create(dai.node.MonoCamera)
            stereo = pipeline.create(dai.node.StereoDepth)
            depth_out = pipeline.create(dai.node.XLinkOut)

            depth_out.setStreamName("depth")

            left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

            left.setResolution(
                dai.MonoCameraProperties.SensorResolution.THE_400_P
            )
            right.setResolution(
                dai.MonoCameraProperties.SensorResolution.THE_400_P
            )

            left.setFps(self.fps)
            right.setFps(self.fps)

            stereo.setDefaultProfilePreset(
                dai.node.StereoDepth.PresetMode.HIGH_DENSITY
            )
            stereo.setLeftRightCheck(True)

            # Align depth pixels to the RGB camera viewpoint.
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            stereo.setOutputSize(self.width, self.height)

            left.out.link(stereo.left)
            right.out.link(stereo.right)
            stereo.depth.link(depth_out.input)

        return pipeline

    def start(self) -> None:
        """Connect to the OAK device and start streaming."""
        if self._device is not None:
            return

        self._device = dai.Device(self._pipeline)

        self._rgb_queue = self._device.getOutputQueue(
            name="rgb",
            maxSize=2,
            blocking=False,
        )

        if self.enable_depth:
            self._depth_queue = self._device.getOutputQueue(
                name="depth",
                maxSize=2,
                blocking=False,
            )

    def get_intrinsics(self) -> tuple[float, float, float, float]:
        """Return RGB camera intrinsics for the configured output size.

        Returns:
            (fx, fy, cx, cy) in pixels.
        """

        if self._device is None:
            raise RuntimeError("Camera is not started. Call start() first.")

        calibration = self._device.readCalibration()

        matrix = np.array(
            calibration.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A,
                self.width,
                self.height,
            ),
            dtype=np.float64,
        )

        fx = float(matrix[0, 0])
        fy = float(matrix[1, 1])
        cx = float(matrix[0, 2])
        cy = float(matrix[1, 2])

        return fx, fy, cx, cy

    def read(self) -> np.ndarray | None:
        """Return the newest RGB frame, or None if no frame is ready."""
        if self._device is None or self._rgb_queue is None:
            raise RuntimeError("Camera is not started. Call start() first.")

        packet = self._rgb_queue.tryGet()

        if packet is None:
            return None

        # DepthAI/OpenCV gives us BGR.
        # detector.py expects an RGB numpy array.
        bgr = packet.getCvFrame()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        return rgb

    def read_rgbd(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return RGB plus aligned depth in millimetres."""

        if not self.enable_depth:
            raise RuntimeError(
                "Depth is disabled. Create OakCamera(enable_depth=True)."
            )

        if (
            self._device is None
            or self._rgb_queue is None
            or self._depth_queue is None
        ):
            raise RuntimeError("Camera is not started. Call start() first.")

        rgb_packet = self._rgb_queue.tryGet()
        depth_packet = self._depth_queue.tryGet()

        if rgb_packet is None or depth_packet is None:
            return None

        bgr = rgb_packet.getCvFrame()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # uint16 array; each value is depth in millimetres.
        depth_mm = depth_packet.getFrame()

        return rgb, depth_mm

    def close(self) -> None:
        """Release the OAK device."""
        self._rgb_queue = None
        self._depth_queue = None

        if self._device is not None:
            self._device.close()
            self._device = None

    def __enter__(self) -> "OakCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def main() -> None:
    """Simple standalone camera test."""
    print("Starting OAK-D Pro. Press q to quit.")

    with OakCamera() as camera:
        while True:
            rgb = camera.read()

            if rgb is None:
                continue

            # OpenCV display expects BGR.
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("HarvestAI - OAK RGB", bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
