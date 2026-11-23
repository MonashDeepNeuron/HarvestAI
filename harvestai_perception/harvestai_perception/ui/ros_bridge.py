"""Background ROS 2 subscriber that lets the Gradio UI show perception_node output.

The UI never opens the camera in ROS mode: perception_node owns it, and this
listener just keeps the latest berries + annotated image for the UI to poll.
"""

from __future__ import annotations

import threading

import numpy as np

RIPENESS_NAMES = {2: "ripe", 1: "turning", 0: "unknown", -1: "unripe"}


class RosListener:
    def __init__(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        from harvestai_interfaces.msg import BerryDetectionList

        if not rclpy.ok():
            rclpy.init()

        self._lock = threading.Lock()
        self._berries = None
        self._image = None

        self._node = rclpy.create_node('harvestai_ui')
        self._node.create_subscription(BerryDetectionList, 'berries', self._on_berries, 10)
        self._node.create_subscription(
            Image, 'berries/image', self._on_image, qos_profile_sensor_data)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        threading.Thread(target=self._executor.spin, daemon=True).start()

    def _on_berries(self, msg):
        with self._lock:
            self._berries = msg

    def _on_image(self, msg):
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        with self._lock:
            self._image = image.copy()

    def latest(self):
        """(BerryDetectionList or None, RGB image or None)."""
        with self._lock:
            return self._berries, self._image


_listener: RosListener | None = None


def get_listener() -> RosListener:
    global _listener
    if _listener is None:
        _listener = RosListener()
    return _listener
