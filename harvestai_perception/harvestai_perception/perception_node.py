"""ROS 2 node: OAK-D Pro -> strawberry detections with 3D positions.

Publishes:
    berries        harvestai_interfaces/BerryDetectionList  (camera optical frame, metres)
    berries/image  sensor_msgs/Image (rgb8, detections drawn on)

Run:
    ros2 run harvestai_perception perception_node
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from harvestai_interfaces.msg import BerryDetection, BerryDetectionList
from harvestai_perception.detector import DEFAULT_MODEL_PATH, get_detector
from harvestai_perception.geometry.camera_geometry import CameraIntrinsics, bbox_to_camera
from harvestai_perception.oak_camera import OakCamera
from harvestai_perception.ripeness import COLOUR_RULE_KEY, get_engine

RIPENESS_TO_MSG = {
    "ripe": BerryDetection.RIPE,
    "turning": BerryDetection.TURNING,
    "unripe": BerryDetection.UNRIPE,
}


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.declare_parameter('model_path', str(DEFAULT_MODEL_PATH))
        self.declare_parameter('ripeness_engine', COLOUR_RULE_KEY)
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('fps', 5.0)
        self.declare_parameter('camera_frame', 'oak_rgb_camera_optical_frame')
        self.declare_parameter('publish_image', True)

        model_path = self.get_parameter('model_path').value
        engine_key = self.get_parameter('ripeness_engine').value
        self._conf = float(self.get_parameter('confidence').value)
        self._frame_id = self.get_parameter('camera_frame').value
        self._publish_image = bool(self.get_parameter('publish_image').value)
        fps = float(self.get_parameter('fps').value)

        self.get_logger().info(f'Loading model: {model_path}')
        self._detector = get_detector(model_path)
        self._engine = get_engine(engine_key)

        self.get_logger().info('Starting OAK-D Pro...')
        self._camera = OakCamera(width=640, height=480, fps=15, enable_depth=True)
        self._camera.start()
        fx, fy, cx, cy = self._camera.get_intrinsics()
        self._intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
        self.get_logger().info(f'Intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}')

        self._berry_pub = self.create_publisher(BerryDetectionList, 'berries', 10)
        self._image_pub = self.create_publisher(Image, 'berries/image', qos_profile_sensor_data)
        self.create_timer(1.0 / max(fps, 0.5), self._on_timer)

    def _on_timer(self):
        rgbd = self._camera.read_rgbd()
        if rgbd is None:
            return
        rgb, depth_mm = rgbd

        result = self._detector.detect(rgb, conf=self._conf, ripeness_engine=self._engine)
        stamp = self.get_clock().now().to_msg()

        msg = BerryDetectionList()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id

        for d in result.detections:
            position = bbox_to_camera(d.x1, d.y1, d.x2, d.y2, depth_mm, self._intrinsics)
            if position is None:
                continue  # no valid depth at the box centre

            berry = BerryDetection()
            berry.position.x, berry.position.y, berry.position.z = position
            berry.ripeness = RIPENESS_TO_MSG.get(d.ripeness, BerryDetection.UNKNOWN)
            berry.confidence = float(d.confidence)
            berry.ripeness_confidence = float(d.ripeness_conf or 0.0)
            msg.berry_detection_list.append(berry)

        self._berry_pub.publish(msg)

        if self._publish_image:
            image = np.ascontiguousarray(result.image, dtype=np.uint8)
            img = Image()
            img.header.stamp = stamp
            img.header.frame_id = self._frame_id
            img.height, img.width = image.shape[:2]
            img.encoding = 'rgb8'
            img.step = img.width * 3
            img.data = image.tobytes()
            self._image_pub.publish(img)

    def destroy_node(self):
        self._camera.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
