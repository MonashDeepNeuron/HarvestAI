import rclpy
from rclpy.node import Node

class GreeterNode(Node):
    def __init__(self):
        super().__init__('greeter')
        self.get_logger().info('Greeter started')

def main():
    rclpy.init()
    node = GreeterNode
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()