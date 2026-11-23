"""Start HarvestAI perception: the OAK-D Pro node plus the Gradio UI.

    ros2 launch harvestai_bringup harvest.launch.py
    ros2 launch harvestai_bringup harvest.launch.py fps:=10.0 confidence:=0.4 ui:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    fps = LaunchConfiguration('fps')
    confidence = LaunchConfiguration('confidence')
    ui = LaunchConfiguration('ui')

    return LaunchDescription([
        DeclareLaunchArgument('fps', default_value='5.0',
                              description='Detections published per second'),
        DeclareLaunchArgument('confidence', default_value='0.5',
                              description='YOLO confidence threshold'),
        DeclareLaunchArgument('ui', default_value='true',
                              description='Also start the Gradio UI'),

        Node(
            package='harvestai_perception',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[{
                'fps': ParameterValue(fps, value_type=float),
                'confidence': ParameterValue(confidence, value_type=float),
            }],
        ),
        Node(
            package='harvestai_perception',
            executable='perception_ui',
            name='perception_ui',
            output='screen',
            condition=IfCondition(ui),
        ),
    ])
