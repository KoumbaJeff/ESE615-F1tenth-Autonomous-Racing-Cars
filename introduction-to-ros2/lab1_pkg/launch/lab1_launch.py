#!/usr/bin/env python3

import launch
import launch_ros.actions

def generate_launch_description():
    talker_node = launch_ros.actions.Node(
        package='lab1_pkg',           # Your package name
        executable='talker.py',       # Name of your talker script
        name='talker',
        parameters=[
            {'v': 1.0},               # Example parameter values
            {'d': 0.5}
        ]
    )

    relay_node = launch_ros.actions.Node(
        package='lab1_pkg',           # Your package name
        executable='relay.py',        # Name of your relay script
        name='relay'
    )

    return launch.LaunchDescription([
        talker_node,
        relay_node
    ])
