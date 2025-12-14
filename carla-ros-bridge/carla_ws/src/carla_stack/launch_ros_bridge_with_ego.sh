#!/bin/bash

# Activate the Python virtual environment (ensure the path is correct).
source ~/carla-venv/bin/activate

# Start the ROS bridge with the example ego vehicle
roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch

