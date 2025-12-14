#!/bin/bash

# 加载 ROS 环境（如果没加到 bashrc，可取消注释）
 source /opt/ros/noetic/setup.bash
 source ~/carla-ros-bridge/catkin_ws/devel/setup.bash

# 激活 Python 虚拟环境
source ~/carla-venv/bin/activate

# 启动 ROS bridge，增加连接等待时间
roslaunch carla_ros_bridge carla_ros_bridge.launch register_all_sensors:=False timeout:=10
