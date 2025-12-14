cmake_minimum_required(VERSION 3.0.2)
project(carla_controller)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  ackermann_msgs
  nav_msgs
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  controller_manager.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_python_setup()