#!/bin/bash
source /opt/ros/humble/setup.bash

# Set robot IP here
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

# Point to robot
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

rviz2 -d /home/user/AMR_PyQt/RViz/slam_view.rviz
