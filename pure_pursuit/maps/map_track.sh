#!/bin/bash
# ========================
# STEP 1: Map the track
# Usage: ./map_track.sh <map_name>
# ========================
if [ -z "$1" ]; then
    echo "Usage: ./map_track.sh <map_name>"
    exit 1
fi

MAP_NAME=$1
MAP_DIR="$HOME/f1tenth_ws/src/pure_pursuit/maps"

source /opt/ros/humble/setup.bash
source ~/f1tenth_ws/install/setup.bash

echo "=== Launching SLAM Toolbox ==="
echo "Drive the car to build the map..."
ros2 launch slam_toolbox online_async_launch.py \
    slam_params_file:=$HOME/f1tenth_ws/slam_params.yaml \
    use_sim_time:=false &
SLAM_PID=$!