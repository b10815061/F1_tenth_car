#!/usr/bin/env python3
"""
Waypoint Recorder for F1Tenth
Drive the car manually with the joystick and this node records
the car's position at regular intervals. Press Ctrl+C to stop
and save the waypoints to a CSV file.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import csv
import math
import sys

class WaypointRecorder(Node):
    def __init__(self):
        super().__init__('waypoint_recorder')

        # Parameters
        self.declare_parameter('output_file', '/home/team10/f1tenth_ws/src/lab6_pkg/path.csv')
        self.declare_parameter('min_distance', 0.3)  # meters between recorded points

        self.output_file = self.get_parameter('output_file').value
        self.min_dist = self.get_parameter('min_distance').value

        # Subscribe to real car odometry
        self.odom_sub = self.create_subscription(
            Odometry, '/pf/pose/odom', self.odom_callback, 10)

        self.waypoints = []
        self.last_x = None
        self.last_y = None

        self.get_logger().info(f"Waypoint Recorder started.")
        self.get_logger().info(f"Drive the car around the track with the joystick.")
        self.get_logger().info(f"Min distance between points: {self.min_dist}m")
        self.get_logger().info(f"Press Ctrl+C to stop and save to: {self.output_file}")

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.last_x is None:
            # Record first point
            self.waypoints.append((x, y))
            self.last_x = x
            self.last_y = y
            self.get_logger().info(f"First waypoint: ({x:.3f}, {y:.3f})")
            return

        dist = math.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)

        if dist >= self.min_dist:
            self.waypoints.append((x, y))
            self.last_x = x
            self.last_y = y
            if len(self.waypoints) % 10 == 0:
                self.get_logger().info(f"Recorded {len(self.waypoints)} waypoints")

    def save_waypoints(self):
        if not self.waypoints:
            self.get_logger().warn("No waypoints recorded!")
            return

        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            for wp in self.waypoints:
                writer.writerow([wp[0], wp[1]])

        self.get_logger().info(f"Saved {len(self.waypoints)} waypoints to {self.output_file}")


def main(args=None):
    rclpy.init(args=args)
    node = WaypointRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_waypoints()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()