#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Create publisher and subscriber
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)

        # Set PID gains
        self.kp = 1.5
        self.kd = 0.05
        self.ki = 0.00

        # Store history
        self.integral = 0.0
        self.prev_error = 0.0
        self.error = 0.0

        # Store necessary values
        self.angle_min = 0.0
        self.angle_increment = 0.0
        self.desired_dist = 0.45
        self.L = 1.0
        self.theta = np.pi * 5/18

    def get_range(self, range_data, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: desired angle in degrees (relative to forward)

        Returns:
            range: range measurement in meters at the given angle

        """
        # Calculate index using self.angle_min and self.angle_increment
        idx = int((angle - self.angle_min) / self.angle_increment)
        idx = np.clip(idx, 0, len(range_data) - 1)
        val = range_data[idx]
        # Handle NaN/inf
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val

    def get_error(self, range_data, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """

        # Get two LiDAR ranges: a at theta, b at 90 deg
        a = self.get_range(range_data, self.theta)
        b = self.get_range(range_data, np.pi/2)
        # If either range is invalid, return 0 error
        if a == 0.0 or b == 0.0:
            return 0.0
        # Calculate alpha
        alpha = np.arctan2(a * np.cos(self.theta) - b, a * np.sin(self.theta))
        # Current distance to wall
        Dt = b * np.cos(alpha)
        # Projected future distance
        Dt1 = Dt + self.L * np.sin(alpha)
        # Error: desired - projected
        error = dist - Dt1
        return error

    def pid_control(self, error):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        """
        # PID calculations
        self.integral += error
        derivative = error - self.prev_error
        angle = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        # Set velocity based on steering angle in degrees
        abs_angle_deg = np.abs(np.rad2deg(angle))
        if abs_angle_deg < 8:
            velocity = 4.0
        elif abs_angle_deg < 12:
            velocity = 3.4
        elif abs_angle_deg < 16:
            velocity = 3.2
        elif abs_angle_deg < 20:
            velocity = 2.8
        else:
            velocity = 1.0

        # Fill and publish drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = -angle  # radians
        drive_msg.drive.speed = velocity
        self.drive_pub.publish(drive_msg)
        #print(f"Error: {error:.3f}, Steering Angle: {angle:.3f}, Speed: {velocity:.3f}")
        #print(f"Angle min: {self.angle_min}, Angle increment: {self.angle_increment}")

    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        # Update angle_min and angle_increment from msg
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment
        self.prev_error = self.error
        error = self.get_error(msg.ranges, self.desired_dist)
        self.error = error
        self.pid_control(error)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
