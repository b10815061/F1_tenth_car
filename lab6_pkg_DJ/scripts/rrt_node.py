#!/usr/bin/env python3
"""
RRT Motion Planning Node for F1Tenth
Lab 6 - 16-663 F1Tenth Autonomous Racing (CMU, Spring 2026)
 
Implements RRT (and optionally RRT*) as a local planner for obstacle
avoidance. Replans at each timestep using a fresh tree built around
the car's current position.
 
Pipeline per timestep:
    1. Receive pose → 2. Receive LiDAR → 3. Build occupancy grid
    4. Select goal waypoint → 5. Run RRT → 6. Pure Pursuit → 7. Publish
"""
 
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
 
import rclpy
from rclpy.node import Node
 
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
from scipy.ndimage import binary_dilation
 
 
@dataclass
class RRTNode:
    """Represents a single node in the RRT tree."""
    x: float = 0.0
    y: float = 0.0
    parent: int = -1
    cost: float = 0.0
    is_root: bool = False
 
 
class RRT(Node):
    def __init__(self):
        super().__init__('rrt_node')

        # Declare ROS parameters up front
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('waypoint_file', '/home/team10/f1tenth_ws/src/lab6_pkg/path.csv')

        # RRT parameters
        self.MAX_ITER = 2000
        self.STEER_RANGE = 0.5
        self.GOAL_THRESHOLD = 0.5
        self.GOAL_BIAS = 0.3
        self.SAMPLE_AREA_X = (-3.0, 8.0)   # Local frame: backward / forward (meters)
        self.SAMPLE_AREA_Y = (-4.0, 4.0)   # Local frame: right / left (meters)

        # RRT* parameters (extra credit)
        self.RRT_STAR = True
        self.SEARCH_RADIUS = 1.0

        # Occupancy grid parameters
        self.GRID_RESOLUTION = 0.05
        self.INFLATION_RADIUS = 0.001

        # Pure Pursuit parameters
        self.LOOKAHEAD_DISTANCE_Clear = 2.0
        self.LOOKAHEAD_DISTANCE_RRT = 1.0
        self.SPEED = 0.8
        self.UPSAMPLE_RES = 0.10

        # Waypoint parameters
        self.WAYPOINT_LOOKAHEAD = 3.0

        # State
        self.current_pose = None
        self.waypoints = None
        self.map_data = None
        self.map_info = None
        self.latest_scan = None

        # Load waypoints from Lab 5
        self.waypoints = self._load_waypoints()

        # Subscribers
        self.pose_sub = self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.pose_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback,
            rclpy.qos.QoSProfile(
                depth=1,
                durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL))

        # Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.tree_viz_pub = self.create_publisher(Marker, '/rrt_tree', 10)
        self.path_viz_pub = self.create_publisher(Marker, '/rrt_path', 10)
        self.goal_viz_pub = self.create_publisher(Marker, '/rrt_goal', 10)
        self.waypoint_viz_pub = self.create_publisher(MarkerArray, '/waypoints', 10)

        self.get_logger().info("RRT Node initialized. Waiting for pose, scan, and map...")
 
    # ============================================================
    # CALLBACKS
    # ============================================================
 
    def pose_callback(self, pose_msg: Odometry):
        self.current_pose = self._extract_pose(pose_msg)
        if self.current_pose is None or self.waypoints is None:
            return

        x, y, theta = self.current_pose

        goal = self._select_goal_waypoint(x, y, theta)
        if goal is None:
            self.get_logger().warn("No valid goal waypoint found.")
            return

        # Check if direct path to goal is clear
        occupancy_grid = self._build_occupancy_grid(x, y, theta)
        root_node = RRTNode(x=x, y=y)

        if not self._check_collision(root_node, goal, occupancy_grid, x, y, theta):
            # Path clear — follow the smooth waypoint sequence directly
            self.get_logger().info("PATH: Direct waypoint following")
            steering_angle = self._pure_pursuit_waypoints(x, y, theta)
        else:
            # Obstacle detected — plan with RRT
            self.get_logger().info("PATH: Using RRT")
            self.get_logger().info(f"Pose: ({x:.2f}, {y:.2f}, {theta:.2f}), Goal: ({goal[0]:.2f}, {goal[1]:.2f})")
            self.get_logger().info(f"Map received: {self.map_data is not None}, Scan received: {self.latest_scan is not None}")
            dx = goal[0] - x
            dy = goal[1] - y
            local_gx = dx * math.cos(theta) + dy * math.sin(theta)
            local_gy = -dx * math.sin(theta) + dy * math.cos(theta)
            gcol = int((local_gx - self.SAMPLE_AREA_X[0]) / self.GRID_RESOLUTION)
            grow = int((local_gy - self.SAMPLE_AREA_Y[0]) / self.GRID_RESOLUTION)
            self.get_logger().info(f"Goal local: ({local_gx:.2f}, {local_gy:.2f}), cell: ({grow},{gcol}), in grid: {0 <= grow < occupancy_grid.shape[0] and 0 <= gcol < occupancy_grid.shape[1]}")

            dx_g = goal[0] - x
            dy_g = goal[1] - y
            dist_to_goal = math.sqrt(dx_g**2 + dy_g**2)
            max_rrt_range = 3.0
            if dist_to_goal > max_rrt_range:
                scale = max_rrt_range / dist_to_goal
                goal = (x + dx_g * scale, y + dy_g * scale)
                self.get_logger().info(f"Intermediate goal: ({goal[0]:.2f}, {goal[1]:.2f})")

            tree, path = self._run_rrt(x, y, theta, goal)

            if path is not None:
                self.get_logger().info(f"RRT path: {len(path)} points, start: ({path[0][0]:.2f},{path[0][1]:.2f}), end: ({path[-1][0]:.2f},{path[-1][1]:.2f})")

            if path is None or len(path) < 2:
                self.get_logger().warn("RRT failed to find a path.")
                return
            steering_angle = self._pure_pursuit(path, x, y, theta)
            self._publish_tree_viz(tree)
            self._publish_path_viz(path)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.SPEED
        self.get_logger().info(f"Steer: {steering_angle:.3f}, Speed: {self.SPEED}")

        self.drive_pub.publish(drive_msg)
        self._publish_goal_viz(goal)
 
    def scan_callback(self, scan_msg: LaserScan):
        """Stores the latest LiDAR scan for occupancy grid construction."""
        self.latest_scan = scan_msg
 
    def map_callback(self, map_msg: OccupancyGrid):
        """Stores the static map received from map_server or slam_toolbox."""
        self.map_data = np.array(map_msg.data).reshape(
            (map_msg.info.height, map_msg.info.width))
        self.map_info = map_msg.info
        self.get_logger().info(
            f"Map received: {map_msg.info.width}x{map_msg.info.height}, "
            f"resolution={map_msg.info.resolution}")
 
    # ============================================================
    # POSE EXTRACTION
    # ============================================================
 
    def _extract_pose(self, odom_msg: Odometry) -> Optional[Tuple[float, float, float]]:
        """Extracts (x, y, theta) from an Odometry message via quaternion-to-yaw conversion."""
        px = odom_msg.pose.pose.position.x
        py = odom_msg.pose.pose.position.y
        qx = odom_msg.pose.pose.orientation.x
        qy = odom_msg.pose.pose.orientation.y
        qz = odom_msg.pose.pose.orientation.z
        qw = odom_msg.pose.pose.orientation.w
        theta = math.atan2(2.0 * (qw * qz + qx * qy),
                           1.0 - 2.0 * (qy * qy + qz * qz))
        return (px, py, theta)
 
    # ============================================================
    # WAYPOINT LOADING AND GOAL SELECTION
    # ============================================================
 
    def _load_waypoints(self) -> Optional[np.ndarray]:
        """Loads pre-recorded waypoints from a CSV file (Lab 5 output)."""
        waypoint_file = self.get_parameter('waypoint_file').value
        try:
            data = np.loadtxt(waypoint_file, delimiter=',', skiprows=1)
            if data.ndim == 2 and data.shape[1] >= 2:
                waypoints = data[:, :2]
            else:
                self.get_logger().error("Waypoint file has unexpected format.")
                return None
            self.get_logger().info(f"Loaded {len(waypoints)} waypoints.")
            return waypoints
        except Exception as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")
            return None
 
    def _select_goal_waypoint(self, x: float, y: float, theta: float) -> Optional[Tuple[float, float]]:
        """Selects the waypoint closest to WAYPOINT_LOOKAHEAD distance ahead of the car."""
        best = None
        best_diff = float('inf')
 
        for wp in self.waypoints:
            dx = wp[0] - x
            dy = wp[1] - y
 
            # Skip waypoints behind the car
            local_x = dx * math.cos(theta) + dy * math.sin(theta)
            if local_x < 0:
                continue
 
            dist = math.sqrt(dx**2 + dy**2)
            diff = abs(dist - self.WAYPOINT_LOOKAHEAD)
            if diff < best_diff:
                best_diff = diff
                best = wp
 
        return (best[0], best[1]) if best is not None else None
 
    # ============================================================
    # OCCUPANCY GRID
    # ============================================================
 
    def _build_occupancy_grid(self, x: float, y: float, theta: float) -> np.ndarray:
        """
        Builds a local binary occupancy grid by fusing the static map
        and live LiDAR scan. Grid is in the car's local frame.
        """
        width = int((self.SAMPLE_AREA_X[1] - self.SAMPLE_AREA_X[0]) / self.GRID_RESOLUTION)
        height = int((self.SAMPLE_AREA_Y[1] - self.SAMPLE_AREA_Y[0]) / self.GRID_RESOLUTION)
        grid = np.zeros((height, width), dtype=np.int8)
 
        # Stamp static map walls onto local grid
        if self.map_data is not None and self.map_info is not None:
            search_radius = max(
                abs(self.SAMPLE_AREA_X[0]), abs(self.SAMPLE_AREA_X[1]),
                abs(self.SAMPLE_AREA_Y[0]), abs(self.SAMPLE_AREA_Y[1])
            )
 
            origin_x = self.map_info.origin.position.x
            origin_y = self.map_info.origin.position.y
            res = self.map_info.resolution
 
            # Only search a small region of the static map near the car
            car_col = int((x - origin_x) / res)
            car_row = int((y - origin_y) / res)
            radius_cells = int(search_radius / res)
 
            row_min = max(0, car_row - radius_cells)
            row_max = min(self.map_data.shape[0], car_row + radius_cells)
            col_min = max(0, car_col - radius_cells)
            col_max = min(self.map_data.shape[1], car_col + radius_cells)
 
            for row in range(row_min, row_max):
                for col in range(col_min, col_max):
                    if self.map_data[row][col] >= 50:
                        # Map pixel → world coordinates
                        world_x = origin_x + col * res
                        world_y = origin_y + row * res
 
                        # World → car's local frame
                        dx = world_x - x
                        dy = world_y - y
                        local_x = dx * math.cos(theta) + dy * math.sin(theta)
                        local_y = -dx * math.sin(theta) + dy * math.cos(theta)
 
                        # Local → grid cell
                        grid_col = int((local_x - self.SAMPLE_AREA_X[0]) / self.GRID_RESOLUTION)
                        grid_row = int((local_y - self.SAMPLE_AREA_Y[0]) / self.GRID_RESOLUTION)
 
                        if 0 <= grid_row < height and 0 <= grid_col < width:
                            grid[grid_row][grid_col] = 1
 
        # Stamp LiDAR hits (already in the car's local frame)
        scan = self.latest_scan
        if scan is not None:
            for i in range(len(scan.ranges)):
                r = scan.ranges[i]
                if r < scan.range_min or r > scan.range_max or math.isinf(r) or math.isnan(r):
                    continue
 
                angle = scan.angle_min + i * scan.angle_increment
                local_x = r * math.cos(angle)
                local_y = r * math.sin(angle)
 
                grid_col = int((local_x - self.SAMPLE_AREA_X[0]) / self.GRID_RESOLUTION)
                grid_row = int((local_y - self.SAMPLE_AREA_Y[0]) / self.GRID_RESOLUTION)
 
                if 0 <= grid_row < height and 0 <= grid_col < width:
                    grid[grid_row][grid_col] = 1

        inflate_cells = int(self.INFLATION_RADIUS / self.GRID_RESOLUTION)  # 0.2 / 0.05 = 4 cells
        struct = np.ones((2 * inflate_cells + 1, 2 * inflate_cells + 1), dtype=np.int8)
        grid = binary_dilation(grid, structure=struct).astype(np.int8)
        self.get_logger().info(f"Grid: {grid.shape}, occupied: {np.sum(grid)}/{grid.size} ({100*np.sum(grid)/grid.size:.1f}%)")

        return grid
 
    # ============================================================
    # RRT CORE
    # ============================================================
 
    def _run_rrt(self, x: float, y: float, theta: float,
                 goal: Tuple[float, float]) -> Tuple[List[RRTNode], Optional[List[Tuple[float, float]]]]:
        """Builds a fresh RRT tree and returns the tree and path (if found)."""
        occupancy_grid = self._build_occupancy_grid(x, y, theta)
        local_col = int((0 - self.SAMPLE_AREA_X[0]) / self.GRID_RESOLUTION)
        local_row = int((0 - self.SAMPLE_AREA_Y[0]) / self.GRID_RESOLUTION)
        self.get_logger().info(f"Car cell ({local_row},{local_col}): {'BLOCKED' if occupancy_grid[local_row][local_col]==1 else 'FREE'}")
 
        root = RRTNode(x=x, y=y, parent=-1, cost=0.0, is_root=True)
        tree = [root]
 
        for i in range(self.MAX_ITER):
            sampled_point = self._sample(goal)
            nearest_idx = self._nearest(tree, sampled_point)
            new_point = self._steer(tree[nearest_idx], sampled_point)
 
            if self._check_collision(tree[nearest_idx], new_point, occupancy_grid, x, y, theta):
                continue
 
            if self.RRT_STAR:
                # RRT*: choose lowest-cost parent and rewire neighbors
                near_indices = self._near(tree, new_point)
                best_parent = nearest_idx
                best_cost = self._cost(tree, nearest_idx) + self._line_cost(tree[nearest_idx], new_point)
 
                for idx in near_indices:
                    candidate_cost = self._cost(tree, idx) + self._line_cost(tree[idx], new_point)
                    if candidate_cost < best_cost:
                        if not self._check_collision(tree[idx], new_point, occupancy_grid, x, y, theta):
                            best_parent = idx
                            best_cost = candidate_cost
 
                new_node = RRTNode(x=new_point[0], y=new_point[1],
                                   parent=best_parent, cost=best_cost)
                tree.append(new_node)
                self._rewire(tree, len(tree) - 1, near_indices, occupancy_grid, x, y, theta)
            else:
                # Basic RRT: connect to nearest node
                new_node = RRTNode(x=new_point[0], y=new_point[1],
                                   parent=nearest_idx, cost=0.0)
                tree.append(new_node)
 
            if self._is_goal(new_node, goal):
                path = self._find_path(tree, len(tree) - 1)
                return tree, path
 
        self.get_logger().debug("RRT: max iterations reached without finding path.")
        self.get_logger().info(f"RRT: {len(tree)} nodes after {self.MAX_ITER} iterations")
        return tree, None
 
    # ============================================================
    # RRT FUNCTIONS
    # ============================================================
 
    def _sample(self, goal: Tuple[float, float]) -> Tuple[float, float]:
        """Samples a random point with goal bias. Returns map-frame coordinates."""
        if np.random.random() < self.GOAL_BIAS:
            return goal
 
        # Sample in car's local frame, then transform to map frame
        rand_x = np.random.uniform(self.SAMPLE_AREA_X[0], self.SAMPLE_AREA_X[1])
        rand_y = np.random.uniform(self.SAMPLE_AREA_Y[0], self.SAMPLE_AREA_Y[1])
        car_x, car_y, theta = self.current_pose
 
        map_x = car_x + rand_x * math.cos(theta) - rand_y * math.sin(theta)
        map_y = car_y + rand_x * math.sin(theta) + rand_y * math.cos(theta)
        return (map_x, map_y)
 
    def _nearest(self, tree: List[RRTNode], point: Tuple[float, float]) -> int:
        """Returns the index of the tree node closest to the given point."""
        EdistList = []
        for i in range(len(tree)):
            EdistList.append(math.dist((tree[i].x, tree[i].y), point))
        return EdistList.index(min(EdistList))
 
    def _steer(self, nearest_node: RRTNode, sampled_point: Tuple[float, float]) -> Tuple[float, float]:
        """Steps from nearest_node toward sampled_point by at most STEER_RANGE."""
        Edist = math.dist((nearest_node.x, nearest_node.y), sampled_point)
        if Edist > self.STEER_RANGE:
            direction = math.atan2(sampled_point[1] - nearest_node.y,
                                   sampled_point[0] - nearest_node.x)
            new_x = nearest_node.x + self.STEER_RANGE * math.cos(direction)
            new_y = nearest_node.y + self.STEER_RANGE * math.sin(direction)
            return (new_x, new_y)
        else:
            return sampled_point
 
    def _check_collision(self, node_from: RRTNode, point_to: Tuple[float, float],
                         occupancy_grid: np.ndarray,
                         car_x: float, car_y: float, theta: float) -> bool:
        """Checks the edge from node_from to point_to for collisions against the occupancy grid."""
        Edist = math.dist((node_from.x, node_from.y), point_to)
        num_steps = int(Edist / self.GRID_RESOLUTION)
 
        for j in range(num_steps + 1):
            t = j / max(num_steps, 1)
            check_x = node_from.x + t * (point_to[0] - node_from.x)
            check_y = node_from.y + t * (point_to[1] - node_from.y)
 
            # Map frame → local frame → grid cell
            dx = check_x - car_x
            dy = check_y - car_y
            local_x = dx * math.cos(theta) + dy * math.sin(theta)
            local_y = -dx * math.sin(theta) + dy * math.cos(theta)
            col = int((local_x - self.SAMPLE_AREA_X[0]) / self.GRID_RESOLUTION)
            row = int((local_y - self.SAMPLE_AREA_Y[0]) / self.GRID_RESOLUTION)
 
            if row < 0 or row >= occupancy_grid.shape[0] or col < 0 or col >= occupancy_grid.shape[1]:
                return True     # Out of bounds — assume collision
            if occupancy_grid[row][col] == 1:
                return True     # Occupied cell
 
        return False            # Entire edge is clear
 
    def _is_goal(self, node: RRTNode, goal: Tuple[float, float]) -> bool:
        """Returns True if the node is within GOAL_THRESHOLD of the goal."""
        return math.dist((node.x, node.y), goal) < self.GOAL_THRESHOLD
 
    def _find_path(self, tree: List[RRTNode], goal_idx: int) -> List[Tuple[float, float]]:
        """Traces parent pointers from goal back to root, returns path car → goal."""
        path = []
        i = goal_idx
        while i != -1:
            path.append((tree[i].x, tree[i].y))
            i = tree[i].parent
        path.reverse()
        return path
 
    # ============================================================
    # RRT* FUNCTIONS (EXTRA CREDIT)
    # ============================================================
 
    def _cost(self, tree: List[RRTNode], node_idx: int) -> float:
        return tree[node_idx].cost
 
    def _line_cost(self, node: RRTNode, point: Tuple[float, float]) -> float:
        return math.dist((node.x, node.y), point)
 
    def _near(self, tree: List[RRTNode], point: Tuple[float, float]) -> List[int]:
        indices = []
        for i in range(len(tree)):
            if math.dist((tree[i].x, tree[i].y), point) < self.SEARCH_RADIUS:
                indices.append(i)
        return indices
 
    def _rewire(self, tree: List[RRTNode], new_idx: int,
                near_indices: List[int], occupancy_grid: np.ndarray,
                car_x: float, car_y: float, car_theta: float):
        new_node = tree[new_idx]
        for idx in near_indices:
            # Cost of reaching this neighbor through the new node
            new_cost = new_node.cost + self._line_cost(new_node, (tree[idx].x, tree[idx].y))

            # If cheaper and collision-free, rewire
            if new_cost < tree[idx].cost:
                if not self._check_collision(new_node, (tree[idx].x, tree[idx].y),
                                            occupancy_grid, car_x, car_y, car_theta):
                    tree[idx].parent = new_idx
                    tree[idx].cost = new_cost
 
    # ============================================================
    # PURE PURSUIT
    # ============================================================
 
    def _pure_pursuit(self, path: List[Tuple[float, float]],
                      x: float, y: float, theta: float) -> float:
        """Follows the RRT path using Pure Pursuit. Returns a steering angle."""
 
        # Upsample the path for smoother waypoint selection
        upsampled = [path[0]]
        for i in range(1, len(path)):
            x0, y0 = path[i - 1]
            x1, y1 = path[i]
            dist = math.dist((x0, y0), (x1, y1))
            num_points = int(dist / self.UPSAMPLE_RES)
            for j in range(1, num_points + 1):
                t = j / (num_points + 1)
                new_x = x0 + t * (x1 - x0)
                new_y = y0 + t * (y1 - y0)
                upsampled.append((new_x, new_y))
            upsampled.append(path[i])
 
        # Find the upsampled point closest to LOOKAHEAD_DISTANCE ahead
        best = None
        best_diff = float('inf')
 
        for tp in upsampled:
            dx = tp[0] - x
            dy = tp[1] - y
            local_x = dx * math.cos(theta) + dy * math.sin(theta)
            if local_x < 0:
                continue
            dist = math.sqrt(dx**2 + dy**2)
            diff = abs(dist - self.LOOKAHEAD_DISTANCE_RRT)
            if diff < best_diff:
                best_diff = diff
                best = tp
 
        if best is None:
            return 0.0
 
        # Transform target to local frame and compute steering via curvature
        DX = best[0] - x
        DY = best[1] - y
        local_x = DX * math.cos(theta) + DY * math.sin(theta)
        local_y = -DX * math.sin(theta) + DY * math.cos(theta)
 
        L = math.sqrt(local_x**2 + local_y**2)
        if L < 0.01:
            return 0.0
 
        # γ = 2y / L²
        self.get_logger().info(f"Pursuit target: ({best[0]:.2f}, {best[1]:.2f}), local: ({local_x:.2f}, {local_y:.2f}), L: {L:.2f}")
        curvature = 2 * local_y / (L ** 2)
        steering_angle = math.atan2(curvature, 1.0)
        steering_angle = max(-0.4, min(0.4, steering_angle))
        return steering_angle

    def _pure_pursuit_waypoints(self, x: float, y: float, theta: float) -> float:
        """Follows the original waypoints directly using Pure Pursuit (no RRT needed)."""
        best = None
        best_diff = float('inf')

        for wp in self.waypoints:
            dx = wp[0] - x
            dy = wp[1] - y
            local_x = dx * math.cos(theta) + dy * math.sin(theta)
            if local_x < 0:
                continue
            dist = math.sqrt(dx**2 + dy**2)
            diff = abs(dist - self.LOOKAHEAD_DISTANCE_Clear)
            if diff < best_diff:
                best_diff = diff
                best = wp

        if best is None:
            return 0.0

        DX = best[0] - x
        DY = best[1] - y
        local_x = DX * math.cos(theta) + DY * math.sin(theta)
        local_y = -DX * math.sin(theta) + DY * math.cos(theta)

        L = math.sqrt(local_x**2 + local_y**2)
        if L < 0.01:
            return 0.0

        curvature = 2 * local_y / (L ** 2)
        steering_angle = math.atan2(curvature, 1.0)
        steering_angle = max(-0.4, min(0.4, steering_angle))
        return steering_angle
 
    # ============================================================
    # VISUALIZATION
    # ============================================================
 
    def _publish_tree_viz(self, tree: List[RRTNode]):
        """Publishes the RRT tree edges as a LINE_LIST marker for RViz."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'rrt_tree'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 0.0
        marker.color.a = 0.6
 
        for node in tree:
            if node.parent == -1:
                continue
            parent = tree[node.parent]
            p1 = Point()
            p1.x, p1.y, p1.z = node.x, node.y, 0.0
            p2 = Point()
            p2.x, p2.y, p2.z = parent.x, parent.y, 0.0
            marker.points.append(p1)
            marker.points.append(p2)
 
        self.tree_viz_pub.publish(marker)
 
    def _publish_path_viz(self, path: List[Tuple[float, float]]):
        """Publishes the chosen RRT path as a LINE_STRIP marker for RViz."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'rrt_path'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
 
        for (px, py) in path:
            p = Point()
            p.x, p.y, p.z = px, py, 0.05
            marker.points.append(p)
 
        self.path_viz_pub.publish(marker)
 
    def _publish_goal_viz(self, goal: Tuple[float, float]):
        """Publishes the current RRT goal as a SPHERE marker for RViz."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'rrt_goal'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = goal[0]
        marker.pose.position.y = goal[1]
        marker.pose.position.z = 0.1
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
 
        self.goal_viz_pub.publish(marker)
 
 
def main(args=None):
    rclpy.init(args=args)
    rrt_node = RRT()
    rclpy.spin(rrt_node)
    rrt_node.destroy_node()
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()