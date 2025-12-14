#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import numpy as np
import rospy
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
from scipy.spatial import Delaunay
from visualization_msgs.msg import Marker, MarkerArray
import pyransac
from pyransac import line2d
import rospkg
from nav_msgs.msg import Odometry
from ccma import CCMA # type: ignore


def get_boxed_indices(start_theta: float,
                      end_theta: float,
                      num_lidar_beams: int,
                      des_lidar_beams: int,
                      aspect_ratio: float):
    '''
    Finds an evenly spaced "boxed" pattern of beams based on the TUM paper
    "ROS-based localization of a race vehicle at high-speed using LIDAR".
    '''
    beam_angles = np.linspace(start_theta, end_theta, num_lidar_beams)

    MID_IDX = num_lidar_beams // 2
    sparse_idxs = [MID_IDX]

    # Structures
    a = aspect_ratio
    beam_proj = 2 * a * np.array([np.cos(beam_angles), np.sin(beam_angles)])
    # Allows us to do intersection math later
    beam_intersections = np.zeros((2, num_lidar_beams))

    # Compute the points of intersection along a uniform corridor of given aspect ratio
    box_corners = [(a, 1), (a, -1), (-a, -1), (-a, 1)]
    for idx, _ in enumerate(box_corners):
        x1, y1 = box_corners[idx]
        x2, y2 = box_corners[0] if idx == 3 else box_corners[idx + 1]
        for i in range(num_lidar_beams):
            x4 = beam_proj[0, i]
            y4 = beam_proj[1, i]

            den = (x1 - x2) * (-y4) - (y1 - y2) * (-x4)
            if den == 0:
                continue    # parallel lines

            t = ((x1) * (-y4) - (y1) * (-x4)) / den
            u = ((x1) * (y1 - y2) - (y1) * (x1 - x2)) / den

            px = u * x4
            py = u * y4
            if 0 <= t <= 1.0 and 0 <= u <= 1.0:
                beam_intersections[0, i] = px
                beam_intersections[1, i] = py

    # Compute the distances for uniform spacing
    dx = np.diff(beam_intersections[0, :])
    dy = np.diff(beam_intersections[1, :])
    dist = np.sqrt(dx**2 + dy**2)
    total_dist = np.sum(dist)
    dist_amt = total_dist / (des_lidar_beams - 1)

    # Calc half of the evenly-spaced interval first, then the other half
    idx = MID_IDX + 1
    DES_BEAMS2 = des_lidar_beams // 2 + 1
    acc = 0
    while len(sparse_idxs) <= DES_BEAMS2:
        acc += dist[idx]
        if acc >= dist_amt:
            acc = 0
            sparse_idxs.append(idx - 1)
        idx += 1

        if idx == num_lidar_beams - 1:
            sparse_idxs.append(num_lidar_beams - 1)
            break

    mirrored_half = []
    for idx in sparse_idxs[1:]:
        new_idx = 2 * sparse_idxs[0] - idx
        mirrored_half.insert(0, new_idx)
    sparse_idxs = mirrored_half + sparse_idxs

    sparse_idxs = np.array(sparse_idxs)
    return sparse_idxs, beam_angles[sparse_idxs]

# Create Delaunay triangulations
def calculate_centerline(points, segment_mask):
    delaunay = Delaunay(points)
    triangles = points[delaunay.simplices]

    # valid masks for triangles
    valid_triangles = np.ones(len(triangles), dtype=bool)
    # centroid = np.ones((len(triangles), 2))

    # Filter for triangles where two sides at least 3 times longer than the third side
    for i, tr in enumerate(triangles[valid_triangles]):
        side_lens = np.linalg.norm(tr - np.roll(tr, 1, axis=0), axis=1)
        side_lens = np.sort(side_lens)
        gleichschenkligkeit = False
        spitz = False
        big = False

        if 0.8 < abs(side_lens[1] / side_lens[2]) < 1.2:
            gleichschenkligkeit = True
        if abs(side_lens[1] / side_lens[0]) > 1.25:
            spitz = True
        if np.linalg.norm(np.cross(tr[1] - tr[0], tr[2] - tr[0])) > 5:
            big = True

        if len(np.unique(segment_mask)) > 1:
            same_side = segment_mask[delaunay.simplices[i][0]] == segment_mask[delaunay.simplices[i][1]] and segment_mask[delaunay.simplices[i][0]] == segment_mask[delaunay.simplices[i][2]]
        else:
            same_side = False
        # valid_triangles[i] = ((gleichschenkligkeit and spitz) or big) and not same_side
        valid_triangles[i] =  not same_side

    # Filter triangles based on side length conditions
    valid_triangles_idx = np.where(valid_triangles)[0]

    # Store already processed edges
    processed_edges = set()
    midpoints = []

    # Calculate midpoints for valid triangles
    for i in valid_triangles_idx:
        triangle = triangles[i]
        segment_idx = segment_mask[delaunay.simplices[i]]
        midpoint = calculate_midpoint(triangle, segment_idx, processed_edges)

        if midpoint is not None:

            midpoints.append(midpoint)

    midpoints = np.array(midpoints)

    assert midpoints.shape[1] == 2, f"Midpoints have wrong shape: {midpoints.shape}"

    return midpoints, delaunay

def calculate_midpoint(triangle, segment_indices, processed_edges):
    """
    Compute the midpoint of the edge across different segment sides in a triangle

    Parameters:
        triangle: Coordinates of the triangle’s three vertices
        segment_indices: Segment indices corresponding to the three vertices
        processed_edges: Set of already processed edges

    Returns:
        np.array: Midpoint coordinates of the edge across different sides, or None if unavailable
    """
    # Find the point on a different segment side
    if segment_indices[0] != segment_indices[1] and segment_indices[0] != segment_indices[2]:
        
        # Point 0 is across different sides
        edge1 = tuple(sorted([tuple(triangle[0]), tuple(triangle[1])]))
        edge2 = tuple(sorted([tuple(triangle[0]), tuple(triangle[2])]))
        
        # Point 0 is across different sides
        if edge1 not in processed_edges:
            processed_edges.add(edge1)
            return (triangle[0] + triangle[1]) / 2
        elif edge2 not in processed_edges:
            processed_edges.add(edge2)
            return (triangle[0] + triangle[2]) / 2
            
    elif segment_indices[1] != segment_indices[0] and segment_indices[1] != segment_indices[2]:
        
        edge1 = tuple(sorted([tuple(triangle[1]), tuple(triangle[0])]))
        edge2 = tuple(sorted([tuple(triangle[1]), tuple(triangle[2])]))
        
        if edge1 not in processed_edges:
            processed_edges.add(edge1)
            return (triangle[1] + triangle[0]) / 2
        elif edge2 not in processed_edges:
            processed_edges.add(edge2)
            return (triangle[1] + triangle[2]) / 2
            
    else:

        edge1 = tuple(sorted([tuple(triangle[2]), tuple(triangle[0])]))
        edge2 = tuple(sorted([tuple(triangle[2]), tuple(triangle[1])]))
        
        if edge1 not in processed_edges:
            processed_edges.add(edge1)
            return (triangle[2] + triangle[0]) / 2
        elif edge2 not in processed_edges:
            processed_edges.add(edge2)
            return (triangle[2] + triangle[1]) / 2
    
    return None

class DelaunayFTG:
    def __init__(
            self,
            mapping=False,
            debug=False,
            safety_radius=40,
            max_lidar_dist=10.0,
            max_speed=5.0,
            range_offset=180,
            track_width=5) -> None:
        """
        Initialize the FTG controller.

        Parameters:
            mapping (bool): Flag indicating whether FTG is used for mapping or not.
        """
        self.mapping = mapping

        self.radians_per_elem = None  # used when calculating the angles of the LiDAR data
        self.range_offset = range_offset
        self.track_width = track_width

        self.velocity = 0
        self.acceleration = 0
        self.jerk = 0   

        # params
        self.DEBUG = debug
        # Lidar processing params
        self.PREPROCESS_CONV_SIZE = 3
        self.SAFETY_RADIUS = safety_radius
        self.MAX_LIDAR_DIST = max_lidar_dist

        self.masking = "range"
        # self.range_mask_threshold = 0.6
        self.diff_threshold = 3

        self.scan_pub = rospy.Publisher('/scan_proc/markers', MarkerArray, queue_size=10)
        self.best_pnt = rospy.Publisher('/best_points/marker', Marker, queue_size=10)
        self.best_gap = rospy.Publisher('/best_gap/markers', MarkerArray, queue_size=10)
        
        # ...existing code...
        self.LIDAR_SAMPLE_IDXS = None
        self.LIDAR_THETA_LUT = None

        # Speed params
        self.MAX_SPEED = max_speed
        scale = 0.6  # .575 is  max
        self.CORNERS_SPEED = 0.4 * self.MAX_SPEED * scale
        self.MILD_CORNERS_SPEED = 0.6 * self.MAX_SPEED * scale
        self.STRAIGHTS_SPEED = 0.8 * self.MAX_SPEED * scale
        self.ULTRASTRAIGHTS_SPEED = self.MAX_SPEED * scale
        
        self.lookahead_qv = 0.5
        self.lookahead_qm = 1.5

        self.accel_limit = 2
        self.decel_limit = 1
        self.friction_coeff = 0.8
        self.friction_scale = 1.0

        # Steering params
        self.STRAIGHTS_STEERING_ANGLE = np.pi / 18  # 10 degrees
        self.MILD_CURVE_ANGLE = np.pi / 6  # 30 degrees
        self.ULTRASTRAIGHTS_ANGLE = np.pi / 60  # 3 deg
       
        if debug:
            plt.ion() 
            self.fig, self.ax = plt.subplots()
            self.ax.set_title("Centerline Debug")
            plt.show()
        # pyransac for RANSAC
        self.RANSAC = False
        self.pyrs_params = pyransac.RansacParams(samples=2,
                                           iterations=100,
                                           confidence=0.999,
                                           threshold=0.5,)
        self.pyrs_model = line2d.Line2D()

        try:
            path = rospkg.RosPack().get_path('stack_master')
            self.ggv  = np.genfromtxt(path +'/config/NUC5/GGV/dubi_tpu/ggv.csv', delimiter=',', skip_header=1) #TODO: this is definitely too much hardcoding??
            self.ax_machine = np.genfromtxt(path + '/config/NUC5/GGV/dubi_tpu/ax_max_machines.csv', delimiter=',', skip_header=1)
            rospy.loginfo("Loaded the ggv and ax_max_machines files")
        except Exception as e:
            self.ggv = None
            self.ax_machine = None
            rospy.logwarn("Could not load the ggv and ax_max_machines files")

        self.ego_velocity = 0.1
        rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry, self._odom_callback)


    def _odom_callback(self, msg: Odometry) -> None:
        """
        Callback to update the velocity of the car.

        Parameters:
            msg (Odometry): The Odometry message containing the car's velocity.
        """
        self.ego_velocity = msg.twist.twist.linear.x

    def _preprocess_lidar(self, ranges) -> np.ndarray:
        """
        Preprocess the LiDAR scan array.

        This method performs preprocessing on the LiDAR scan array. The preprocessing steps include:
        1. Setting each value to the mean over a specified window.
        2. Rejecting high values (e.g., values greater than 3m).

        Parameters:
            ranges (numpy.ndarray): The LiDAR scan array.

        Returns:
            numpy.ndarray: The preprocessed LiDAR scan array.
        """

        downsampled_ranges = np.array(ranges)[self.LIDAR_SAMPLE_IDXS]
        return downsampled_ranges[::1]

    def _get_best_range_point(self, proc_ranges) -> tuple:
        """
        Find the best point i.e. the middle of the largest gap within the bubble radius.

        Parameters:
            proc_ranges (list): List of processed ranges.

        Returns:
            tuple: The x and y coordinates of the best point.
        """
        # Get the bubble radius
        radius = self._get_radius()

        # Find the largest gap
        gap_left, gap_right = self._find_largest_gap(ranges=proc_ranges, radius=radius)
        gap_left += self.range_offset - 180
        gap_right += self.range_offset - 180
        gap_middle = int((gap_right + gap_left) / 2)
        # Calculate cartesian point of the best point position from the lidar measurements in laser frame
        best_x = np.cos(self.LIDAR_THETA_LUT[gap_middle]) * radius
        best_y = np.sin(self.LIDAR_THETA_LUT[gap_middle]) * radius

        if self.DEBUG:
            # Delete old gaps from RVIZ
            self._delete_gap_markers()

            # Visualise the gap
            gap_markers = MarkerArray()
            for i in range(gap_left, gap_right):
                mrk = Marker()
                mrk.header.frame_id = 'laser'
                mrk.header.stamp = rospy.Time.now()
                mrk.type = mrk.SPHERE
                mrk.scale.x = 0.05
                mrk.scale.y = 0.05
                mrk.scale.z = 0.05
                mrk.color.a = 1.0
                mrk.color.r = 1.0
                mrk.color.g = 1.0
                mrk.id = i - gap_left
                # Calculate cartesian point of the gap  marker position from the lidar measurements in laser frame
                mrk.pose.position.x = math.cos(self.LIDAR_THETA_LUT[i]) * radius
                mrk.pose.position.y = math.sin(self.LIDAR_THETA_LUT[i]) * radius

                mrk.pose.orientation.w = 1
                gap_markers.markers.append(mrk)
            self.best_gap.publish(gap_markers)

            # visualize best point aka middle of the gap
            best_mrk = Marker()
            best_mrk.header.frame_id = 'laser'
            best_mrk.header.stamp = rospy.Time.now()
            best_mrk.type = best_mrk.SPHERE
            best_mrk.scale.x = 0.2
            best_mrk.scale.y = 0.2
            best_mrk.scale.z = 0.2
            best_mrk.color.a = 1.0
            best_mrk.color.b = 1.0
            best_mrk.color.g = 1.0
            best_mrk.id = 0
            best_mrk.pose.position.y = best_y
            best_mrk.pose.position.x = best_x
            best_mrk.pose.orientation.w = 1
            self.best_pnt.publish(best_mrk)

        return best_x, best_y

    def process_lidar(self, ranges) -> tuple:
        """
        Process each LiDAR scan as per the Follow Gap algorithm &
        calculate the speed and steering angle.

        Parameters:
            ranges (list): List of LiDAR scan ranges

        Returns:
            tuple: A tuple containing the speed and steering angle
        """
        # Preprocess the LiDAR to smoothen it
        if self.LIDAR_SAMPLE_IDXS is None or self.LIDAR_THETA_LUT is None:
            angle_min = -135.0 * np.pi / 180
            angle_max = 135.0 * np.pi / 180
            self.LIDAR_SAMPLE_IDXS, self.LIDAR_THETA_LUT = get_boxed_indices(angle_min, angle_max, len(ranges), 41, 5.0)
            
        proc_ranges = self._preprocess_lidar(ranges)

        use_safety_borders = False
        if use_safety_borders:
            proc_ranges = self._safety_border(proc_ranges)

        if self.DEBUG:
            scan_markers = MarkerArray()
            for i, scan in enumerate(proc_ranges):
                mrk = Marker()
                mrk.header.frame_id = 'laser'
                mrk.header.stamp = rospy.Time.now()
                mrk.type = mrk.SPHERE
                mrk.scale.x = 0.05
                mrk.scale.y = 0.05
                mrk.scale.z = 0.05
                mrk.color.a = 1.0
                mrk.color.r = 1.0
                mrk.color.b = 1.0

                mrk.id = i
                mrk.pose.position.x = math.cos(self.LIDAR_THETA_LUT[i]) * scan
                mrk.pose.position.y = math.sin(self.LIDAR_THETA_LUT[i]) * scan

                mrk.pose.orientation.w = 1
                scan_markers.markers.append(mrk)
            self.scan_pub.publish(scan_markers)


        segment_idxs = self._left_right_bounds(ranges=proc_ranges, diff_threshold=self.diff_threshold)#1.5

        # Extract centerline from lidar scans
        # try:
        centerline, kappa = self._extract_centerline(proc_ranges, segment_idxs)
        # except Exception as e:
        #     rospy.logwarn(f"[FTG] Error in centerline extraction: {e}")
        #     centerline = None
        #     kappa = None

        if centerline is not None and len(centerline) > 0:
            lookahead_dist = self.lookahead_qv * self.ego_velocity + self.lookahead_qm
            # lookahead_dist = 1
            whb = 0.5
            distances = np.linalg.norm(centerline - np.array([0, 0]), axis=1)
            
            # 找到距离最接近lookahead_dist的点
            lookahead_idx = np.argmin(np.abs(distances - lookahead_dist))
            lookahead_point = centerline[lookahead_idx]
            
            eta = np.arctan2(lookahead_point[1], lookahead_point[0])
            steering_angle = np.arctan(2 * whb * np.sin(eta) / lookahead_dist)

            kappa_lookahead = kappa[max(lookahead_idx - 2, 0):min(lookahead_idx + 2, len(kappa))]
            kappa_lookahead = np.mean(kappa_lookahead)
            
            if not self.mapping:
                # 分段选择目标速度
                # if np.abs(kappa_lookahead) > 0.2:
                #     speed = self.CORNERS_SPEED
                # elif np.abs(kappa_lookahead) > 0.1:
                #     speed = self.MILD_CORNERS_SPEED
                # elif np.abs(kappa_lookahead) > 0.05:
                #     speed = self.STRAIGHTS_SPEED
                # else:
                #     speed = self.ULTRASTRAIGHTS_SPEED
                #     rospy.loginfo(f"speed = {speed}")
                # # 限制加减速度变化
                # speed_diff = speed - self.ego_velocity
                # if speed_diff < 0:
                #     speed_diff = max(speed_diff, -self.decel_limit)
                # else:
                #     speed_diff = min(speed_diff, self.accel_limit)
                speed = 1
            else:
                speed = 1.25
        else:
            speed = 0
            steering_angle = 0

        return speed, steering_angle


    def _left_right_bounds(self, ranges, diff_threshold=1.5) -> np.ndarray:
        first_ranges_deriv = np.diff(ranges)
        #pad the first element with the first element of the derivative
        first_ranges_deriv = np.insert(first_ranges_deriv, 0, first_ranges_deriv[0])

        # Segment ranges into pieces where the abs of the derivative is too high
        threshold = diff_threshold
        segment_idx = np.argwhere(np.abs(first_ranges_deriv) > threshold)

        return segment_idx


    def _extract_centerline(self, ranges, segment_idxs) -> np.ndarray:
        """
        Extract the centerline from the LiDAR scan data using Delaunay triangulation.

        Parameters:
            ranges (numpy.ndarray): The LiDAR scan data.

        Returns:
            numpy.ndarray: The centerline of the LiDAR scan data.
        """
        segment_mask = np.zeros(len(ranges), dtype=int)
        split_points = [segment[0] for segment in segment_idxs]
        # Assign segment indices based on split points
        if len(split_points) != 0:
            for i, (start, end) in enumerate(zip([0] + split_points, split_points)):
                segment_mask[start:end] = i
            # Handle the remaining part after the last segment
            segment_mask[split_points[-1]:] = len(split_points)
        else:
            rospy.logwarn("[FTG] No segments found!")

        if self.masking == "range":
            ranges_mask = ranges < self.MAX_LIDAR_DIST
            x = ranges[ranges_mask] * np.cos(self.LIDAR_THETA_LUT[ranges_mask])
            y = ranges[ranges_mask] * np.sin(self.LIDAR_THETA_LUT[ranges_mask])
        elif self.masking == "angle":
            angle_rad = 25 * np.pi / 180 # 25 degrees
            angle_mask = (self.LIDAR_THETA_LUT < -angle_rad) | (self.LIDAR_THETA_LUT > angle_rad)
            x = ranges[angle_mask] * np.cos(self.LIDAR_THETA_LUT[angle_mask])
            y = ranges[angle_mask] * np.sin(self.LIDAR_THETA_LUT[angle_mask])
        else:
            x = ranges * np.cos(self.LIDAR_THETA_LUT)
            y = ranges * np.sin(self.LIDAR_THETA_LUT)

        segment_mask = segment_mask[ranges_mask] # mask out the segments as well

        og_points = np.vstack((x, y)).T

        delaunay_midpoints, delaunay = calculate_centerline(og_points, segment_mask)
        # Filter out midpoints that are outside of the min and max of the lidar points
        delaunay_midpoints = delaunay_midpoints[(delaunay_midpoints[:, 0] > min(x)) & (
            delaunay_midpoints[:, 0] < max(x)) & (delaunay_midpoints[:, 1] > min(y)) & (delaunay_midpoints[:, 1] < max(y))]
        #Filter out all points with negative x
        delaunay_midpoints = delaunay_midpoints[delaunay_midpoints[:, 0] > 0]
        # Sort points based on their x-coordinates for consistency
        # Combine x and y coordinates into a 2D array
        centerline = np.array(delaunay_midpoints)
        if len(centerline) < 6:
            rospy.logwarn(f"[FTG] Short Raceline: {len(centerline)}!")
        points = np.vstack((centerline[:, 0], centerline[:, 1])).T

        try:
            # Find the point closest to the origin (0,0)
            origin = np.array([0, 0])
            start_idx = np.argmin(np.linalg.norm(points - origin, axis=1))
        except:
            rospy.logwarn(f"[FTG] No argmin found, POINTS: {len(points)}")
            return None, None

        # Create a mask to keep track of visited points
        visited = np.zeros(len(points), dtype=bool)
        visited[start_idx] = True

        # Vectorized path building
        sorted_points = [points[start_idx]]
        current_point = points[start_idx]

        i = 0

      
        while not np.all(visited) and i < 1000:
            # Determine whether the final step is interpolation
            did_interpolate = False
            # Vectorized computation of distances to all unvisited points
            distances = np.linalg.norm(points - current_point, axis=1)
            distances[visited] = np.inf  # Ignore already visited points by setting a large value
            
            # Filtering condition: based on projection in the x-direction
            if len(sorted_points) > 1:  
                prev_point = sorted_points[-2]  
                direction_vector = current_point - prev_point 
                
                # Compute projection for each unvisited point
                for j in range(len(points)):
                    if not visited[j]:
                        next_candidate = points[j]
                        to_candidate = next_candidate - current_point
                        
                        if np.linalg.norm(direction_vector) > 0: 
                            projection_length = np.dot(to_candidate, direction_vector) / np.linalg.norm(direction_vector)
                            
                            if projection_length < 0:
                                distances[j] = np.inf
            else:
                distances[points[:,0] - current_point[0] < -0.3] = np.inf
            
            # Find the nearest unvisited point
            next_idx = np.argmin(distances)
            if distances[next_idx] == np.inf:
                break

            next_point = points[next_idx]
            
            # Interpolation: if the distance > 1 meter, insert 5 intermediate points
            dist = np.linalg.norm(next_point - current_point)
            if dist > 1.0 :
                did_interpolate = True
                for t in np.linspace(0, 1, 7)[1:-1]:  
                    interp_point = (1 - t) * current_point + t * next_point
                    sorted_points.append(interp_point)
                    

            sorted_points.append(points[next_idx])
            visited[next_idx] = True
            current_point = points[next_idx]

            i += 1

        # Convert sorted points back to x and y arrays
        sorted_points = np.array(sorted_points)

        positive_points = np.linalg.norm(sorted_points - np.array([0, 0]), axis=1)     

        if did_interpolate and np.sum(positive_points > 0) >= 12:
            sorted_points = sorted_points[:-6]
        elif np.sum(positive_points > 0) >= 8:
            sorted_points = sorted_points[:-2]

        ccma = CCMA(w_ma=5, w_cc=3, distrib_ma="hanning", distrib_cc="pascal")

        if len(sorted_points) >= 3:
            sorted_points = ccma.filter(sorted_points, cc_mode=True)
        else:
            rospy.logwarn(f"[FTG] Not enough points for CCMA filtering.")

        
        sorted_x = sorted_points[:, 0]
        sorted_y = sorted_points[:, 1]

        window = min(5, len(sorted_x))
        if window % 2 == 0:
            window -= 1

        try:
            centerline_x_old = savgol_filter(sorted_x, window_length=window, polyorder=2)
            centerline_y_old = savgol_filter(sorted_y, window_length=window, polyorder=2)
            if self.RANSAC:
                points = [line2d.Point2D(x, y) for x, y in zip(centerline_x_old, centerline_y_old)]
                inliers = pyransac.find_inliers(points=points,
                                    model=self.pyrs_model,
                                    params=self.pyrs_params)
                centerline_x_postrs = [point.x for point in inliers]
                centerline_y_postrs = [point.y for point in inliers]
            else:
                centerline_x_postrs = centerline_x_old
                centerline_y_postrs = centerline_y_old
            # print(f"Ransac removed {len(centerline_x_old) - len(centerline_x)} points")
            #spline the centerline
            tck, u = splprep([centerline_x_postrs, centerline_y_postrs], s=0)
            u_new = np.linspace(u.min(), u.max(), 100)
            centerline_x, centerline_y = splev(u_new, tck)
            # Calculate the second derivatives
            centerline_der_x, centerline_der_y = splev(u_new, tck, der=1)
            centerline_der2_x, centerline_der2_y = splev(u_new, tck, der=2)

            # Calculate the curvature
            curvature = np.abs(centerline_der_x * centerline_der2_y - centerline_der_y * centerline_der2_x) / (centerline_der_x**2 + centerline_der_y**2)**1.5

            if self.DEBUG:
                self.ax.cla()

                # 设置标题和轴标签
                self.ax.set_title('Centerline Extraction Without CCMA')
                self.ax.set_xlabel('x [m]')
                self.ax.set_ylabel('y [m]')
                
                # 绘制Delaunay三角网格的边（注意使用self.ax而不是plt）
                self.ax.triplot(og_points[:, 0], og_points[:, 1], delaunay.simplices.copy())
                
                # 绘制Delaunay三角形的中点，红色
                self.ax.scatter(delaunay_midpoints[:, 0], delaunay_midpoints[:, 1], c='r', label='Midpoints')
                
                # 绘制原始点，默认颜色
                self.ax.scatter(og_points[:, 0], og_points[:, 1], c='black', s=10, alpha=0.5, label='Raw Points')

                # 绘制已通过RANSAC过滤的中心线点，绿色菱形
                self.ax.scatter(centerline_x_postrs, centerline_y_postrs, c='green', marker="d", label='Ransacced Centerline', alpha=0.5)
                
                # 绘制未通过RANSAC的中心线点，紫色叉形
                self.ax.scatter(centerline_x_old, centerline_y_old, c='purple', marker="x", label='Unransacced Centerline', alpha=0.5)

                # 绘制基于分段的LiDAR点，使用不同颜色（默认会使用区分的颜色）
                for i in range(len(segment_idxs) + 1):
                    self.ax.scatter(x[segment_mask == i], y[segment_mask == i], s=200)

                # 添加图例
                self.ax.legend()

                # 设置图形的坐标比例相等
                self.ax.set_aspect('equal')

                # 显示图形
                self.ax.figure.canvas.draw()
                self.ax.figure.canvas.flush_events()
                plt.pause(0.001)

            return np.vstack((centerline_x, centerline_y)).T, curvature
        except Exception as e:
            rospy.logwarn(f"Error in centerline extraction, could not spline the centerline")
            return None, None

    def _find_largest_gap(self, ranges, radius) -> tuple:
        """
        Find the index of the starting and ending of the largest gap and its width

        Parameters:
            ranges (numpy.ndarray): Array of range values
            radius (float): Threshold radius value

        Returns:
            tuple: A tuple containing the index of the starting of the largest gap,
                    the index of the ending of the largest gap, and the width of the largest gap.

        """
        # Binarise the ranges in zeros for values under the radius threshold and ones for above and equal
        bin_ranges = np.where(ranges >= radius, 1, 0)

        # Get largest gap from binary ranges
        bin_diffs = np.abs(np.diff(bin_ranges))
        bin_diffs[0] = 1
        bin_diffs[-1] = 1

        diff_idxs = bin_diffs.nonzero()[0]
        # Check that binarised ranges are positive
        high_gaps = []
        for i in range(len(diff_idxs) - 1):
            low = diff_idxs[i]
            high = diff_idxs[i + 1]
            high_gaps.append(np.mean(bin_ranges[low:high]) > 0.5)

        gap_left = diff_idxs[np.argmax(high_gaps * np.diff(diff_idxs))]
        gap_width = np.max(high_gaps * np.diff(diff_idxs))
        gap_right = gap_left + gap_width

        return gap_left, gap_right

    def _get_radius(self) -> float:
        """
        Calculate the radius based on the track width and velocity.

        Returns:
            float: The calculated radius.
        """
        # Empirically determined that this radius choosing makes sense
        return min(5., self.track_width / 2 + 2 * (self.velocity / self.MAX_SPEED))

    def set_vel(self, velocity) -> None:
        """
        Set the velocity of the car.

        Parameters:
            velocity (float): The desired velocity value.
        """
        self.velocity = velocity

    def _safety_border(self, ranges) -> np.ndarray:
        """
        Add a safety bubble if there is a big increase in the range between two points.

        Parameters:
            ranges (list): List of range values.

        Returns:
            np.ndarray: Array of filtered range values.
        """
        filtered = list(ranges)
        ranges_len = len(ranges)
        i = 0
        while i < ranges_len - 1:
            if ranges[i + 1] - ranges[i] > 0.5:
                for j in range(self.SAFETY_RADIUS):
                    if i + j < ranges_len:
                        filtered[i + j] = ranges[i]
                i += self.SAFETY_RADIUS - 2
            i += 1
        # in other direction
        i = ranges_len - 1
        while i > 0:
            if ranges[i - 1] - ranges[i] > 0.5:
                for j in range(self.SAFETY_RADIUS):
                    if i - j >= 0:
                        filtered[i - j] = ranges[i]
                i = i - self.SAFETY_RADIUS + 2
            i -= 1
        return np.array(filtered)

    def _delete_gap_markers(self) -> None:
        """
        Delete marker for rviz when not needed
        """
        del_mrk_array = MarkerArray()
        for i in range(1):
            del_mrk = Marker()
            del_mrk.header.frame_id = 'laser'
            del_mrk.header.stamp = rospy.Time.now()
            del_mrk.action = del_mrk.DELETEALL
            del_mrk.id = i
            del_mrk_array.markers.append(del_mrk)
        self.best_gap.publish(del_mrk_array)
