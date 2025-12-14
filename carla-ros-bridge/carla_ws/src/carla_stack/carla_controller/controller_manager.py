#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在原始控制器基础上，针对“深度相机在过弯时内侧赛道突然消失、视野翻到外侧远处”问题，
实现一套 **检测→触发→补偿→回落** 的“事件型记忆补偿”完整可运行方案，并带可视化与自检。

要点：
1) 不改变相机 FOV，仅在识别到“内侧丢失/翻转事件”时，对**内侧异常扇区**启用记忆补偿；
2) 记忆补偿采用**位姿配准的多帧加权中位数** + **可信度(Trust)与时效衰减**，避免记忆幻影；
3) 退出事件后**平滑回落**到实时观测；
4) 完整 RViz 调试话题与 JSON 指标便于调参；
5) 与原版接口/控制流程兼容，可直接替换。

主要调试话题（RViz 友好）：
- ~debug/scan_lidar     (sensor_msgs/LaserScan)
- ~debug/scan_depth     (sensor_msgs/LaserScan)
- ~debug/scan_active    (sensor_msgs/LaserScan)  # 当前源，融合前
- ~debug/scan_fused     (sensor_msgs/LaserScan)  # 事件型补偿后（用于控制）
- ~debug/markers        (visualization_msgs/MarkerArray)  # 当前/融合/历史虚拟/补偿扇区
- ~debug/memory_info    (std_msgs/String)  # JSON 指标：事件判定、增益、覆盖率、信任等

运行：
- rosrun your_pkg controller_manager_with_memory_viz.py
- RViz Frame 设为 base_link，订阅上述话题对比观察。
"""

import numpy as np
import threading
import math
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import LaserScan, PointCloud2, Image, CameraInfo
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import String
from delaunay_ftg.delaunay_ftg import DelaunayFTG
from delaunay.delaunay import DelaunayFTG as Delaunay
from carla_msgs.msg import CarlaEgoVehicleControl
import tf2_ros
from tf.transformations import quaternion_matrix, euler_from_quaternion


class Controller_manager:
    """
    改进点（与原版相比）：
    - 深度→伪2D + LiDAR→伪2D，保留；
    - 历史帧 SE(2) 配准，保留；
    - **事件型记忆补偿**（本次新增核心）：
        · 检测“内侧大面积突然远化 + 正在转弯（外侧远处增强/内侧无效暴增）” → 触发事件；
        · 仅对内侧异常扇区做“多帧加权中位”记忆补偿；
        · 为每个角bin维护 trust（可信度），随时间衰减，有真实观测一致性则回升；
        · 退出事件后，α(记忆权重)随 trust 平滑减小，回落到实时观测；
        · 附带几何/物理与平滑约束，避免“不可能”的补偿形状。
    """

    class FakeScan:
        def __init__(self, ranges: np.ndarray):
            self.ranges = ranges

    def __init__(self):
        self.name = "control_node"
        rospy.init_node(self.name, anonymous=True)
        self.lock = threading.Lock()
        self.loop_rate = rospy.get_param("~loop_rate", 200)  # Hz

        # 控制器
        self.delaunay_ftg_controller = DelaunayFTG(mapping=False, debug=True)
        self.delaunay_controller = Delaunay(mapping=False, debug=True)
        self.ctrl_algo = rospy.get_param("~ctrl_algo", "DELAUNAY_FTG")

        # 数据源优先：True=深度相机，False=LiDAR；若不可用自动回退
        self.use_depth = rospy.get_param("~depth", False)

        # 事件型记忆配置
        self.enable_comp = rospy.get_param("~enable_comp", True)
        self.comp_frames = int(rospy.get_param("~comp_frames", 3))      # 融合最近 K 帧
        self.fusion_mode = rospy.get_param("~fusion_mode", "event_only")  # event_only | global_min（可选）

        # 伪雷达角域
        self.num_bins = int(rospy.get_param("~num_bins", 270))
        self.angle_min = float(rospy.get_param("~angle_min", -135.0))
        self.angle_max = float(rospy.get_param("~angle_max",  135.0))
        self.max_range = float(rospy.get_param("~max_range",   30.0))
        self.bin_size = (self.angle_max - self.angle_min) / self.num_bins
        self.theta_centers = self.angle_min + (np.arange(self.num_bins) + 0.5) * self.bin_size  # deg

        # 内/外侧扇区定义
        self.inner_left = rospy.get_param("~inner_left_deg",  [-135.0, -15.0])  # 左转内侧
        self.inner_right = rospy.get_param("~inner_right_deg", [15.0, 135.0])   # 右转内侧（镜像）
        self.center_exclude = float(rospy.get_param("~center_exclude_deg", 15.0))

        # 高度窗（世界系 Z）
        self.height_center = float(rospy.get_param("~height_center", 0.5))   # m
        self.height_halfwin = float(rospy.get_param("~height_halfwin", 0.5)) # m

        # 深度相机采样
        self.sample_step = int(rospy.get_param("~sample_step", 2))

        # 车辆动力学/速度控制参数（保留）
        self.accel_limit = rospy.get_param("~accel_limit", 2.0)
        self.decel_limit = rospy.get_param("~decel_limit", 2.0)
        self.friction_coeff = rospy.get_param("~friction_coeff", 0.8)
        self.friction_scale = rospy.get_param("~friction_scale", 1.0)

        # 调试参数
        self.debug = rospy.get_param("~debug", True)
        self.viz_rate_hz = float(rospy.get_param("~viz_rate_hz", 15.0))
        self.last_viz_pub = rospy.Time(0)
        self.marker_lifetime = rospy.get_param("~marker_lifetime", 0.15)  # s

        # ---- 事件检测阈值（可调） ----
        self.delta_far = float(rospy.get_param("~delta_far", 2.0))       # 与参考相比“远化”阈值
        self.tau_far_abs = float(rospy.get_param("~tau_far_abs", 10.0))  # 绝对远距离阈值
        self.ratio_far_thr = float(rospy.get_param("~ratio_far_thr", 0.6))
        self.tau_out_far = float(rospy.get_param("~tau_out_far", 15.0))
        self.ratio_out_far_delta_thr = float(rospy.get_param("~ratio_out_far_delta_thr", 0.2))
        self.tau_yaw = math.radians(rospy.get_param("~tau_yaw_deg", 5.0))
        self.tau_yaw_hys = math.radians(rospy.get_param("~tau_yaw_hys_deg", 3.0))

        # ---- 记忆融合参数 ----
        self.alpha_max = float(rospy.get_param("~alpha_max", 0.85))
        self.gamma_event = float(rospy.get_param("~gamma_event", 1.5))
        self.tau_time = float(rospy.get_param("~tau_time", 0.8))  # s，历史帧时效
        self.sigma_yaw = math.radians(rospy.get_param("~sigma_yaw_deg", 5.0))
        self.trust_decay = float(rospy.get_param("~trust_decay", 2.0))  # s，信任衰减常数
        self.eta_fast = float(rospy.get_param("~eta_fast", 0.3))        # 真实近距出现时的快速回信任
        self.r_phys_min = float(rospy.get_param("~r_phys_min", 1.0))
        self.tv_factor = float(rospy.get_param("~tv_factor", 1.5))       # TV 平滑约束倍数

        # 运行态数据
        self.scan = None            # 用于控制（事件补偿后）
        self.scan_lidar = None      # LiDAR 伪扫描
        self.scan_depth = None      # 深度伪扫描
        self.scan_active = None     # 当前源（融合前）
        self.scan_virtuals = []     # 最近用于补偿的虚拟扫描（配准到当前）
        self.speed_now = 0.0
        self.pose_xyyaw = (0.0, 0.0, 0.0)
        self.yaw_rate = 0.0
        self.last_odom_stamp = None
        self.active_source_name = "none"
        self.used_fusion = False
        self.prev_fused_ranges = None

        self.history = []
        # 事件状态 & trust
        self.event_active = False
        self.event_good_count = 0  # 退出滞回计数
        self.trust = np.full(self.num_bins, 0.3, dtype=np.float32)
        self.last_tick = rospy.Time.now()

        # 发布器
        self.vel_pub = rospy.Publisher('/carla/ego_vehicle/vehicle_control_cmd', CarlaEgoVehicleControl, queue_size=1)
        self.pub_scan_lidar = rospy.Publisher("~debug/scan_lidar", LaserScan, queue_size=1)
        self.pub_scan_depth = rospy.Publisher("~debug/scan_depth", LaserScan, queue_size=1)
        self.pub_scan_active = rospy.Publisher("~debug/scan_active", LaserScan, queue_size=1)
        self.pub_scan_fused = rospy.Publisher("~debug/scan_fused", LaserScan, queue_size=1)
        self.pub_markers = rospy.Publisher("~debug/markers", MarkerArray, queue_size=1)
        self.pub_info = rospy.Publisher("~debug/memory_info", String, queue_size=5)

        # PID 参数（沿用）
        self.Kp_T, self.Ki_T, self.Kd_T = 0.4, 0.005, 0.0
        self.prev_error_T, self.integral_T, self.derivative_T = 0, 0, 0
        self.Kp_B, self.Ki_B, self.Kd_B = 0.00005, 0.0, 0.0
        self.prev_error_B, self.integral_B, self.derivative_B = 0, 0, 0

        # 订阅
        rospy.Subscriber("/carla/ego_vehicle/lidar", PointCloud2, self.lidar_cb, queue_size=1)
        rospy.Subscriber("/carla/ego_vehicle/depth_front/camera_info", CameraInfo, self.camera_info_cb, queue_size=1)
        rospy.Subscriber("/carla/ego_vehicle/depth_front/image", Image, self.depth_cb, queue_size=1)
        rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry, self.odom_cb, queue_size=50)

        # 相机内参缓存
        self.fx = self.fy = self.cx = self.cy = None
        self.img_w = self.img_h = None
        self.cam_frame = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    # ---------------- LiDAR → 伪2D ----------------
    def lidar_cb(self, data: PointCloud2):
        try:
            pts = np.fromiter(
                pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True),
                dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)]
            )
            if pts.size == 0:
                self.scan_lidar = None
                return
            xyz = np.stack([pts['x'], pts['y'], pts['z']], axis=1)

            z = xyz[:, 2]
            mask = np.abs(z - self.height_center) < self.height_halfwin
            filtered = xyz[mask]
            if filtered.shape[0] == 0:
                self.scan_lidar = None
                return

            x = filtered[:, 0]
            y = filtered[:, 1]
            dist = np.hypot(x, y)
            ang = np.degrees(np.arctan2(y, x))

            valid = (ang >= self.angle_min) & (ang <= self.angle_max)
            dist, ang = dist[valid], ang[valid]

            ranges = np.full(self.num_bins, np.inf, dtype=np.float32)
            idx = ((ang - self.angle_min) / self.bin_size).astype(np.int32)
            for i, d in zip(idx, dist):
                if 0 <= i < self.num_bins and d < ranges[i]:
                    ranges[i] = d

            self._fill_holes_inplace(ranges, self.max_range)
            self.scan_lidar = self.FakeScan(ranges)
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[lidar_cb] {e}")

    # ---------------- 相机参数 ----------------
    def camera_info_cb(self, msg: CameraInfo):
        self.fx, self.fy = msg.K[0], msg.K[4]
        self.cx, self.cy = msg.K[2], msg.K[5]
        self.img_w, self.img_h = msg.width, msg.height
        self.cam_frame = msg.header.frame_id

    # ---------------- 深度 → 伪2D ----------------
    def depth_cb(self, msg: Image):
        if self.fx is None or self.img_w is None:
            return
        if msg.encoding in ('32FC1', '32FC'):
            Z = np.frombuffer(msg.data, dtype=np.float32)
        elif msg.encoding == '16UC1':
            Z = np.frombuffer(msg.data, dtype=np.uint16).astype(np.float32) / 1000.0
        else:
            return
        try:
            Z = Z.reshape(msg.height, msg.width)
        except Exception:
            return

        step = max(1, self.sample_step)
        Z = Z[::step, ::step]
        h, w = Z.shape
        if h == 0 or w == 0:
            return

        Z = np.clip(Z, 0.05, self.max_range)
        valid = np.isfinite(Z) & (Z > 0.05)
        if not np.any(valid):
            self.scan_depth = None
            return

        uu, vv = np.meshgrid(
            (np.arange(w) * step).astype(np.float32),
            (np.arange(h) * step).astype(np.float32)
        )

        Xc = (uu - self.cx) / self.fx * Z
        Yc = (vv - self.cy) / self.fy * Z
        Zc = Z

        Xc, Yc, Zc = Xc[valid], Yc[valid], Zc[valid]

        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", self.cam_frame, msg.header.stamp, rospy.Duration(0.01)
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
            tvec = np.array([t.x, t.y, t.z]).reshape(3, 1)
            P = np.vstack([Xc, Yc, Zc])
            Pb = R.dot(P) + tvec
            Xb, Yb, Zb = Pb[0], Pb[1], Pb[2]
        except Exception:
            # 退化近似
            Xb = Zc
            Yb = -Xc
            Zb = -Yc

        sel = (Zb >= self.height_center - self.height_halfwin) & \
              (Zb <= self.height_center + self.height_halfwin)
        if not np.any(sel):
            self.scan_depth = None
            return
        Xb, Yb = Xb[sel], Yb[sel]

        r = np.hypot(Xb, Yb)
        theta = np.degrees(np.arctan2(Yb, Xb))
        m = (theta >= self.angle_min) & (theta <= self.angle_max)
        r, theta = r[m], theta[m]
        if r.size == 0:
            self.scan_depth = None
            return

        ranges = np.full(self.num_bins, np.inf, dtype=np.float32)
        idx = ((theta - self.angle_min) / self.bin_size).astype(np.int32)
        for i, d in zip(idx, r):
            if 0 <= i < self.num_bins and d < ranges[i]:
                ranges[i] = d

        self._fill_holes_inplace(ranges, self.max_range)
        self.scan_depth = self.FakeScan(ranges)

    # ---------------- 工具：空洞插值 ----------------
    def _fill_holes_inplace(self, ranges: np.ndarray, fill_val: float):
        n = ranges.size
        inf_mask = np.isinf(ranges)
        if not np.any(inf_mask):
            return
        for i in np.where(inf_mask)[0]:
            left = ranges[i - 1] if i > 0 else np.nan
            right = ranges[i + 1] if i < n - 1 else np.nan
            if np.isfinite(left) and np.isfinite(right):
                ranges[i] = 0.5 * (left + right)
            elif np.isfinite(left):
                ranges[i] = left
            elif np.isfinite(right):
                ranges[i] = right
            else:
                ranges[i] = fill_val
        np.clip(ranges, 0.0, self.max_range, out=ranges)

    # ---------------- 数据源选择 + 准备历史虚拟 ----------------
    def _pick_active_scan(self):
        self.scan_virtuals = []
        self.used_fusion = False

        primary = self.scan_depth if self.use_depth else self.scan_lidar
        backup = self.scan_lidar if self.use_depth else self.scan_depth

        if primary is not None:
            scan = primary
            self.active_source_name = 'depth' if self.use_depth else 'lidar'
        elif backup is not None:
            scan = backup
            self.active_source_name = 'lidar' if self.use_depth else 'depth'
        else:
            scan = None
            self.active_source_name = 'none'

        self.scan_active = scan

        # 仅准备历史虚拟（不做全局 min 融合；事件时再选择性使用）
        if self.enable_comp and scan is not None and len(self.history) > 0:
            cx, cy, cyaw = self.pose_xyyaw
            hist_used = self.history[-self.comp_frames:]
            for item in hist_used:
                virt = self._warp_ranges(item["ranges"], item["pose"][0], item["pose"][1], item["pose"][2],
                                         cx, cy, cyaw)
                self.scan_virtuals.append({
                    "ranges": virt,
                    "yaw_rate": item.get("yaw_rate", 0.0),
                    "stamp": item.get("stamp", None)
                })

        # 初步设定：控制输入依据当前源（若事件触发将在后续替换）
        self.scan = scan

        # 维护历史：推入“当前真实帧”（融合前）
        if scan is not None:
            ranges_to_push = scan.ranges
            self._push_history(ranges_to_push, self.pose_xyyaw, self.yaw_rate, rospy.Time.now())

    def _push_history(self, ranges: np.ndarray, pose_xyyaw, yaw_rate: float, stamp):
        if not hasattr(self, 'history'):
            self.history = []
        self.history.append({
            "ranges": ranges.copy(),
            "pose": pose_xyyaw,
            "yaw_rate": float(yaw_rate),
            "stamp": stamp
        })
        max_keep = max(self.comp_frames * 2, 8)
        if len(self.history) > max_keep:
            self.history = self.history[-max_keep:]

    def _warp_ranges(self, ranges_prev: np.ndarray,
                     px, py, pyaw, cx, cy, cyaw) -> np.ndarray:
        thetas = np.deg2rad(self.theta_centers)
        r = np.clip(ranges_prev, 0.0, self.max_range)
        x_prev = r * np.cos(thetas)
        y_prev = r * np.sin(thetas)
        dtheta = cyaw - pyaw
        c, s = np.cos(dtheta), np.sin(dtheta)
        dx = cx - px
        dy = cy - py
        x_cur = c * x_prev - s * y_prev + dx
        y_cur = s * x_prev + c * y_prev + dy
        r_cur = np.hypot(x_cur, y_cur)
        th_cur = np.rad2deg(np.arctan2(y_cur, x_cur))
        m = (th_cur >= self.angle_min) & (th_cur <= self.angle_max) & (r_cur > 0.01)
        if not np.any(m):
            return np.full(self.num_bins, np.inf, dtype=np.float32)
        idx = ((th_cur[m] - self.angle_min) / self.bin_size).astype(np.int32)
        virt = np.full(self.num_bins, np.inf, dtype=np.float32)
        for i, d in zip(idx, r_cur[m]):
            if 0 <= i < self.num_bins and d < virt[i]:
                virt[i] = d
        self._fill_holes_inplace(virt, self.max_range)
        return virt

    # ---------------- 里程计：维护速度、位姿与 yaw_rate ----------------
    def odom_cb(self, data: Odometry):
        self.speed_now = data.twist.twist.linear.x
        p = data.pose.pose.position
        q = data.pose.pose.orientation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.pose_xyyaw = (p.x, p.y, yaw)
        # 优先使用提供的角速度；若没有可改为差分 yaw
        self.yaw_rate = data.twist.twist.angular.z
        self.last_odom_stamp = data.header.stamp

    # ---------------- PID 与发布（保留） ----------------
    def publish_target_velocity(self, PID_signalT, PID_signalB, angular_z=0.0):
        msg = CarlaEgoVehicleControl()
        msg.throttle = PID_signalT
        msg.steer = -angular_z
        msg.brake = PID_signalB
        msg.hand_brake = False
        msg.reverse = False
        self.vel_pub.publish(msg)

    def PID_ComputeT(self, target_speed, current_speed):
        error = target_speed - current_speed
        P = self.Kp_T * error
        self.integral_T += error
        I = self.Ki_T * self.integral_T
        self.derivative_T = error - self.prev_error_T if self.prev_error_T != 0 else 0
        D = self.Kd_T * self.derivative_T
        self.prev_error_T = error
        return np.clip(P + I + D, 0, 1)

    def PID_ComputeB(self, target_speed, current_speed):
        error = target_speed - current_speed
        P = self.Kp_B * error
        self.integral_B += error
        I = self.Ki_B * self.integral_B
        self.derivative_B = error - self.prev_error_B if self.prev_error_B != 0 else 0
        D = self.Kd_B * self.derivative_B
        self.prev_error_B = error
        return np.abs(np.clip(P + I + D, -1, 0))

    # ---------------- 可视化与自检 ----------------
    def _ranges_to_scan_msg(self, ranges: np.ndarray) -> LaserScan:
        msg = LaserScan()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.angle_min = math.radians(self.angle_min)
        msg.angle_max = math.radians(self.angle_max)
        msg.angle_increment = math.radians(self.bin_size)
        msg.time_increment = 0.0
        msg.scan_time = 1.0 / max(1.0, self.loop_rate)
        msg.range_min = 0.05
        msg.range_max = float(self.max_range)
        rr = np.clip(np.nan_to_num(ranges, nan=np.inf), 0.0, self.max_range)
        msg.ranges = [float(x) if np.isfinite(x) else float('inf') for x in rr]
        return msg

    def _polar_points(self, ranges: np.ndarray):
        thetas = np.deg2rad(self.theta_centers)
        r = np.clip(np.nan_to_num(ranges, nan=np.inf), 0.0, self.max_range)
        x = r * np.cos(thetas)
        y = r * np.sin(thetas)
        pts = []
        for xi, yi, ri in zip(x, y, r):
            if np.isfinite(ri) and ri < self.max_range - 1e-3:
                p = Point(); p.x, p.y, p.z = float(xi), float(yi), 0.0
                pts.append(p)
        return pts

    def _publish_markers(self, inner_mask=None):
        ma = MarkerArray()
        now = rospy.Time.now()
        lifetime = rospy.Duration(self.marker_lifetime)

        def add_points(ns, mid, pts, rgba, scale=0.06, mtype=Marker.POINTS):
            m = Marker()
            m.header.frame_id = "base_link"; m.header.stamp = now
            m.ns = ns; m.id = mid; m.type = mtype; m.action = Marker.ADD
            m.scale.x = scale; m.scale.y = scale
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.points = pts
            m.lifetime = lifetime
            ma.markers.append(m)

        # 当前源（蓝）
        if self.scan_active is not None:
            add_points("active", 0, self._polar_points(self.scan_active.ranges), (0.15, 0.45, 1.0, 0.9))
        # 融合后（红）
        if self.scan is not None:
            add_points("fused", 1, self._polar_points(self.scan.ranges), (1.0, 0.2, 0.25, 0.9))
        # 历史虚拟（绿）
        for k, v in enumerate(self.scan_virtuals):
            alpha = max(0.25, 0.9 - 0.25 * k)
            add_points("virt", 10 + k, self._polar_points(v["ranges"]), (0.2, 1.0, 0.4, alpha))
        # 被补偿的内侧扇区（橙色线带）
        if inner_mask is not None and np.any(inner_mask):
            pts = []
            thetas = np.deg2rad(self.theta_centers[inner_mask])
            r = np.full_like(thetas, 0.8)  # 画在车体前 0.8m 的弧段
            for th in thetas:
                p = Point(); p.x, p.y, p.z = float(r[0]*np.cos(th)), float(r[0]*np.sin(th)), 0.0
                pts.append(p)
            add_points("inner", 99, pts, (1.0, 0.6, 0.1, 0.9), scale=0.03, mtype=Marker.LINE_STRIP)

        self.pub_markers.publish(ma)

    def _publish_debug_topics(self, extra_info=None, inner_mask=None):
        if self.scan_lidar is not None: self.pub_scan_lidar.publish(self._ranges_to_scan_msg(self.scan_lidar.ranges))
        if self.scan_depth is not None: self.pub_scan_depth.publish(self._ranges_to_scan_msg(self.scan_depth.ranges))
        if self.scan_active is not None: self.pub_scan_active.publish(self._ranges_to_scan_msg(self.scan_active.ranges))
        if self.scan is not None: self.pub_scan_fused.publish(self._ranges_to_scan_msg(self.scan.ranges))
        self._publish_markers(inner_mask)

        info = {
            "source": self.active_source_name,
            "speed": round(float(self.speed_now), 4),
            "history_len": len(getattr(self, 'history', [])),
            "comp_enabled": bool(self.enable_comp),
            "comp_frames": int(self.comp_frames),
            "used_fusion": bool(self.used_fusion),
            "event_active": bool(self.event_active),
        }
        if extra_info: info.update(extra_info)
        self.pub_info.publish(String(data=str(info)))

    # ---------------- 扇区掩码（内/外侧） ----------------
    def _sector_masks(self, yaw_rate):
        # 左转内侧区域
        il0, il1 = self.inner_left
        ir0, ir1 = self.inner_right
        th = self.theta_centers
        excl = (np.abs(th) <= self.center_exclude)
        left_mask = (th >= il0) & (th <= il1) & (~excl)
        right_mask = (th >= ir0) & (th <= ir1) & (~excl)
        if yaw_rate >= 0.0:   # 左转：内侧=left，外侧=right
            return left_mask, right_mask
        else:                 # 右转：内侧=right，外侧=left
            return right_mask, left_mask

    # ---------------- 事件检测 ----------------
    def _detect_inner_flip_event(self, r_now: np.ndarray, r_ref: np.ndarray):
        yaw_rate = float(self.yaw_rate)
        inner_mask, outer_mask = self._sector_masks(yaw_rate)
        if r_now is None or r_ref is None:
            return False, inner_mask, outer_mask, {}

        # 大幅远化（内侧）
        far_jump = ((r_now - r_ref) > self.delta_far) & (r_now > self.tau_far_abs)
        ratio_far_in = float(np.mean(far_jump[inner_mask])) if np.any(inner_mask) else 0.0

        # 外侧远处增强
        ratio_out_far_now = float(np.mean((r_now > self.tau_out_far)[outer_mask])) if np.any(outer_mask) else 0.0
        ratio_out_far_prev = float(np.mean((r_ref > self.tau_out_far)[outer_mask])) if np.any(outer_mask) else 0.0
        ratio_out_far_delta = ratio_out_far_now - ratio_out_far_prev

        # 角速度门限
        turning = abs(yaw_rate) > self.tau_yaw

        # 内侧无效暴增（可选，简单用 Inf 判断）
        invalid_now = np.isinf(r_now) | (r_now >= self.max_range - 1e-3)
        invalid_ref = np.isinf(r_ref) | (r_ref >= self.max_range - 1e-3)
        inv_in_now = float(np.mean(invalid_now[inner_mask])) if np.any(inner_mask) else 0.0
        inv_in_prev = float(np.mean(invalid_ref[inner_mask])) if np.any(inner_mask) else 0.0
        inv_in_delta = inv_in_now - inv_in_prev

        condA = ratio_far_in > self.ratio_far_thr
        condB = ratio_out_far_delta > self.ratio_out_far_delta_thr
        condC = turning
        condD = inv_in_delta > 0.3
        event = (condA and condC) and (condB or condD)

        info = {
            "ratio_far_in": round(ratio_far_in, 3),
            "ratio_out_far_now": round(ratio_out_far_now, 3),
            "ratio_out_far_delta": round(ratio_out_far_delta, 3),
            "inv_in_now": round(inv_in_now, 3),
            "inv_in_delta": round(inv_in_delta, 3),
            "turning": bool(turning)
        }
        return event, inner_mask, outer_mask, info

    # ---------------- 加权中位工具 ----------------
    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray):
        m = np.isfinite(values)
        if not np.any(m):
            return np.inf
        v = values[m]; w = weights[m]
        if v.size == 1:
            return float(v[0])
        order = np.argsort(v)
        v = v[order]; w = w[order]
        cdf = np.cumsum(w) / (np.sum(w) + 1e-9)
        idx = np.searchsorted(cdf, 0.5)
        idx = min(idx, v.size - 1)
        return float(v[idx])

    # ---------------- 事件内侧补偿 ----------------
    def _compensate_inner_side(self, r_now: np.ndarray):
        # 构造历史帧权重
        if len(self.scan_virtuals) == 0:
            return r_now.copy(), None, {"alpha_mean_in": 0.0, "trust_mean_in": float(np.mean(self.trust))}
        now_time = rospy.Time.now()
        vals = [v["ranges"] for v in self.scan_virtuals]
        yaw_hist = [v.get("yaw_rate", 0.0) for v in self.scan_virtuals]
        t_hist = [v.get("stamp", None) for v in self.scan_virtuals]
        w_list = []
        for yk, tk in zip(yaw_hist, t_hist):
            w_yaw = math.exp(-abs(float(self.yaw_rate) - float(yk)) / (self.sigma_yaw + 1e-6))
            if tk is None:
                w_time = 1.0
            else:
                dt = max(0.0, (now_time - tk).to_sec())
                w_time = math.exp(-dt / (self.tau_time + 1e-6))
            w_list.append(w_yaw * w_time)
        w_arr = np.array(w_list, dtype=np.float32)
        if np.all(w_arr <= 1e-6):
            w_arr = np.ones_like(w_arr)

        # 事件内：内/外侧掩码（再次取，用于标注）
        inner_mask, _ = self._sector_masks(self.yaw_rate)

        # 计算 TV 基线（当前内侧）
        def total_variation(arr):
            a = arr[np.isfinite(arr)]
            if a.size < 2: return 0.0
            return float(np.sum(np.abs(np.diff(a))))
        tv_now_in = total_variation(r_now[inner_mask])

        # 信任衰减（全域）
        dt_tick = (rospy.Time.now() - self.last_tick).to_sec()
        if dt_tick > 0:
            self.trust *= math.exp(-dt_tick / max(1e-3, self.trust_decay))
            self.trust = np.clip(self.trust, 0.0, 1.0)
            self.last_tick = rospy.Time.now()

        # 逐 bin 融合
        r_fused = r_now.copy()
        far_jump = ((r_now - (self.prev_fused_ranges if self.prev_fused_ranges is not None else r_now)) > self.delta_far) & (r_now > self.tau_far_abs)
        inner_abnormal = inner_mask & far_jump

        alphas = []
        for i in np.where(inner_abnormal)[0]:
            # 聚集历史候选
            cand = np.array([vals[k][i] for k in range(len(vals))], dtype=np.float32)
            # 物理下界
            cand = np.maximum(cand, self.r_phys_min)
            wm = self._weighted_median(cand, w_arr)
            if not np.isfinite(wm):
                continue
            # 可信度→记忆权重
            alpha = min(self.alpha_max, float(self.trust[i] * self.gamma_event))
            if np.isfinite(r_now[i]) and (r_now[i] + 0.5 < wm):
                # 当前明显更近，信任快速回升，采用当前
                self.trust[i] = min(1.0, self.trust[i] + self.eta_fast)
                r_fused[i] = r_now[i]
            else:
                # 记忆与当前加权
                r_fused[i] = alpha * wm + (1.0 - alpha) * r_now[i]
                # 有一致性则慢速回升
                if np.isfinite(r_now[i]) and abs(r_now[i] - wm) < 1.0:
                    self.trust[i] = min(1.0, self.trust[i] + 0.05)
            alphas.append(alpha)

        # TV 约束：若补偿后 TV 过大，退让
        tv_new_in = total_variation(r_fused[inner_mask])
        if tv_now_in > 0 and tv_new_in > self.tv_factor * tv_now_in:
            # 过度锯齿，缩小补偿幅度
            r_fused[inner_abnormal] = 0.5 * (r_fused[inner_abnormal] + r_now[inner_abnormal])

        alpha_mean = float(np.mean(alphas)) if len(alphas) > 0 else 0.0
        info = {"alpha_mean_in": round(alpha_mean, 3), "trust_mean_in": round(float(np.mean(self.trust[inner_mask])) if np.any(inner_mask) else float(np.mean(self.trust)), 3)}
        return r_fused, inner_abnormal, info

    # ---------------- 主循环 ----------------
    def control_loop(self):
        rate = rospy.Rate(self.loop_rate)
        rospy.wait_for_message('/carla/ego_vehicle/odometry', Odometry)
        rospy.loginfo(f"[{self.name}] Ready with EVENT-BASED memory compensation!")

        last_viz_time = rospy.Time.now()
        viz_period = 1.0 / max(1.0, self.viz_rate_hz)

        while not rospy.is_shutdown():
            # 选择数据源并准备历史虚拟帧
            self._pick_active_scan()

            inner_mask_viz = None
            extra_info = {}

            # 事件检测 & 选择性补偿
            if self.enable_comp and self.scan_active is not None:
                r_now = self.scan_active.ranges
                r_ref = self.prev_fused_ranges if self.prev_fused_ranges is not None else r_now
                event, inner_mask, outer_mask, det_info = self._detect_inner_flip_event(r_now, r_ref)

                # 滞回：进入用 tau_yaw，退出用 tau_yaw_hys & 连续低 ratio_far
                if event:
                    self.event_active = True
                    self.event_good_count = 0
                else:
                    if self.event_active:
                        # 退出条件：内侧远化比例降低 or 角速度回落
                        if det_info.get("ratio_far_in", 0.0) < 0.3 or abs(self.yaw_rate) < self.tau_yaw_hys:
                            self.event_good_count += 1
                        else:
                            self.event_good_count = 0
                        if self.event_good_count >= 3:
                            self.event_active = False
                            self.event_good_count = 0
                    else:
                        self.event_active = False

                if self.event_active:
                    fused, inner_abnormal, comp_info = self._compensate_inner_side(r_now)
                    self.scan = self.FakeScan(fused)
                    self.used_fusion = True
                    inner_mask_viz = inner_abnormal
                    extra_info.update(det_info)
                    extra_info.update(comp_info)
                else:
                    # 非事件：可选全局最小融合或直接用当前
                    if self.fusion_mode == "global_min" and len(self.scan_virtuals) > 0:
                        fused = r_now.copy()
                        for v in self.scan_virtuals:
                            fused = np.minimum(fused, v["ranges"])
                        self.scan = self.FakeScan(fused)
                        self.used_fusion = True
                    else:
                        self.scan = self.scan_active
                        self.used_fusion = False
                    extra_info.update(det_info)

            # 控制器调用
            speed_cmd, steer_cmd = 0.0, 0.0
            try:
                if self.scan is not None:
                    if self.ctrl_algo == "DELAUNAY_FTG":
                        self.delaunay_ftg_controller.set_vel(self.speed_now)
                        speed_cmd, steer_cmd = self.delaunay_ftg_controller.process_lidar(self.scan.ranges)
                    elif self.ctrl_algo == "DELAUNAY":
                        self.delaunay_controller.set_vel(self.speed_now)
                        speed_cmd, steer_cmd = self.delaunay_controller.process_lidar(self.scan.ranges)
                    else:
                        rospy.logwarn_throttle(1.0, f"[{self.name}] Unknown ctrl_algo: {self.ctrl_algo}")
                PID_T = self.PID_ComputeT(speed_cmd, self.speed_now)
                PID_B = self.PID_ComputeB(speed_cmd, self.speed_now)
                self.publish_target_velocity(PID_T, PID_B, steer_cmd)
            except Exception as e:
                rospy.logerr(f"[{self.name}] control_loop error: {e}")
                self.publish_target_velocity(0.0, 0.0, 0.0)

            # 可视化发布（限频）
            if self.debug:
                now = rospy.Time.now()
                if (now - last_viz_time).to_sec() >= viz_period:
                    self._publish_debug_topics(extra_info, inner_mask_viz)
                    last_viz_time = now

            # 保存当前融合结果作为下一帧参考
            if self.scan is not None:
                self.prev_fused_ranges = self.scan.ranges.copy()

            rate.sleep()


if __name__ == "__main__":
    cm = Controller_manager()
    cm.control_loop()
