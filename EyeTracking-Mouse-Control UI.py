import sys
import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import socket
import argparse
import time
import csv
from datetime import datetime
import os
import pyautogui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                            QCheckBox, QVBoxLayout, QHBoxLayout, QWidget,
                            QGroupBox, QFormLayout, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex
from PyQt5.QtGui import QImage, QPixmap

class AngleBuffer:
    def __init__(self, size=10):
        self.size = size
        self.buffer = []

    def add(self, angles):
        self.buffer.append(angles)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def get_average(self):
        if not self.buffer:
            return (0, 0, 0)
        return tuple(np.mean(self.buffer, axis=0))

class EyeTrackingThread(QThread):
    frame_updated = pyqtSignal(QImage)
    blink_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    calibration_updated = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        # 核心参数
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = pyautogui.size()
        self.last_blink_time = 0
        self.SMOOTH_MOUSE = True
        self.prev_total_blinks = 0
        self.BLINK_INTERVAL_THRESHOLD = 500

        # 灵敏度参数
        self.EYE_CONTROL_SENSITIVITY = 0.8
        self.MOUSE_SMOOTH_FACTOR = 0.2

        # 眨眼检测参数
        self.BLINK_THRESHOLD = 0.53
        self.EYE_AR_CONSEC_FRAMES = 3
        self.IS_BLINKING = False
        self.BLINK_RECOVERY_FRAMES = 4
        self.blink_recovery_counter = 0

        # 瞳孔校准参数
        self.CALIBRATED = False
        self.LEFT_EYE_CALIBRATION = {"center_x": 0, "center_y": 0, "width": 0, "height": 0}
        self.RIGHT_EYE_CALIBRATION = {"center_x": 0, "center_y": 0, "width": 0, "height": 0}

        # 其他基础参数
        self.USER_FACE_WIDTH = 140  # [mm]
        self.NOSE_TO_CAMERA_DISTANCE = 600  # [mm]
        self.PRINT_DATA = False
        self.DEFAULT_WEBCAM = 0
        self.SHOW_ALL_FEATURES = False
        self.LOG_DATA = True
        self.LOG_ALL_FEATURES = False
        self.ENABLE_HEAD_POSE = True
        self.LOG_FOLDER = "logs"
        self.SERVER_IP = "127.0.0.1"
        self.SERVER_PORT = 7070
        self.SHOW_ON_SCREEN_DATA = True
        self.TOTAL_BLINKS = 0
        self.EYES_BLINK_FRAME_COUNTER = 0
        self.NOSE_TIP_INDEX = 4
        self.CHIN_INDEX = 152
        self.LEFT_EYE_LEFT_CORNER_INDEX = 33
        self.RIGHT_EYE_RIGHT_CORNER_INDEX = 263
        self.LEFT_MOUTH_CORNER_INDEX = 61
        self.RIGHT_MOUTH_CORNER_INDEX = 291

        self.MIN_DETECTION_CONFIDENCE = 0.8
        self.MIN_TRACKING_CONFIDENCE = 0.8
        self.MOVING_AVERAGE_WINDOW = 10
        self.initial_pitch, self.initial_yaw, self.initial_roll = None, None, None
        self.SHOW_BLINK_COUNT_ON_SCREEN = True
        self.IS_RECORDING = False
        self._indices_pose = [1, 33, 61, 199, 263, 291]
        self.SERVER_ADDRESS = (self.SERVER_IP, self.SERVER_PORT)

        # 新增控制参数
        self.eye_control_enabled = False  # 修改：默认不启用眼动控制

        # 特征点索引定义
        self.LEFT_EYE_IRIS = [474, 475, 476, 477]
        self.RIGHT_EYE_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE_OUTER_CORNER = [33]
        self.LEFT_EYE_INNER_CORNER = [133]
        self.RIGHT_EYE_OUTER_CORNER = [362]
        self.RIGHT_EYE_INNER_CORNER = [263]
        self.RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
        self.LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]

        # 初始化
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE,
        )
        self.cam_source = 0
        self.cap = None
        self.running = False
        self.angle_buffer = AngleBuffer(size=self.MOVING_AVERAGE_WINDOW)
        self.prev_mouse_x, self.prev_mouse_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        self.iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.csv_data = []

        # 新增：帧缓存和线程锁（解决校准资源冲突）
        self.latest_frame = None
        self.frame_lock = QMutex()

        if not os.path.exists(self.LOG_FOLDER):
            os.makedirs(self.LOG_FOLDER)

        self.column_names = [
            "Timestamp (ms)", "Left Eye Center X", "Left Eye Center Y",
            "Right Eye Center X", "Right Eye Center Y",
            "Left Iris Relative Pos Dx", "Left Iris Relative Pos Dy",
            "Right Iris Relative Pos Dx", "Right Iris Relative Pos Dy",
            "Total Blink Count",
        ]
        if self.ENABLE_HEAD_POSE:
            self.column_names.extend(["Pitch", "Yaw", "Roll"])
        if self.LOG_ALL_FEATURES:
            self.column_names.extend([f"Landmark_{i}_X" for i in range(468)] + [f"Landmark_{i}_Y" for i in range(468)])

    def run(self):
        self.running = True
        self.cap = cv.VideoCapture(self.cam_source)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # 缓存最新帧（加锁保证线程安全）
            self.frame_lock.lock()
            self.latest_frame = frame.copy()
            self.frame_lock.unlock()

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = self.mp_face_mesh.process(rgb_frame)
            pitch, yaw, roll = 0, 0, 0

            if results.multi_face_landmarks:
                mesh_points = np.array(
                    [np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                     for p in results.multi_face_landmarks[0].landmark]
                )

                mesh_points_3D = np.array(
                    [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
                )
                head_pose_points_3D = np.multiply(
                    mesh_points_3D[self._indices_pose], [img_w, img_h, 1]
                )
                head_pose_points_2D = mesh_points[self._indices_pose]

                nose_3D_point = np.multiply(head_pose_points_3D[0], [1, 1, 3000])
                nose_2D_point = head_pose_points_2D[0]

                focal_length = 1 * img_w
                cam_matrix = np.array(
                    [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
                )
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                head_pose_points_2D = np.delete(head_pose_points_3D, 2, axis=1)
                head_pose_points_3D = head_pose_points_3D.astype(np.float64)
                head_pose_points_2D = head_pose_points_2D.astype(np.float64)
                success, rot_vec, trans_vec = cv.solvePnP(
                    head_pose_points_3D, head_pose_points_2D, cam_matrix, dist_matrix
                )
                rotation_matrix, jac = cv.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)

                angle_x = angles[0] * 360
                angle_y = angles[1] * 360
                z = angles[2] * 360

                threshold_angle = 10
                face_looks = "Forward"
                if angle_y < -threshold_angle:
                    face_looks = "Left"
                elif angle_y > threshold_angle:
                    face_looks = "Right"
                elif angle_x < -threshold_angle:
                    face_looks = "Down"
                elif angle_x > threshold_angle:
                    face_looks = "Up"

                if self.SHOW_ON_SCREEN_DATA:
                    cv.putText(
                        frame, f"Face: {face_looks}",
                        (img_w - 250, 80), cv.FONT_HERSHEY_TRIPLEX,
                        0.8, (0, 255, 0), 2, cv.LINE_AA
                    )

                nose_3d_projection, jacobian = cv.projectPoints(
                    nose_3D_point, rot_vec, trans_vec, cam_matrix, dist_matrix
                )
                p1 = nose_2D_point
                p2 = (int(nose_2D_point[0] + angle_y * 10), int(nose_2D_point[1] - angle_x * 10))
                cv.line(frame, p1, p2, (255, 0, 255), 3)

                # 眨眼检测逻辑
                eyes_aspect_ratio = self.blinking_ratio(mesh_points_3D)
                if eyes_aspect_ratio <= self.BLINK_THRESHOLD:
                    self.EYES_BLINK_FRAME_COUNTER += 1
                    self.IS_BLINKING = True
                    self.blink_recovery_counter = 0
                else:
                    if self.EYES_BLINK_FRAME_COUNTER > self.EYE_AR_CONSEC_FRAMES:
                        self.TOTAL_BLINKS += 1
                        self.blink_updated.emit(self.TOTAL_BLINKS)
                    self.EYES_BLINK_FRAME_COUNTER = 0
                    if self.IS_BLINKING:
                        self.blink_recovery_counter += 1
                        if self.blink_recovery_counter >= self.BLINK_RECOVERY_FRAMES:
                            self.IS_BLINKING = False
                            self.blink_recovery_counter = 0

                # 绘制眼球特征点
                if self.SHOW_ALL_FEATURES:
                    for point in mesh_points:
                        cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)

                # 获取虹膜和眼角坐标
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[self.LEFT_EYE_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[self.RIGHT_EYE_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)

                left_outer_corner = mesh_points[self.LEFT_EYE_OUTER_CORNER][0]
                left_inner_corner = mesh_points[self.LEFT_EYE_INNER_CORNER][0]
                right_outer_corner = mesh_points[self.RIGHT_EYE_OUTER_CORNER][0]
                right_inner_corner = mesh_points[self.RIGHT_EYE_INNER_CORNER][0]

                # 绘制虹膜和眼角标记
                cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA)
                cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA)
                cv.circle(frame, left_inner_corner, 3, (255, 255, 255), -1, cv.LINE_AA)
                cv.circle(frame, left_outer_corner, 3, (0, 255, 255), -1, cv.LINE_AA)
                cv.circle(frame, right_inner_corner, 3, (255, 255, 255), -1, cv.LINE_AA)
                cv.circle(frame, right_outer_corner, 3, (0, 255, 255), -1, cv.LINE_AA)

                # 计算相对位置
                l_dx, l_dy = self.vector_position(left_outer_corner, center_left)
                r_dx, r_dy = self.vector_position(right_outer_corner, center_right)

                # 头部姿态估计
                if self.ENABLE_HEAD_POSE:
                    pitch, yaw, roll = self.estimate_head_pose(mesh_points, (img_h, img_w))
                    self.angle_buffer.add([pitch, yaw, roll])
                    pitch, yaw, roll = self.angle_buffer.get_average()

                    if self.initial_pitch is None:
                        self.initial_pitch, self.initial_yaw, self.initial_roll = pitch, yaw, roll

                    pitch -= self.initial_pitch
                    yaw -= self.initial_yaw
                    roll -= self.initial_roll

                # 鼠标控制（仅在眼动控制启用时生效）
                current_time = int(time.time() * 1000)
                if not self.IS_BLINKING and self.blink_recovery_counter == 0 and self.eye_control_enabled:
                    target_x, target_y = self.map_pupil_direction_to_screen(
                        center_left, center_right,
                        left_outer_corner, left_inner_corner,
                        right_outer_corner, right_inner_corner
                    )
                    # 平滑移动
                    smooth_x = int(self.prev_mouse_x * (1 - self.MOUSE_SMOOTH_FACTOR) + target_x * self.MOUSE_SMOOTH_FACTOR)
                    smooth_y = int(self.prev_mouse_y * (1 - self.MOUSE_SMOOTH_FACTOR) + target_y * self.MOUSE_SMOOTH_FACTOR)
                    pyautogui.moveTo(smooth_x, smooth_y, duration=0.05)
                    self.prev_mouse_x, self.prev_mouse_y = smooth_x, smooth_y

                # 处理单双击（仅在眼动控制启用时生效）
                if self.TOTAL_BLINKS > self.prev_total_blinks and self.eye_control_enabled:
                    self.handle_blink(current_time)
                    self.prev_total_blinks = self.TOTAL_BLINKS

                # 日志记录与UDP传输
                if self.LOG_DATA and self.IS_RECORDING:
                    log_entry = [
                        current_time, l_cx, l_cy, r_cx, r_cy,
                        l_dx, l_dy, r_dx, r_dy, self.TOTAL_BLINKS
                    ]
                    if self.ENABLE_HEAD_POSE:
                        log_entry.extend([pitch, yaw, roll])
                    if self.LOG_ALL_FEATURES:
                        log_entry.extend([p for point in mesh_points for p in point])
                    self.csv_data.append(log_entry)

                packet = np.array([current_time], dtype=np.int64).tobytes() + \
                         np.array([l_cx, l_cy, l_dx, l_dy], dtype=np.int32).tobytes()
                self.iris_socket.sendto(packet, self.SERVER_ADDRESS)

                # 屏幕显示状态信息
                if self.SHOW_ON_SCREEN_DATA:
                    if self.IS_RECORDING:
                        cv.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                    cv.putText(frame, f"Blinks: {self.TOTAL_BLINKS}", (30, 80),
                               cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    if self.ENABLE_HEAD_POSE:
                        cv.putText(frame, f"Pitch: {int(pitch)}", (30, 110),
                                   cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                        cv.putText(frame, f"Yaw: {int(yaw)}", (30, 140),
                                   cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

                    cv.putText(frame, f"Range: Full Screen", (30, 170),
                               cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2, cv.LINE_AA)
                    cv.putText(frame, f"Calib: {'Done' if self.CALIBRATED else 'Press K'}", (30, 200),
                               cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0) if self.CALIBRATED else (0, 0, 255), 2, cv.LINE_AA)
                    cv.putText(frame, f"Control: {'On' if self.eye_control_enabled else 'Off'}", (30, 230),
                               cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0) if self.eye_control_enabled else (0, 0, 255), 2, cv.LINE_AA)

            # 转换为QImage并发送信号
            rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_updated.emit(q_image)

            # 状态更新
            status = f"眼动控制: {'开启' if self.eye_control_enabled else '关闭'} | 校准: {'完成' if self.CALIBRATED else '未完成'}"
            self.status_updated.emit(status)

        # 清理资源
        if self.cap:
            self.cap.release()
        self.iris_socket.close()

        # 保存日志
        if self.LOG_DATA and self.IS_RECORDING and self.csv_data:
            timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            csv_file_name = os.path.join(self.LOG_FOLDER, f"eye_tracking_log_{timestamp_str}.csv")
            with open(csv_file_name, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self.column_names)
                writer.writerows(self.csv_data)

    def stop(self):
        self.running = False
        self.wait()

    # 核心功能函数
    def calibrate_eye_position(self, left_outer, left_inner, right_outer, right_inner, left_iris, right_iris):
        left_eye_width = np.linalg.norm(left_outer - left_inner)
        left_eye_height = (np.linalg.norm(left_iris - left_outer) + np.linalg.norm(left_iris - left_inner)) / 2
        right_eye_width = np.linalg.norm(right_outer - right_inner)
        right_eye_height = (np.linalg.norm(right_iris - right_outer) + np.linalg.norm(right_iris - right_inner)) / 2

        self.LEFT_EYE_CALIBRATION = {
            "center_x": left_iris[0], "center_y": left_iris[1],
            "width": left_eye_width, "height": left_eye_height
        }
        self.RIGHT_EYE_CALIBRATION = {
            "center_x": right_iris[0], "center_y": right_iris[1],
            "width": right_eye_width, "height": right_eye_height
        }
        self.CALIBRATED = True
        self.calibration_updated.emit(True)

    def map_pupil_direction_to_screen(self, left_iris, right_iris, left_outer, left_inner, right_outer, right_inner):
        if not self.CALIBRATED:
            return self.map_eye_to_screen((left_iris[0] + right_iris[0])/2,
                                    (left_iris[1] + right_iris[1])/2,
                                    self.SCREEN_WIDTH, self.SCREEN_HEIGHT)

        # 计算瞳孔相对眼睛中心的偏移（归一化）
        left_x = (left_iris[0] - self.LEFT_EYE_CALIBRATION["center_x"]) / (self.LEFT_EYE_CALIBRATION["width"] / 2)
        left_y = (left_iris[1] - self.LEFT_EYE_CALIBRATION["center_y"]) / (self.LEFT_EYE_CALIBRATION["height"] / 2)
        right_x = (right_iris[0] - self.RIGHT_EYE_CALIBRATION["center_x"]) / (self.RIGHT_EYE_CALIBRATION["width"] / 2)
        right_y = (right_iris[1] - self.RIGHT_EYE_CALIBRATION["center_y"]) / (self.RIGHT_EYE_CALIBRATION["height"] / 2)

        # 应用灵敏度系数 + 方向镜像
        avg_x = -((left_x + right_x) / 2 * self.EYE_CONTROL_SENSITIVITY)
        avg_y = ((left_y + right_y) / 2 * self.EYE_CONTROL_SENSITIVITY)

        # 解除范围限制
        avg_x = max(-1.0, min(1.0, avg_x))
        avg_y = max(-1.0, min(1.0, avg_y))

        # 映射到屏幕坐标
        screen_x = self.SCREEN_WIDTH / 2 + (avg_x * self.SCREEN_WIDTH / 2)
        screen_y = self.SCREEN_HEIGHT / 2 + (avg_y * self.SCREEN_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def map_eye_to_screen(self, eye_x, eye_y, cam_width, cam_height):
        screen_x = self.SCREEN_WIDTH - (eye_x / cam_width) * self.SCREEN_WIDTH * self.EYE_CONTROL_SENSITIVITY
        screen_x = self.SCREEN_WIDTH - screen_x  # 水平镜像
        screen_y = (eye_y / cam_height) * self.SCREEN_HEIGHT * self.EYE_CONTROL_SENSITIVITY
        screen_y = self.SCREEN_HEIGHT - screen_y  # 垂直镜像
        return max(0, min(self.SCREEN_WIDTH, int(screen_x))), max(0, min(self.SCREEN_HEIGHT, int(screen_y)))

    def handle_blink(self, current_time):
        blink_interval = current_time - self.last_blink_time

        if 0 < blink_interval <= self.BLINK_INTERVAL_THRESHOLD and self.eye_control_enabled:
            pyautogui.doubleClick()
        elif (blink_interval > self.BLINK_INTERVAL_THRESHOLD or self.last_blink_time == 0) and self.eye_control_enabled:
            pyautogui.click()

        self.last_blink_time = current_time

    # 辅助工具函数
    def vector_position(self, point1, point2):
        x1, y1 = point1.ravel()
        x2, y2 = point2.ravel()
        return x2 - x1, y2 - y1

    def euclidean_distance_3D(self, points):
        P0, P3, P4, P5, P8, P11, P12, P13 = points
        numerator = (np.linalg.norm(P3 - P13)**3 + np.linalg.norm(P4 - P12)** 3 + np.linalg.norm(P5 - P11)**3)
        denominator = 3 * np.linalg.norm(P0 - P8)** 3
        return numerator / denominator

    def estimate_head_pose(self, landmarks, image_size):
        scale_factor = self.USER_FACE_WIDTH / 150.0
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0 * scale_factor, -65.0 * scale_factor),
            (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),
            (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),
            (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),
            (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)
        ])

        focal_length = image_size[1]
        center = (image_size[1] / 2, image_size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))
        image_points = np.array([
            landmarks[self.NOSE_TIP_INDEX], landmarks[self.CHIN_INDEX],
            landmarks[self.LEFT_EYE_LEFT_CORNER_INDEX], landmarks[self.RIGHT_EYE_RIGHT_CORNER_INDEX],
            landmarks[self.LEFT_MOUTH_CORNER_INDEX], landmarks[self.RIGHT_MOUTH_CORNER_INDEX]
        ], dtype="double")

        (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                                                 flags=cv.SOLVEPNP_ITERATIVE)
        rotation_matrix, _ = cv.Rodrigues(rotation_vector)
        projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
        _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
        pitch, yaw, roll = euler_angles.flatten()[:3]
        pitch = self.normalize_pitch(pitch)
        return pitch, yaw, roll

    def normalize_pitch(self, pitch):
        if pitch > 180:
            pitch -= 360
        pitch = -pitch
        if pitch < -90:
            pitch = -(180 + pitch)
        elif pitch > 90:
            pitch = 180 - pitch
        pitch = -pitch
        return pitch

    def blinking_ratio(self, landmarks):
        right_eye_ratio = self.euclidean_distance_3D(landmarks[self.RIGHT_EYE_POINTS])
        left_eye_ratio = self.euclidean_distance_3D(landmarks[self.LEFT_EYE_POINTS])
        return (right_eye_ratio + left_eye_ratio + 1) / 2

    # UI控制接口 - 修复校准功能
    def toggle_eye_control(self, enabled):
        self.eye_control_enabled = enabled

    def calibrate(self):
        # 从缓存获取最新帧（线程安全）
        self.frame_lock.lock()
        frame = self.latest_frame
        self.frame_lock.unlock()

        if frame is None:
            return False  # 无有效帧数据

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = self.mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                 for p in results.multi_face_landmarks[0].landmark]
            )

            (l_cx, l_cy), _ = cv.minEnclosingCircle(mesh_points[self.LEFT_EYE_IRIS])
            (r_cx, r_cy), _ = cv.minEnclosingCircle(mesh_points[self.RIGHT_EYE_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            self.calibrate_eye_position(
                mesh_points[self.LEFT_EYE_OUTER_CORNER][0],
                mesh_points[self.LEFT_EYE_INNER_CORNER][0],
                mesh_points[self.RIGHT_EYE_OUTER_CORNER][0],
                mesh_points[self.RIGHT_EYE_INNER_CORNER][0],
                center_left,
                center_right
            )
            return True
        return False

    def toggle_recording(self):
        self.IS_RECORDING = not self.IS_RECORDING
        return self.IS_RECORDING


class EyeTrackingUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("眼动控制鼠标")
        self.setGeometry(100, 100, 1024, 768)

        # 创建主部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧：摄像头显示
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.camera_label, 3)

        # 右侧：控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel, 1)

        # 状态信息
        status_group = QGroupBox("状态信息")
        status_layout = QFormLayout()
        self.status_label = QLabel("等待启动...")
        self.blink_label = QLabel("0")
        self.calibration_label = QLabel("未完成")
        status_layout.addRow("系统状态:", self.status_label)
        status_layout.addRow("眨眼次数:", self.blink_label)
        status_layout.addRow("校准状态:", self.calibration_label)
        status_group.setLayout(status_layout)
        control_layout.addWidget(status_group)

        # 控制按钮
        control_group = QGroupBox("控制")
        control_layout_inner = QVBoxLayout()

        # 眼动控制开关（默认关闭）
        self.control_switch = QCheckBox("启用眼动控制鼠标")
        self.control_switch.setChecked(False)  # 修改：默认不选中

        # 校准按钮
        self.calibrate_btn = QPushButton("校准眼睛位置")
        self.calibrate_btn.clicked.connect(self.calibrate_eyes)

        # 录制按钮
        self.record_btn = QPushButton("开始录制数据")
        self.record_btn.clicked.connect(self.toggle_recording)

        # 添加到布局
        control_layout_inner.addWidget(self.control_switch)
        control_layout_inner.addWidget(self.calibrate_btn)
        control_layout_inner.addWidget(self.record_btn)
        control_group.setLayout(control_layout_inner)
        control_layout.addWidget(control_group)

        # 空白填充
        control_layout.addStretch()

        # 底部信息
        info_label = QLabel("按Q退出程序\n注视屏幕中心后点击校准按钮\n按K键也可触发校准")
        info_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(info_label)

        # 初始化眼动追踪线程
        self.tracking_thread = EyeTrackingThread()
        self.tracking_thread.frame_updated.connect(self.update_frame)
        self.tracking_thread.blink_updated.connect(self.update_blink_count)
        self.tracking_thread.status_updated.connect(self.update_status)
        self.tracking_thread.calibration_updated.connect(self.update_calibration_status)
        self.control_switch.stateChanged.connect(self.on_control_switch_changed)

        # 启动线程
        self.tracking_thread.start()

    def update_frame(self, q_image):
        # 显示摄像头画面
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)

    def update_blink_count(self, count):
        self.blink_label.setText(str(count))

    def update_status(self, status):
        self.status_label.setText(status)

    def update_calibration_status(self, calibrated):
        self.calibration_label.setText("完成" if calibrated else "未完成")
        color = "green" if calibrated else "red"
        self.calibration_label.setStyleSheet(f"color: {color}")

    def on_control_switch_changed(self, state):
        enabled = (state == Qt.Checked)
        self.tracking_thread.toggle_eye_control(enabled)

    def calibrate_eyes(self):
        # 检查摄像头是否打开
        if not self.tracking_thread.cap or not self.tracking_thread.cap.isOpened():
            QMessageBox.warning(self, "校准失败", "摄像头未打开，请检查设备")
            return

        # 检查是否有有效帧
        self.tracking_thread.frame_lock.lock()
        has_frame = self.tracking_thread.latest_frame is not None
        self.tracking_thread.frame_lock.unlock()

        if not has_frame:
            QMessageBox.warning(self, "校准失败", "未获取到摄像头画面，请确保摄像头正常工作")
            return

        # 执行校准
        if self.tracking_thread.calibrate():
            QMessageBox.information(self, "校准完成", "眼睛位置校准已完成！")
        else:
            QMessageBox.warning(self, "校准失败", "未检测到面部特征，请正对摄像头并确保光线充足")

    def toggle_recording(self):
        is_recording = self.tracking_thread.toggle_recording()
        if is_recording:
            self.record_btn.setText("停止录制数据")
            QMessageBox.information(self, "开始录制", "数据录制已开始")
        else:
            self.record_btn.setText("开始录制数据")
            QMessageBox.information(self, "停止录制", "数据录制已停止")

    # 新增：支持K键校准（兼容原有操作习惯）
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_K:
            self.calibrate_eyes()
        elif event.key() == Qt.Key_Q:
            self.close()

    def closeEvent(self, event):
        # 关闭窗口时停止线程
        self.tracking_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeTrackingUI()
    window.show()
    sys.exit(app.exec_())
