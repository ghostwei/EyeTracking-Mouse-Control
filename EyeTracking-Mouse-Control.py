# 第一步：先屏蔽所有冗余日志（放在代码最顶部，彻底消除TensorFlow/MediaPipe提示）
import os

# 屏蔽TensorFlow oneDNN提示
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 屏蔽TensorFlow所有INFO/WARNING/ERROR日志（仅保留致命错误）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 屏蔽MediaPipe所有冗余日志
os.environ['GLOG_minloglevel'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

# 正常导入所有依赖库
import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import socket
import argparse
import time
import csv
from datetime import datetime
import pyautogui


# 嵌入AngleBuffer类（避免缺少文件报错）
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
            return 0, 0, 0
        return np.mean(self.buffer, axis=0).tolist()


# -----------------------------------------------------------------------------------------------------------------------------------
# 核心参数
# -----------------------------------------------------------------------------------------------------------------------------------
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
last_blink_time = 0
SMOOTH_MOUSE = True
prev_total_blinks = 0
BLINK_INTERVAL_THRESHOLD = 500

EYE_CONTROL_SENSITIVITY = 0.8
MOUSE_SMOOTH_FACTOR = 0.2

BLINK_THRESHOLD = 0.53
EYE_AR_CONSEC_FRAMES = 3
IS_BLINKING = False
BLINK_RECOVERY_FRAMES = 4
blink_recovery_counter = 0

CALIBRATED = False
LEFT_EYE_CALIBRATION = {"center_x": 0, "center_y": 0, "width": 0, "height": 0}
RIGHT_EYE_CALIBRATION = {"center_x": 0, "center_y": 0, "width": 0, "height": 0}

# -----------------------------------------------------------------------------------------------------------------------------------
# 其他基础参数（优化摄像头配置）
# -----------------------------------------------------------------------------------------------------------------------------------
USER_FACE_WIDTH = 140  # [mm]
NOSE_TO_CAMERA_DISTANCE = 600  # [mm]
PRINT_DATA = True  # 开启自定义日志
DEFAULT_WEBCAM = 0  # 默认摄像头索引，可手动改为1/2测试
SHOW_ALL_FEATURES = False
LOG_DATA = True
LOG_ALL_FEATURES = False
ENABLE_HEAD_POSE = True
LOG_FOLDER = "logs"
SERVER_IP = "127.0.0.1"
SERVER_PORT = 7070
SHOW_ON_SCREEN_DATA = True
TOTAL_BLINKS = 0
EYES_BLINK_FRAME_COUNTER = 0
NOSE_TIP_INDEX = 4
CHIN_INDEX = 152
LEFT_EYE_LEFT_CORNER_INDEX = 33
RIGHT_EYE_RIGHT_CORNER_INDEX = 263
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291

MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8
MOVING_AVERAGE_WINDOW = 10
initial_pitch, initial_yaw, initial_roll = None, None, None
SHOW_BLINK_COUNT_ON_SCREEN = True
IS_RECORDING = False
_indices_pose = [1, 33, 61, 199, 263, 291]
SERVER_ADDRESS = (SERVER_IP, SERVER_PORT)

# 摄像头与窗口放大参数
CAM_WIDTH = 1280
CAM_HEIGHT = 720
SCALE_FACTOR = 1.5
NEW_WINDOW_WIDTH = int(CAM_WIDTH * SCALE_FACTOR)
NEW_WINDOW_HEIGHT = int(CAM_HEIGHT * SCALE_FACTOR)

# 命令行参数
parser = argparse.ArgumentParser(description="Eye Tracking with Mouse Control")
parser.add_argument("-c", "--camSource", help="Source of camera", default=str(DEFAULT_WEBCAM))
args = parser.parse_args()


# -----------------------------------------------------------------------------------------------------------------------------------
# 核心功能函数
# -----------------------------------------------------------------------------------------------------------------------------------
def calibrate_eye_position(left_outer, left_inner, right_outer, right_inner, left_iris, right_iris):
    global LEFT_EYE_CALIBRATION, RIGHT_EYE_CALIBRATION, CALIBRATED
    left_eye_width = np.linalg.norm(left_outer - left_inner)
    left_eye_height = (np.linalg.norm(left_iris - left_outer) + np.linalg.norm(left_iris - left_inner)) / 2
    right_eye_width = np.linalg.norm(right_outer - right_inner)
    right_eye_height = (np.linalg.norm(right_iris - right_outer) + np.linalg.norm(right_iris - right_inner)) / 2

    LEFT_EYE_CALIBRATION = {
        "center_x": left_iris[0], "center_y": left_iris[1],
        "width": left_eye_width, "height": left_eye_height
    }
    RIGHT_EYE_CALIBRATION = {
        "center_x": right_iris[0], "center_y": right_iris[1],
        "width": right_eye_width, "height": right_eye_height
    }
    CALIBRATED = True
    if PRINT_DATA:
        print("✅ 眼睛校准完成")


def map_pupil_direction_to_screen(left_iris, right_iris, left_outer, left_inner, right_outer, right_inner):
    if not CALIBRATED:
        return map_eye_to_screen((left_iris[0] + right_iris[0]) / 2,
                                 (left_iris[1] + right_iris[1]) / 2,
                                 SCREEN_WIDTH, SCREEN_HEIGHT)

    left_x = (left_iris[0] - LEFT_EYE_CALIBRATION["center_x"]) / (LEFT_EYE_CALIBRATION["width"] / 2)
    left_y = (left_iris[1] - LEFT_EYE_CALIBRATION["center_y"]) / (LEFT_EYE_CALIBRATION["height"] / 2)
    right_x = (right_iris[0] - RIGHT_EYE_CALIBRATION["center_x"]) / (RIGHT_EYE_CALIBRATION["width"] / 2)
    right_y = (right_iris[1] - RIGHT_EYE_CALIBRATION["center_y"]) / (RIGHT_EYE_CALIBRATION["height"] / 2)

    avg_x = -((left_x + right_x) / 2 * EYE_CONTROL_SENSITIVITY)
    avg_y = ((left_y + right_y) / 2 * EYE_CONTROL_SENSITIVITY)

    avg_x = max(-1.0, min(1.0, avg_x))
    avg_y = max(-1.0, min(1.0, avg_y))

    screen_x = SCREEN_WIDTH / 2 + (avg_x * SCREEN_WIDTH / 2)
    screen_y = SCREEN_HEIGHT / 2 + (avg_y * SCREEN_HEIGHT / 2)
    return int(screen_x), int(screen_y)


def map_eye_to_screen(eye_x, eye_y, cam_width, cam_height):
    screen_x = SCREEN_WIDTH - (eye_x / cam_width) * SCREEN_WIDTH * EYE_CONTROL_SENSITIVITY
    screen_x = SCREEN_WIDTH - screen_x
    screen_y = (eye_y / cam_height) * SCREEN_HEIGHT * EYE_CONTROL_SENSITIVITY
    screen_y = SCREEN_HEIGHT - screen_y
    return max(0, min(SCREEN_WIDTH, int(screen_x))), max(0, min(SCREEN_HEIGHT, int(screen_y)))


def handle_blink(current_time):
    global last_blink_time
    blink_interval = current_time - last_blink_time

    if 0 < blink_interval <= BLINK_INTERVAL_THRESHOLD:
        pyautogui.doubleClick()
        if PRINT_DATA:
            print(f"✅ 双击触发（间隔{blink_interval}ms）")
    elif blink_interval > BLINK_INTERVAL_THRESHOLD or last_blink_time == 0:
        pyautogui.click()
        if PRINT_DATA:
            status = "首次" if last_blink_time == 0 else f"{blink_interval}"
            print(f"✅ 单击触发（间隔{status}ms）")

    last_blink_time = current_time


# -----------------------------------------------------------------------------------------------------------------------------------
# 辅助工具函数
# -----------------------------------------------------------------------------------------------------------------------------------
def vector_position(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return x2 - x1, y2 - y1


def euclidean_distance_3D(points):
    P0, P3, P4, P5, P8, P11, P12, P13 = points
    numerator = (np.linalg.norm(P3 - P13) ** 3 + np.linalg.norm(P4 - P12) ** 3 + np.linalg.norm(P5 - P11) ** 3)
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3
    return numerator / denominator


def estimate_head_pose(landmarks, image_size):
    scale_factor = USER_FACE_WIDTH / 150.0
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
        landmarks[NOSE_TIP_INDEX], landmarks[CHIN_INDEX],
        landmarks[LEFT_EYE_LEFT_CORNER_INDEX], landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],
        landmarks[LEFT_MOUTH_CORNER_INDEX], landmarks[RIGHT_MOUTH_CORNER_INDEX]
    ], dtype="double")

    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                                                 flags=cv.SOLVEPNP_ITERATIVE)
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
    pitch, yaw, roll = euler_angles.flatten()[:3]
    pitch = normalize_pitch(pitch)
    return pitch, yaw, roll


def normalize_pitch(pitch):
    if pitch > 180:
        pitch -= 360
    pitch = -pitch
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch
    pitch = -pitch
    return pitch


def blinking_ratio(landmarks):
    right_eye_ratio = euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])
    left_eye_ratio = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])
    return (right_eye_ratio + left_eye_ratio + 1) / 2


# -----------------------------------------------------------------------------------------------------------------------------------
# 特征点索引定义
# -----------------------------------------------------------------------------------------------------------------------------------
LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER_CORNER = [33]
LEFT_EYE_INNER_CORNER = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]

# -----------------------------------------------------------------------------------------------------------------------------------
# 初始化（增加摄像头可用性检查）
# -----------------------------------------------------------------------------------------------------------------------------------
if PRINT_DATA:
    print("=" * 50)
    print("✅ 初始化眼动追踪模块...")
    print(f"📺 屏幕分辨率: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print(f"🎯 眼动控制灵敏度: {EYE_CONTROL_SENSITIVITY}")
    print(f"🖼️  窗口放大尺寸: {NEW_WINDOW_WIDTH}x{NEW_WINDOW_HEIGHT}")
    print(f"📷 默认摄像头索引: {DEFAULT_WEBCAM}")
    print("⌨️  操作提示：")
    print("   - K键：校准眼睛位置")
    print("   - C键：校准头部姿态")
    print("   - R键：开始/暂停日志记录")
    print("   - Q键：退出程序")
    print("=" * 50)

# 初始化MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

# 初始化摄像头（增加重试和可用性检查）
cam_source = int(args.camSource)
cap = None
# 尝试多个摄像头索引（0,1,2），自动匹配可用摄像头
for cam_idx in [cam_source, 1, 2]:
    cap = cv.VideoCapture(cam_idx)
    if cap.isOpened():
        # 设置摄像头分辨率
        cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        # 设置摄像头缓冲区，减少延迟
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        if PRINT_DATA:
            print(f"✅ 成功打开摄像头，索引：{cam_idx}")
        break
    else:
        if cap is not None:
            cap.release()
        cap = None
        if PRINT_DATA:
            print(f"⚠️  摄像头索引 {cam_idx} 不可用，尝试下一个...")

# 若所有摄像头都不可用，直接退出
if cap is None or not cap.isOpened():
    if PRINT_DATA:
        print("❌ 未找到可用摄像头，请检查摄像头连接后重试！")
        print("=" * 50)
    exit(1)

# 初始化UDP Socket
iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 创建日志文件夹
csv_data = []
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)
    if PRINT_DATA:
        print(f"📁 日志文件夹已创建：{LOG_FOLDER}")

# 定义日志列名
column_names = [
    "Timestamp (ms)", "Left Eye Center X", "Left Eye Center Y",
    "Right Eye Center X", "Right Eye Center Y",
    "Left Iris Relative Pos Dx", "Left Iris Relative Pos Dy",
    "Right Iris Relative Pos Dx", "Right Iris Relative Pos Dy",
    "Total Blink Count",
]
if ENABLE_HEAD_POSE:
    column_names.extend(["Pitch", "Yaw", "Roll"])
if LOG_ALL_FEATURES:
    column_names.extend([f"Landmark_{i}_X" for i in range(468)] + [f"Landmark_{i}_Y" for i in range(468)])

# -----------------------------------------------------------------------------------------------------------------------------------
# 主循环
# -----------------------------------------------------------------------------------------------------------------------------------
try:
    angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)
    prev_total_blinks = 0
    prev_mouse_x, prev_mouse_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

    # 创建窗口并设置尺寸
    cv.namedWindow("Eye Tracking (Mouse Control)", cv.WINDOW_NORMAL)
    cv.resizeWindow("Eye Tracking (Mouse Control)", NEW_WINDOW_WIDTH, NEW_WINDOW_HEIGHT)
    cv.setWindowProperty("Eye Tracking (Mouse Control)", cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)

    while True:
        ret, frame = cap.read()
        # 增加帧读取重试机制（修正语法错误：移除无效的still关键字）
        if not ret:
            # 重新读取一次，避免临时帧丢失
            ret, frame = cap.read()
            if not ret:
                if PRINT_DATA:
                    print("⚠️  临时无法读取摄像头帧，正在重试...")
                time.sleep(0.01)
                continue

        # 放大画面
        frame = cv.resize(frame, (NEW_WINDOW_WIDTH, NEW_WINDOW_HEIGHT), interpolation=cv.INTER_LINEAR)
        img_h, img_w = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)
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
                mesh_points_3D[_indices_pose], [img_w, img_h, 1]
            )
            head_pose_points_2D = mesh_points[_indices_pose]

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

            # 头部朝向判断
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

            # 绘制头部朝向信息
            if SHOW_ON_SCREEN_DATA:
                cv.putText(
                    frame, f"Face: {face_looks}",
                    (img_w - 250, 80), cv.FONT_HERSHEY_TRIPLEX,
                    0.8, (0, 255, 0), 2, cv.LINE_AA
                )

            # 绘制鼻子朝向线
            nose_3d_projection, jacobian = cv.projectPoints(
                nose_3D_point, rot_vec, trans_vec, cam_matrix, dist_matrix
            )
            p1 = nose_2D_point
            p2 = (int(nose_2D_point[0] + angle_y * 10), int(nose_2D_point[1] - angle_x * 10))
            cv.line(frame, p1, p2, (255, 0, 255), 3)

            # 眨眼检测逻辑
            eyes_aspect_ratio = blinking_ratio(mesh_points_3D)
            if eyes_aspect_ratio <= BLINK_THRESHOLD:
                EYES_BLINK_FRAME_COUNTER += 1
                IS_BLINKING = True
                blink_recovery_counter = 0
            else:
                if EYES_BLINK_FRAME_COUNTER > EYE_AR_CONSEC_FRAMES:
                    TOTAL_BLINKS += 1
                EYES_BLINK_FRAME_COUNTER = 0
                if IS_BLINKING:
                    blink_recovery_counter += 1
                    if blink_recovery_counter >= BLINK_RECOVERY_FRAMES:
                        IS_BLINKING = False
                        blink_recovery_counter = 0

            # 绘制所有特征点（可选）
            if SHOW_ALL_FEATURES:
                for point in mesh_points:
                    cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)

            # 获取虹膜和眼角坐标
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            left_outer_corner = mesh_points[LEFT_EYE_OUTER_CORNER][0]
            left_inner_corner = mesh_points[LEFT_EYE_INNER_CORNER][0]
            right_outer_corner = mesh_points[RIGHT_EYE_OUTER_CORNER][0]
            right_inner_corner = mesh_points[RIGHT_EYE_INNER_CORNER][0]

            # 绘制虹膜和眼角标记
            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, left_inner_corner, 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, left_outer_corner, 3, (0, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, right_inner_corner, 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, right_outer_corner, 3, (0, 255, 255), -1, cv.LINE_AA)

            # 计算相对位置
            l_dx, l_dy = vector_position(left_outer_corner, center_left)
            r_dx, r_dy = vector_position(right_outer_corner, center_right)

            # 头部姿态估计
            if ENABLE_HEAD_POSE:
                pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
                angle_buffer.add([pitch, yaw, roll])
                pitch, yaw, roll = angle_buffer.get_average()

                if initial_pitch is None:
                    initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll

                pitch -= initial_pitch
                yaw -= initial_yaw
                roll -= initial_roll

            # 鼠标控制逻辑
            current_time = int(time.time() * 1000)
            if not IS_BLINKING and blink_recovery_counter == 0:
                target_x, target_y = map_pupil_direction_to_screen(
                    center_left, center_right,
                    left_outer_corner, left_inner_corner,
                    right_outer_corner, right_inner_corner
                )

                # 核心修正：限制鼠标坐标在屏幕范围内，避免触发PyAutoGUI fail-safe
                # 预留10像素边距，彻底远离屏幕角落
                safe_margin = 10
                target_x = max(safe_margin, min(SCREEN_WIDTH - safe_margin, target_x))
                target_y = max(safe_margin, min(SCREEN_HEIGHT - safe_margin, target_y))

                # 平滑移动鼠标
                smooth_x = int(prev_mouse_x * (1 - MOUSE_SMOOTH_FACTOR) + target_x * MOUSE_SMOOTH_FACTOR)
                smooth_y = int(prev_mouse_y * (1 - MOUSE_SMOOTH_FACTOR) + target_y * MOUSE_SMOOTH_FACTOR)

                # 再次限制平滑后的坐标，双重保险
                smooth_x = max(safe_margin, min(SCREEN_WIDTH - safe_margin, smooth_x))
                smooth_y = max(safe_margin, min(SCREEN_HEIGHT - safe_margin, smooth_y))

                pyautogui.moveTo(smooth_x, smooth_y, duration=0.05)
                prev_mouse_x, prev_mouse_y = smooth_x, smooth_y

            # 处理单双击
            if TOTAL_BLINKS > prev_total_blinks:
                handle_blink(current_time)
                prev_total_blinks = TOTAL_BLINKS

            # 日志记录与UDP传输
            if LOG_DATA and IS_RECORDING:
                log_entry = [
                    current_time, l_cx, l_cy, r_cx, r_cy,
                    l_dx, l_dy, r_dx, r_dy, TOTAL_BLINKS
                ]
                if ENABLE_HEAD_POSE:
                    log_entry.extend([pitch, yaw, roll])
                if LOG_ALL_FEATURES:
                    log_entry.extend([p for point in mesh_points for p in point])
                csv_data.append(log_entry)

            packet = np.array([current_time], dtype=np.int64).tobytes() + \
                     np.array([l_cx, l_cy, l_dx, l_dy], dtype=np.int32).tobytes()
            iris_socket.sendto(packet, SERVER_ADDRESS)

            # 绘制屏幕状态信息
            if SHOW_ON_SCREEN_DATA:
                # 录制状态标记
                if IS_RECORDING:
                    cv.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                # 眨眼次数
                cv.putText(frame, f"Blinks: {TOTAL_BLINKS}", (30, 80),
                           cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                # 头部姿态信息
                if ENABLE_HEAD_POSE:
                    cv.putText(frame, f"Pitch: {int(pitch)}", (30, 110),
                               cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    cv.putText(frame, f"Yaw: {int(yaw)}", (30, 140),
                               cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                # 范围状态
                cv.putText(frame, f"Range: Full Screen", (30, 170),
                           cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2, cv.LINE_AA)
                # 校准状态
                calib_status = "Done" if CALIBRATED else "Press K"
                calib_color = (0, 255, 0) if CALIBRATED else (0, 0, 255)
                cv.putText(frame, f"Calib: {calib_status}", (30, 200),
                           cv.FONT_HERSHEY_DUPLEX, 0.8, calib_color, 2, cv.LINE_AA)

        # 显示画面
        cv.imshow("Eye Tracking (Mouse Control)", frame)
        key = cv.waitKey(1) & 0xFF

        # 按键控制逻辑
        if key == ord('c') and ENABLE_HEAD_POSE:
            initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
            if PRINT_DATA:
                print("✅ 头部姿态已校准")
        if key == ord('k') and results.multi_face_landmarks:
            # 执行眼睛校准
            (l_cx, l_cy), _ = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
            (r_cx, r_cy), _ = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            calibrate_eye_position(
                mesh_points[LEFT_EYE_OUTER_CORNER][0],
                mesh_points[LEFT_EYE_INNER_CORNER][0],
                mesh_points[RIGHT_EYE_OUTER_CORNER][0],
                mesh_points[RIGHT_EYE_INNER_CORNER][0],
                center_left,
                center_right
            )
        if key == ord('r'):
            IS_RECORDING = not IS_RECORDING
            status = "开始" if IS_RECORDING else "暂停"
            if PRINT_DATA:
                print(f"📝 日志记录{status}")
        if key == ord('q'):
            if PRINT_DATA:
                print("=" * 50)
                print("📤 正在退出程序...")
            break

except Exception as e:
    print(f"❌ 程序运行错误: {e}")
finally:
    # 释放资源
    if cap is not None:
        cap.release()
    cv.destroyAllWindows()
    iris_socket.close()

    # 保存日志
    if LOG_DATA and IS_RECORDING and csv_data:
        if PRINT_DATA:
            print("📁 正在保存日志数据...")
        timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        csv_file_name = os.path.join(LOG_FOLDER, f"eye_tracking_log_{timestamp_str}.csv")
        with open(csv_file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(column_names)
            writer.writerows(csv_data)
        if PRINT_DATA:
            print(f"✅ 日志已保存至：{csv_file_name}")

    if PRINT_DATA:
        print("✅ 程序已正常退出")
        print("=" * 50)
