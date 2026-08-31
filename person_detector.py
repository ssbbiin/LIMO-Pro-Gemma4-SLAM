#!/usr/bin/env python3

import time
import threading
import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ultralytics import YOLO


class PersonFollower(Node):

    def __init__(self):
        super().__init__("person_follower_trt")

        # ============================================================
        # 기본 설정
        # ============================================================

        self.RGB_TOPIC = "/camera/color/image_raw"
        self.DEPTH_TOPIC = "/camera/depth/image_raw"
        self.CMD_TOPIC = "/cmd_vel"

        self.MODEL_PATH = "/home/wego/llama.cpp/yolo11n.engine"

        # YOLO
        self.CONF = 0.30
        self.IMG_SIZE = 256

        # ------------------------------------------------------------
        # 추종 거리
        # ------------------------------------------------------------

        # 이 거리 근처에서 정지
        self.TARGET_DISTANCE = 1.00

        # 목표거리 ± 이 값에서는 전후진 안 함
        self.DIST_DEADBAND = 0.12

        # 너무 가까우면 무조건 전진 금지
        self.HARD_STOP_DISTANCE = 0.65

        # ------------------------------------------------------------
        # 선속도
        # ------------------------------------------------------------

        self.MAX_LINEAR = 0.96
        self.MIN_LINEAR = 0.18

        # 거리 오차 -> 속도
        self.K_LINEAR = 1.65

        # ------------------------------------------------------------
        # 회전 제어
        # ------------------------------------------------------------

        # 화면 중앙으로부터 이 정도는 무시
        # 640 px 기준 약 ±38 px
        self.CENTER_DEADBAND = 0.06

        # 이전보다 일부러 약하게 설정
        self.K_ANGULAR = 1.125

        self.MAX_ANGULAR = 0.975
        self.MIN_ANGULAR = 0.075

        # 사람이 너무 옆에 있으면 전진보다 방향부터 맞춤
        self.TURN_FIRST_ERROR = 0.28

        # ------------------------------------------------------------
        # 안전 / 시간
        # ------------------------------------------------------------

        # 마지막 detection이 이것보다 오래되면 즉시 정지
        self.TARGET_TIMEOUT = 0.20

        # 제어 루프 30 Hz
        self.CONTROL_HZ = 30.0

        # 속도를 갑자기 확 바꾸지 않기 위한 slew rate
        self.MAX_LINEAR_CHANGE = 0.035
        self.MAX_ANGULAR_CHANGE = 0.10

        # ============================================================
        # 상태 변수
        # ============================================================

        self.bridge = CvBridge()

        print("Loading TensorRT YOLO engine...")

        self.model = YOLO(
            self.MODEL_PATH,
            task="detect"
        )

        print("TensorRT model ready.")

        self.latest_rgb = None
        self.latest_depth = None

        self.frame_lock = threading.Lock()

        self.target_lock = threading.Lock()

        self.target_visible = False
        self.target_cx = None
        self.target_cy = None
        self.target_distance = None
        self.target_conf = None

        self.last_detection_time = 0.0

        self.current_linear = 0.0
        self.current_angular = 0.0

        self.running = True

        # ============================================================
        # ROS
        # ============================================================

        self.rgb_sub = self.create_subscription(
            Image,
            self.RGB_TOPIC,
            self.rgb_callback,
            1
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.DEPTH_TOPIC,
            self.depth_callback,
            1
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            self.CMD_TOPIC,
            10
        )

        self.control_timer = self.create_timer(
            1.0 / self.CONTROL_HZ,
            self.control_loop
        )

        # YOLO는 별도 thread에서 최신 프레임만 처리
        self.inference_thread = threading.Thread(
            target=self.inference_loop,
            daemon=True
        )

        self.inference_thread.start()

        print("")
        print("========================================")
        print(" TensorRT Person Follower START")
        print("========================================")
        print(f"Target distance : {self.TARGET_DISTANCE:.2f} m")
        print(f"Target timeout  : {self.TARGET_TIMEOUT:.2f} s")
        print(f"Control rate    : {self.CONTROL_HZ:.1f} Hz")
        print("========================================")


    # ================================================================
    # RGB callback
    # ================================================================

    def rgb_callback(self, msg):

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

            # 최신 프레임으로 그냥 교체
            with self.frame_lock:
                self.latest_rgb = frame

        except Exception as e:
            self.get_logger().error(
                f"RGB conversion error: {e}"
            )


    # ================================================================
    # Depth callback
    # ================================================================

    def depth_callback(self, msg):

        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough"
            )

            with self.frame_lock:
                self.latest_depth = depth

        except Exception as e:
            self.get_logger().error(
                f"Depth conversion error: {e}"
            )


    # ================================================================
    # Depth 계산
    # ================================================================

    def get_depth(self, depth, cx, cy):

        if depth is None:
            return None

        h, w = depth.shape[:2]

        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))

        # 중심 한 픽셀만 보면 depth hole 때문에 튈 수 있으므로
        # 작은 영역의 median 사용
        radius = 5

        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius + 1)

        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius + 1)

        region = depth[y1:y2, x1:x2]

        if region.size == 0:
            return None

        values = region.reshape(-1).astype(np.float32)

        # Orbbec depth가 16UC1이면 일반적으로 mm
        if depth.dtype == np.uint16:
            values = values[values > 0]

            if len(values) == 0:
                return None

            distance = float(np.median(values)) / 1000.0

        else:
            values = values[np.isfinite(values)]
            values = values[values > 0]

            if len(values) == 0:
                return None

            distance = float(np.median(values))

        # 비정상 값 제거
        if distance < 0.15 or distance > 8.0:
            return None

        return distance


    # ================================================================
    # YOLO inference thread
    # ================================================================

    def inference_loop(self):

        # 첫 inference 워밍업
        warmed_up = False

        while self.running and rclpy.ok():

            with self.frame_lock:

                if self.latest_rgb is None:
                    frame = None
                    depth = None
                else:
                    # inference 중 callback이 새 프레임으로 교체 가능
                    frame = self.latest_rgb
                    depth = self.latest_depth

            if frame is None:
                time.sleep(0.005)
                continue

            try:
                start = time.perf_counter()

                results = self.model(
                    frame,
                    classes=[0],
                    conf=self.CONF,
                    imgsz=self.IMG_SIZE,
                    verbose=False
                )

                inference_ms = (
                    time.perf_counter() - start
                ) * 1000.0

                r = results[0]

                if not warmed_up:
                    warmed_up = True
                    print(
                        f"TensorRT warmup complete "
                        f"({inference_ms:.1f} ms)"
                    )
                    continue

                boxes = r.boxes

                # ----------------------------------------------------
                # 사람이 없는 경우
                # ----------------------------------------------------

                if boxes is None or len(boxes) == 0:

                    # target_visible을 즉시 False로 만들되
                    # 실제 정지는 control_loop에서 timeout으로 처리
                    with self.target_lock:
                        self.target_visible = False

                    time.sleep(0.001)
                    continue

                # ----------------------------------------------------
                # 사람 선택
                #
                # 현재는 confidence가 가장 높은 사람 선택.
                # 이후 tracking ID를 붙일 수 있음.
                # ----------------------------------------------------

                best_box = None
                best_conf = -1.0

                for box in boxes:

                    conf = float(box.conf[0])

                    if conf > best_conf:
                        best_conf = conf
                        best_box = box

                if best_box is None:
                    continue

                x1, y1, x2, y2 = (
                    best_box.xyxy[0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                cx = int((x1 + x2) / 2.0)

                # bbox 중앙보다 약간 아래쪽 depth를 사용
                # 몸통/복부 쪽을 보도록 설정
                cy = int(
                    y1 + (y2 - y1) * 0.55
                )

                distance = self.get_depth(
                    depth,
                    cx,
                    cy
                )

                # depth가 없으면 이동 명령에 사용하지 않음
                if distance is None:

                    with self.target_lock:
                        self.target_visible = False

                    continue

                now = time.monotonic()

                with self.target_lock:

                    self.target_visible = True
                    self.target_cx = cx
                    self.target_cy = cy
                    self.target_distance = distance
                    self.target_conf = best_conf

                    self.last_detection_time = now

                # 너무 많이 출력하면 제어/터미널 모두 방해하므로
                # inference마다 print하지 않음

            except Exception as e:

                self.get_logger().error(
                    f"Inference error: {e}"
                )

                time.sleep(0.01)


    # ================================================================
    # slew limiter
    # ================================================================

    def approach(
        self,
        current,
        target,
        max_change
    ):

        diff = target - current

        if diff > max_change:
            diff = max_change

        elif diff < -max_change:
            diff = -max_change

        return current + diff


    # ================================================================
    # 제어
    # ================================================================

    def control_loop(self):

        now = time.monotonic()

        with self.target_lock:

            visible = self.target_visible
            cx = self.target_cx
            distance = self.target_distance
            conf = self.target_conf
            age = now - self.last_detection_time

        # ------------------------------------------------------------
        # Detection 오래됨 -> 즉시 STOP
        # ------------------------------------------------------------

        if (
            not visible
            or cx is None
            or distance is None
            or age > self.TARGET_TIMEOUT
        ):

            # 안전정지는 smoothing 안 함
            self.current_linear = 0.0
            self.current_angular = 0.0

            self.publish_cmd(
                0.0,
                0.0
            )

            return

        # 카메라 width
        with self.frame_lock:

            if self.latest_rgb is None:
                return

            width = self.latest_rgb.shape[1]

        center = width / 2.0

        # ------------------------------------------------------------
        # normalized horizontal error
        #
        # 왼쪽  -> +
        # 오른쪽 -> -
        # ------------------------------------------------------------

        horizontal_error = (
            center - cx
        ) / center

        # ============================================================
        # ANGULAR CONTROL
        # ============================================================

        if abs(horizontal_error) < self.CENTER_DEADBAND:

            target_angular = 0.0

        else:

            target_angular = (
                self.K_ANGULAR
                * horizontal_error
            )

            target_angular = float(
                np.clip(
                    target_angular,
                    -self.MAX_ANGULAR,
                    self.MAX_ANGULAR
                )
            )

            # 아주 작은 명령 때문에 덜덜거리는 것 방지
            if abs(target_angular) < self.MIN_ANGULAR:

                target_angular = (
                    math.copysign(
                        self.MIN_ANGULAR,
                        target_angular
                    )
                )

        # ============================================================
        # LINEAR CONTROL
        # ============================================================

        distance_error = (
            distance - self.TARGET_DISTANCE
        )

        # 너무 가까우면 무조건 전진 금지
        if distance <= self.HARD_STOP_DISTANCE:

            target_linear = 0.0

        elif abs(distance_error) <= self.DIST_DEADBAND:

            target_linear = 0.0

        elif distance_error > 0:

            target_linear = (
                self.K_LINEAR
                * distance_error
            )

            target_linear = float(
                np.clip(
                    target_linear,
                    self.MIN_LINEAR,
                    self.MAX_LINEAR
                )
            )

        else:

            # 사람이 너무 가까운 경우
            # 현재는 후진하지 않고 정지만 함.
            target_linear = 0.0

        # ------------------------------------------------------------
        # 사람이 화면 옆쪽이면 일단 방향부터 맞춤
        # ------------------------------------------------------------

        if abs(horizontal_error) > self.TURN_FIRST_ERROR:

            target_linear *= 0.20

        # ------------------------------------------------------------
        # 중앙에서 벗어날수록 전진속도 감소
        # ------------------------------------------------------------

        alignment_factor = max(
            0.0,
            1.0 - abs(horizontal_error) * 1.8
        )

        target_linear *= alignment_factor

        # ============================================================
        # Smooth command
        # ============================================================

        self.current_linear = self.approach(
            self.current_linear,
            target_linear,
            self.MAX_LINEAR_CHANGE
        )

        self.current_angular = self.approach(
            self.current_angular,
            target_angular,
            self.MAX_ANGULAR_CHANGE
        )

        # ------------------------------------------------------------
        # 목표거리 도달 시 linear smoothing도 즉시 제거
        # ------------------------------------------------------------

        if (
            distance <= self.TARGET_DISTANCE
            + self.DIST_DEADBAND
        ):

            self.current_linear = 0.0

        self.publish_cmd(
            self.current_linear,
            self.current_angular
        )


    # ================================================================
    # cmd_vel publish
    # ================================================================

    def publish_cmd(
        self,
        linear,
        angular
    ):

        msg = Twist()

        msg.linear.x = float(linear)
        msg.angular.z = float(angular)

        self.cmd_pub.publish(msg)


    # ================================================================
    # 종료
    # ================================================================

    def stop(self):

        self.running = False

        self.current_linear = 0.0
        self.current_angular = 0.0

        # 여러 번 0을 보내 확실히 정지
        for _ in range(5):

            self.publish_cmd(
                0.0,
                0.0
            )

            time.sleep(0.02)


def main():

    rclpy.init()

    node = PersonFollower()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        print("\nStopping person follower...")

    finally:

        node.stop()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
