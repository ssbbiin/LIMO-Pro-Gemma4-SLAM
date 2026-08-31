#!/usr/bin/env python3

import time
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO


class PersonDepthMask(Node):

    def __init__(self):
        super().__init__("person_depth_mask")

        self.RGB_TOPIC = "/camera/color/image_raw"
        self.DEPTH_TOPIC = "/camera/depth/image_raw"
        self.OUTPUT_TOPIC = "/camera/depth_masked/image_raw"

        self.MODEL_PATH = "/home/wego/llama.cpp/yolo11n.engine"

        self.CONF = 0.30
        self.IMG_SIZE = 256

        self.bridge = CvBridge()

        print("Loading TensorRT YOLO engine...")

        self.model = YOLO(
            self.MODEL_PATH,
            task="detect"
        )

        print("TensorRT model ready.")

        self.lock = threading.Lock()

        self.latest_rgb = None
        self.latest_depth = None
        self.latest_depth_header = None

        # 새로운 Depth가 들어왔는지 확인하기 위한 번호
        self.depth_seq = 0
        self.processed_depth_seq = -1

        self.running = True

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

        self.depth_pub = self.create_publisher(
            Image,
            self.OUTPUT_TOPIC,
            10
        )

        self.worker = threading.Thread(
            target=self.process_loop
        )

        self.worker.start()

        print("")
        print("===================================")
        print(" Person Depth Mask START")
        print("===================================")
        print(f"RGB    : {self.RGB_TOPIC}")
        print(f"Depth  : {self.DEPTH_TOPIC}")
        print(f"Output : {self.OUTPUT_TOPIC}")
        print("===================================")


    def rgb_callback(self, msg):

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

            with self.lock:
                self.latest_rgb = frame

        except Exception as e:
            self.get_logger().error(
                f"RGB error: {e}"
            )


    def depth_callback(self, msg):

        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough"
            )

            with self.lock:

                self.latest_depth = depth
                self.latest_depth_header = msg.header

                self.depth_seq += 1

        except Exception as e:
            self.get_logger().error(
                f"Depth error: {e}"
            )


    def process_loop(self):

        warmed_up = False

        while self.running and rclpy.ok():

            with self.lock:

                # 아직 입력 준비 안 됨
                if (
                    self.latest_rgb is None
                    or self.latest_depth is None
                    or self.latest_depth_header is None
                ):
                    rgb = None
                    depth = None
                    header = None
                    seq = None

                # 이미 처리한 Depth 프레임
                elif self.depth_seq == self.processed_depth_seq:

                    rgb = None
                    depth = None
                    header = None
                    seq = None

                else:

                    rgb = self.latest_rgb.copy()
                    depth = self.latest_depth.copy()
                    header = self.latest_depth_header
                    seq = self.depth_seq

                    # 이 Depth는 이제 처리 중
                    self.processed_depth_seq = seq

            if rgb is None:
                time.sleep(0.001)
                continue

            try:

                results = self.model(
                    rgb,
                    classes=[0],
                    conf=self.CONF,
                    imgsz=self.IMG_SIZE,
                    verbose=False
                )

                if not warmed_up:

                    warmed_up = True
                    print("TensorRT warmup complete")

                masked = depth.copy()

                boxes = results[0].boxes

                person_count = 0

                if boxes is not None:

                    h, w = masked.shape[:2]

                    for box in boxes:

                        x1, y1, x2, y2 = (
                            box.xyxy[0]
                            .detach()
                            .cpu()
                            .tolist()
                        )

                        # 사람 가장자리까지 조금 넓게 제거
                        box_w = x2 - x1
                        box_h = y2 - y1

                        pad_x = int(box_w * 0.05)
                        pad_y = int(box_h * 0.03)

                        x1 = max(
                            0,
                            int(x1) - pad_x
                        )

                        y1 = max(
                            0,
                            int(y1) - pad_y
                        )

                        x2 = min(
                            w,
                            int(x2) + pad_x
                        )

                        y2 = min(
                            h,
                            int(y2) + pad_y
                        )

                        # 사람 영역의 Depth 제거
                        masked[
                            y1:y2,
                            x1:x2
                        ] = 0

                        person_count += 1

                out = self.bridge.cv2_to_imgmsg(
                    masked,
                    encoding="passthrough"
                )

                # 원본 Depth timestamp / frame_id 그대로 사용
                out.header = header

                self.depth_pub.publish(out)

                print(
                    f"\rPERSON MASKED: {person_count}",
                    end="",
                    flush=True
                )

            except Exception as e:

                self.get_logger().error(
                    f"Mask error: {e}"
                )

                time.sleep(0.005)


    def stop(self):

        print("\nStopping mask node...")

        self.running = False

        if (
            self.worker is not None
            and self.worker.is_alive()
        ):

            self.worker.join(
                timeout=3.0
            )


def main():

    rclpy.init()

    node = PersonDepthMask()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.stop()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
