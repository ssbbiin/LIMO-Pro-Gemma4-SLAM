# LIMO Pro Gemma 4-based Intelligent Mobile Robot

**Gemma 4 E2B · YOLO11 · TensorRT · RGB-D Person Following · Dynamic Object-Aware SLAM**

LIMO Pro mobile robot project integrating multimodal AI, real-time person perception,
RGB-D based person following, and dynamic-object-aware RTAB-Map SLAM.

## 1. Project Overview

본 프로젝트에서는 LIMO Pro 모바일 로봇에 Gemma 4 E2B,
YOLO11, TensorRT 및 RGB-D Camera를 통합하여
사람 인식·추종과 동적 객체를 고려한 SLAM 시스템을 구현하였다.

실시간 사람 인식은 YOLO11을 이용하여 수행하고,
TensorRT FP16 기반으로 추론을 가속하여 모바일 로봇에서
실시간으로 동작할 수 있도록 구성하였다.

검출된 사람의 영상 내 위치와 RGB-D Camera의 Depth 정보를 결합하여
사람과의 거리 및 방향을 추정하고, 이를 ROS 2 `/cmd_vel` 명령으로 변환하여
LIMO Pro가 사람을 따라가는 Person Following 기능을 구현하였다.

또한 검출된 사람 영역의 Depth 정보를 제거한 Masked Depth Image를 생성하고,
이를 RTAB-Map의 입력으로 사용하여 이동하는 사람이 지도에 정적 장애물로
누적되는 영향을 줄이는 Dynamic Object-Aware SLAM 파이프라인을 구성하였다.

Gemma 4 E2B는 실시간 제어를 직접 수행하는 대신,
자연어 명령 해석 및 상황 판단과 같은 상위 수준의
로봇 의사결정 모듈로 활용하는 구조를 설계하였다.

## 2. Project Objectives

본 프로젝트의 주요 목표는 다음과 같다.

- Jetson 환경에서 Gemma 4 E2B 멀티모달 모델 구동
- ROS 2 RGB-D Camera와 AI perception pipeline 연동
- YOLO11 기반 실시간 사람 검출
- TensorRT FP16을 이용한 inference acceleration
- RGB-D Depth 기반 사람 거리 및 방향 추정
- `/cmd_vel` 기반 LIMO Pro Person Following 구현
- 사람 영역의 Depth 정보를 제거하는 Dynamic Object Masking 구현
- Masked Depth와 RTAB-Map을 연동한 Dynamic Object-Aware SLAM 구성

## 3. System Architecture

본 시스템은 실시간 인지 및 제어 계층과
상위 수준 AI 판단 계층으로 구성하였다.

### Real-Time Perception & Control

RGB-D Camera  
→ YOLO11 + TensorRT Person Detection  
→ Person Position / Depth Estimation  
→ Person Following Controller  
→ `/cmd_vel`  
→ LIMO Pro

### Dynamic Object-Aware SLAM

RGB-D Camera  
→ Person Detection  
→ Person Depth Masking  
→ `/camera/depth_masked/image_raw`  
→ RTAB-Map  
→ 3D Map / Occupancy Grid

### AI

RGB Camera / Robot State  
→ Gemma 4 E2B  
→ Natural Language Understanding / High-Level Decision  
→ Robot Behavior

실시간성이 요구되는 사람 검출 및 추종은 YOLO11과 TensorRT가 담당하고,
Gemma 4 E2B는 자연어 명령 해석과 상황 판단 등 상대적으로
낮은 주기의 상위 수준 의사결정을 담당하도록 역할을 분리하였다.

이를 통해 대규모 멀티모달 모델이 매 프레임의 저수준 제어를 직접 수행하지 않으면서도,
실시간 perception pipeline과 AI 판단을 하나의 모바일 로봇 시스템에
통합할 수 있도록 설계하였다.

## 4. Hardware & Software

| Category | Hardware / Software | Role |
|---|---|---|
| Mobile Robot | LIMO Pro | Mobile robot platform |
| Computing Platform | NVIDIA Jetson Orin | On-board AI inference and ROS 2 processing |
| RGB-D Camera | Orbbec RGB-D Camera | RGB image and depth acquisition |
| OS | Ubuntu 22.04 | Development environment |
| Middleware | ROS 2 Humble | Robot communication and system integration |
| Vision-Language Model | Gemma 4 E2B | Vision inference and high-level decision making |
| LLM Runtime | llama.cpp | Gemma 4 local inference |
| Object Detection | YOLO11n | Real-time person detection |
| Inference Acceleration | TensorRT FP16 | YOLO11 inference acceleration |
| SLAM | RTAB-Map | RGB-D based mapping |
| Navigation | Nav2 | Navigation framework |
| Computer Vision | OpenCV | Image processing |
| Programming | Python | ROS 2 nodes and perception pipeline implementation |

## 5. Gemma 4 E2B Integration

LIMO Pro의 Jetson 환경에서 멀티모달 모델을 직접 구동하기 위해
Gemma 4 E2B의 GGUF 모델과 llama.cpp 기반 로컬 inference 환경을 구성하였다.

llama.cpp를 CUDA 환경에서 빌드하고 Gemma 4 E2B Q4_0 모델과
multimodal projector를 적용하여 Text뿐만 아니라 Image 입력을 처리할 수 있도록 구성하였다.

실제 Camera Image를 Gemma 4에 입력하여 이미지 내 상황을 분석하는
Vision inference를 수행하였으며, 이를 통해 로봇의 시각 정보를
상위 수준의 상황 판단에 활용할 수 있는 환경을 구축하였다.

실시간 사람 검출 및 로봇 제어는 YOLO11과 TensorRT 기반 perception pipeline이 담당하고,
Gemma 4는 자연어 명령 해석 및 상황 판단과 같은 high-level intelligence를
담당하도록 역할을 분리하였다.

<p align="center">
  <img src="gemma4_vision_inference.png" width="850">
</p>

<p align="center">
  <em>Gemma 4 E2B multimodal vision inference on the LIMO Pro Jetson platform</em>
</p>

## 6. YOLO11 + TensorRT Person Detection

실시간 Person Following을 위해 YOLO11n을 이용하여
RGB Camera 영상에서 사람을 검출하였다.

초기 PyTorch 기반 YOLO11n에서는 실제 ROS 2 Camera 입력 처리 시
약 700–900 ms의 inference latency가 발생하여
실시간 로봇 제어에 한계가 있었다.

이를 개선하기 위해 YOLO11n 모델을 TensorRT FP16 Engine으로 변환하고,
Jetson GPU 기반 inference pipeline을 구성하였다.

### Inference Performance

| Method | Inference Latency | Processing Rate |
|---|---:|---:|
| PyTorch CPU | 700–900 ms | 1.1–1.4 FPS |
| TensorRT FP16 | 10–17 ms | 60–100 FPS |

TensorRT 적용을 통해 inference latency를 크게 감소시켰으며,
실제 RGB Camera 입력 속도에 맞춰 실시간 Person Detection이 가능하도록 개선하였다.

## 7. RGB-D Person Following

YOLO11을 통해 검출된 사람의 영상 내 위치와
RGB-D Camera의 Depth 정보를 결합하여 대상의 방향과 거리를 추정하였다.

사람의 Bounding Box 중심과 영상 중심 사이의 오차를 이용하여
좌우 방향을 판단하고, Depth 정보를 이용하여 로봇과 사람 사이의 거리를 계산하였다.

계산된 방향 및 거리 오차를 기반으로
ROS 2 `geometry_msgs/msg/Twist`의 `angular.z`와 `linear.x`를 생성하고,
`/cmd_vel`을 통해 LIMO Pro를 제어하였다.

### Control Pipeline

`Person Detection` → `RGB-D Distance Estimation` → `Direction & Distance Error` → `Velocity Command` → `/cmd_vel` → `LIMO Pro`

목표 거리는 **1.0 m**로 설정하였으며,
TensorRT 기반 실시간 Person Detection과 30 Hz 제어 주기를 적용하여
사람의 이동에 따라 LIMO Pro가 연속적으로 추종하도록 구성하였다.

### Person Following Demo

실제 실내 환경에서 사용자가 이동함에 따라 LIMO Pro가
대상의 방향과 거리를 추정하며 주행하는 모습을 확인하였다.

▶️ **Person Following Demo Video**

[person_following_demo.mp4](./person_following_demo.mp4)
