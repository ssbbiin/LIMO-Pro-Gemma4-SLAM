# LIMO Pro Gemma 4-based Dynamic Object-Aware SLAM

> Gemma 4 기반 동적 객체 인식 및 Depth Masking을 적용한  
> LIMO Pro RGB-D SLAM 프로젝트

## Project Overview

실내 환경에서 RGB-D SLAM을 수행할 때 사람과 같은 동적 객체가 카메라에 포함되면,
해당 객체의 Depth 정보가 지도 생성에 반영되어 실제 환경에는 존재하지 않는
장애물이나 흔적이 지도에 남을 수 있다.

본 프로젝트에서는 LIMO Pro 모바일 로봇의 RGB-D 카메라와 Gemma 4를 활용하여
사람을 동적 객체로 인식하고, 해당 영역의 Depth 정보를 제거한 뒤
RTAB-Map에 입력하는 Dynamic Object-Aware SLAM 파이프라인을 구성하였다.

또한 Gemma 4를 통해 검출한 사람의 위치와 RGB-D Camera의 Depth 정보를 활용하여
대상과의 거리 및 방향을 추정하고, 이를 기반으로 LIMO Pro가 사람을 따라 이동하는
Person Following 기능을 구현하였다.

이를 통해 동적 객체가 포함된 실내 환경에서 사람의 Depth 정보가
SLAM 및 Occupancy Grid 생성에 미치는 영향을 확인하고,
동적 객체 제거를 적용한 Mapping 환경을 구현하는 것을 목표로 하였다.

## System Pipeline

RGB-D Camera  
↓  
Gemma 4 Person Detection  
↓  
Person Mask Generation  
↓  
Depth Masking  
↓  
RTAB-Map RGB-D SLAM  
↓  
Dynamic Object-Filtered Map
