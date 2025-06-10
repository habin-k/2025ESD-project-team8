# 2025ESD-project-team8
라즈베리파이4와 카메라를 활용해 고령자의 낙상 사고를 실시간으로 자동 감지하고, 낙상 발생 시 즉각적인 경고 및 외부 전송을 수행하는 경량 엣지 기반 감지 시스템 개발

---

## 고령자 낙상 감지 경량 엣지 시스템

### 프로젝트 개요
 프로젝트의 목표는 라즈베리파이4와 카메라를 활용하여 고령자의 낙상 사고를 실시간으로 자동 감지하고, 낙상 발생 시 즉각적인 경고 출력 및 외부 전송 기능을 제공하는 경량 엣지 기반 감지 시스템을 개발하는 것입니다.

### 주요 특징
- **YOLOv8-pose**를 이용하여 실시간으로 사람의 관절 위치(keypoint)를 추출
- 추출된 관절 정보를 **LSTM 모델**에 입력, 자세의 시간 변화 흐름을 분석하여 낙상 여부를 판단
- 낙상 감지 시 즉각적으로 경고음을 출력하고, 외부 시스템(예: 보호자, 의료진)으로 알림 전송
- 라즈베리파이4와 같이 리소스가 제한된 엣지 디바이스에서 동작하도록 경량화 및 최적화

### 기대 효과
- 고령자의 낙상 사고를 빠르게 인지 및 대응하여 2차 피해 예방
- 실시간 동작, 높은 정확도, 경량화로 실제 현장(가정, 요양시설 등) 적용 가능성 증대

### 기술 스택 및 환경
- 하드웨어: Raspberry Pi 4, 카메라 모듈
- 소프트웨어: Python, YOLOv8-pose, PyTorch, LSTM, 기타 필요 라이브러리
- 실시간 영상 스트리밍 및 경고/알림 시스템

---

### Weekly Schedule

| **Week** | **Task** | **Details** |
|----------|----------|--------------|
| **Week 8** | 데이터셋 확보 | AIHub 데이터셋 확보, 샘플 데이터 전처리 |
| **Week 9** | LSTM 학습 | LSTM 코드 작성, 샘플 데이터 학습 |
| **Week 10** | 데이터셋 전처리 | 전체 데이터셋에서 시계열 데이터 추출 |
| **Week 11** | LSTM 모델 최적화 | PR Curve 활용, 모델 성능 개선 |
| **Week 12** | 모델 통합 | YOLO-pose + LSTM 통합, 실시간 처리 파이프라인 구축 |
| **Week 13** | 라즈베리파이 이식 | ONNX 변환, 라즈베리파이 추론 테스트 |
| **Week 14** | 최적화 및 데모 준비 | 해상도 및 추론 주기(fps) 최적화, 사전 시연 테스트 |
| **Week 15** | 최종 발표 | 데모 준비, 최종 시스템 통합 |

---

### Changelog
- [25/04/22] Initialize repo
- [25/04/27] 1st progress report submit
- [25/05/07] 2차 발표
- [25/05/18] 2nd progress report submit
- [25/06/10] Final commit before Demo


### DATASET
- 한국지능정보사회진흥원 산하 AI Hub에서 제공하는 낙상사고 위험동작 영상-센서 쌍 데이터(https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71641) 활용


### 참고 논문 및 관련 연구




## LSTM Models (LSTM 모델)

### Description
This directory contains LSTM model definitions code and trained model checkpoints. It is specifically designed for the fall detection system, leveraging temporal sequence data for accurate fall detection.

주차별 개선되는 LSTM 모델 정의와 학습 코드를 포함하는 하위 폴더. 

### Contents
- **lstm_train.py**: LSTM model Training script | LSTM 모델 학습 스크립트
- **lstm_test.py**: test for LSTM trained model | LSTM 학습 모델 테스트 코드
- **precision.py**: draw P-R curve, and find the best threshold value

### 🔗 Related Data (관련 데이터)
- **Data_Extract/processed/**: Processed training and validation data