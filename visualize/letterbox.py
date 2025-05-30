import cv2
import matplotlib.pyplot as plt

# 파일 경로 설정
video_path = './00151_H_A_BY_C1.mp4'

# 영상 로드
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

# 프레임이 정상적으로 로드된 경우
if ret:
    # 원본 프레임
    original_frame = frame.copy()

    # 640x640으로 resizing (Letterbox 방식)
    resized_frame = cv2.resize(original_frame, (640, 640), interpolation=cv2.INTER_LINEAR)

    # BGR -> RGB 변환 (OpenCV는 BGR, Matplotlib은 RGB 사용)
    original_frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # 결과 출력
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_frame_rgb)
    plt.title("Original Frame")

    plt.subplot(1, 2, 2)
    plt.imshow(resized_frame_rgb)
    plt.title("Resized (640x640) Frame")
    plt.show()
else:
    print("Error: Unable to read the video frame.")