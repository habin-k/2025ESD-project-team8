import cv2

# 입력과 출력 파일 경로
input_path = 'input_video.mp4'
output_path = 'demo_video.mp4'

# 입력 영상 열기
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"❌ Cannot open video: {input_path}")

# 원본 FPS, 해상도 정보
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count_static = fps * 10  # 10초 정지 영상 (예: 600프레임)
width, height = 320, 240

# 비디오 저장용 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 첫 프레임 읽기
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("❌ Failed to read the first frame.")

# 리사이즈
first_frame_resized = cv2.resize(first_frame, (width, height))

# 정지 영상 프레임 반복 쓰기
for _ in range(frame_count_static):
    out.write(first_frame_resized)

# 나머지 원본 영상 이어 붙이기
while True:
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (width, height))
    out.write(resized_frame)

# 자원 해제
cap.release()
out.release()

print(f"✅ 저장 완료: {output_path}")
