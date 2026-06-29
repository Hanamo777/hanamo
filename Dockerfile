# 1. PlayMCP 필수 규격 및 파이썬 환경
FROM --platform=linux/amd64 python:3.11-slim

# 2. 작업 폴더 설정
WORKDIR /app

# 3. 다운로드 도구 및 OpenCV/MediaPipe 구동용 필수 리눅스 라이브러리
RUN apt-get update && apt-get install -y wget libglib2.0-0 libgl1 libxcb1 libgles2 libegl1 && rm -rf /var/lib/apt/lists/*

# 4. 파이썬 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 핵심: 초경량 객체 인식 AI 모델 미리 다운로드
RUN wget -q -O efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite

# 6. 나머지 코드 복사
COPY . .

# 7. 클라우드 배포용 포트 환경변수 설정 및 노출 (PlayMCP 표준 대응)
ENV PORT=8080
EXPOSE 8080

CMD ["python", "server.py"]