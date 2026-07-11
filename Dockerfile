# 파이썬 3.10 슬림 버전 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성(OpenCV 구동용) 및 wget 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget libglib2.0-0 libgl1 libxcb1 libgles2 libegl1 \
    && rm -rf /var/lib/apt/lists/*

# 요구사항 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MediaPipe 객체 인식 경량 모델 미리 다운로드
RUN wget -q -O efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite

# 소스 코드 복사
COPY main.py .

# 서버 포트 노출
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]