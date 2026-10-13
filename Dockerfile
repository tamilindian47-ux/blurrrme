FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgles2 libegl1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mediapipe imageio-ffmpeg pillow numpy runpod

RUN python -c "import cv2, mediapipe, imageio_ffmpeg; print('cv2', cv2.__version__, 'mediapipe', mediapipe.__version__)"

RUN mkdir -p /models && curl -fL -o /models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    && ls -la /models

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
