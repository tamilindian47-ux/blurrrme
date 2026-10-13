FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mediapipe imageio-ffmpeg pillow numpy runpod

RUN python -c "import cv2, mediapipe, imageio_ffmpeg; print('cv2', cv2.__version__, 'mediapipe', mediapipe.__version__)"

RUN python -c "import imageio_ffmpeg,subprocess;e=imageio_ffmpeg.get_ffmpeg_exe();o=subprocess.run([e,'-hide_banner','-encoders'],capture_output=True,text=True).stdout;assert 'libx264' in o and ' aac' in o,'ffmpeg missing encoders';print('ffmpeg ok')"

RUN mkdir -p /models && curl -fL -o /models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    && ls -la /models

RUN python -c "from mediapipe.tasks import python as p; from mediapipe.tasks.python import vision as v; v.FaceLandmarker.create_from_options(v.FaceLandmarkerOptions(base_options=p.BaseOptions(model_asset_path='/models/face_landmarker.task'))); print('landmarker ok')"

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
