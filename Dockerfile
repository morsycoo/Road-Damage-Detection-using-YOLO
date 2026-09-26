FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 \
       python3-pip \
       libxcb1 \
       libx11-6 \
       libxext6 \
       libxrender1 \
       libglib2.0-0 \
       libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --break-system-packages \
       torch==2.13.0 torchvision==0.28.0 \
       --index-url https://download.pytorch.org/whl/cu130

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir --break-system-packages \
       -r requirements.txt

COPY app ./app
COPY inference ./inference
COPY models ./models

RUN mkdir -p runtime/uploads outputs/image outputs/video

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
