FROM python:3.10.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PYVISTA_OFF_SCREEN=true

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 xvfb libgl1-mesa-dev -y

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Welcome_Page.py", "--server.port=8501", "--server.address=0.0.0.0"]
