FROM python:3.10.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["uvicorn", "app:app", "--reload", "--port", "8000", "--host", "0.0.0.0"]

