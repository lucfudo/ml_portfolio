FROM python:3.9.16-slim

# Set working directory to /app
WORKDIR /app_home

COPY ./bin ./bin
COPY ./requirements.txt ./requirements.txt
RUN apt-get update && apt-get install -y gcc
RUN bash ./bin/install_requirements.sh

COPY . /app_home/web_service

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

WORKDIR /app_home/web_service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]