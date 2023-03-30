# Base image
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

COPY ./bin ./bin
COPY ./lib ./lib
COPY ./requirements.txt ./requirements.txt
RUN apt-get update && apt-get install -y gcc
RUN bash ./bin/install_requirements.sh
COPY . .

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

VOLUME /app/data

# Run the application
CMD ["python", "main.py"]
