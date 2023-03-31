FROM python:3.9.16-slim

# Set working directory to /app
WORKDIR /app

COPY ./bin ./bin
COPY ./lib ./lib
COPY ./requirements.txt ./requirements.txt
RUN apt-get update && apt-get install -y gcc
RUN bash ./bin/install_requirements.sh
COPY . .

ENV PYTHONPATH "$PYTHONPATH:/app/lib"
ENV JUPYTER_ENABLE_LAB=yes
ENV JUPYTER_TOKEN=docker
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

VOLUME /app/data

# Expose port 8888 for jupyter lab
EXPOSE 8888

# Run Jupyter Lab when container starts up
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token='pipeline'"]
