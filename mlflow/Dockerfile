FROM ghcr.io/mlflow/mlflow:v2.1.1

# Set working directory
WORKDIR /mlflow

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Expose port
EXPOSE 5000

# Start MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0"]