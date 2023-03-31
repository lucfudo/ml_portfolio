# Train Container 
This Docker container contains the necessary code to train a machine learning model using the Surprise library and MLflow. It is designed to work with the app_config.py file which contains the configuration settings for the MLflow tracking and registered model URIs.

## Getting started
To use this container, you will need to have Docker installed on your machine. Once Docker is installed, you can build the container by running the following command:
```
docker build -t train .
```
This will build a Docker image called train that contains the necessary libraries and code to train a machine learning model.
