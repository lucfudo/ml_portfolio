# MACHINE LEARNING PROJECT - Anime Recommender System
This is a project that provides anime recommendations based on a user's watching history. It consists of several services:

- ml_pipeline: A service for dig into datasets and make assuptions.
- mlflow: A service for managing machine learning experiments and tracking model performance.
- prefect: A workflow management system for automating and scheduling data processing and training.
- train: A service for training the recommendation model using collaborative filtering.
- api: A FastAPI service for serving the recommendation model.
- anime_application: A React web application for allowing users to input their user ID and view anime recommendations.
- quality_checks: A service for performing data quality checks.

## Prerequisites
You need to have Docker on your machine. Follow the instructions to install it if you don't have them already.

Let's first make sure you have access to the Docker. If you are not sure, follow the instructions below.
### Docker
Check that you have *Docker desktop* installed in your machine by running:
```
docker -v
```
If that is not the case, just follow the official instructions:
- [Install Docker - Mac OS](https://docs.docker.com/desktop/install/mac-install/)
- [Install Docker - Linux](https://docs.docker.com/desktop/install/linux-install/)
- [Install Docker - Windows](https://docs.docker.com/desktop/install/windows-install/)

For those of you working on Windows, you might need to update Windows Subsystem for Linux. To do so, simply open PowerShell and run:
```
wsl --update
```
Once docker is installed, make sure that it is running correctly by running:
```
docker run -p 80:80 docker/getting-started
```

## Getting Started
Clone this repository to your local machine.
In the root directory of the project, run :
```
docker-compose build
docker-compose up
```

Once all services are up and running, navigate to http://localhost:3000 in your browser to access the anime application.

## Services
### **ml_pipeline**
This is where we create the first notebook to get information about the data and make assumptions to choose the best model with the best hyperparameters to build the most effective machine learning algorithm to recommend anime.
Password for jupyter is **pipeline**.

### **mlflow**
MLflow is an open source platform for managing machine learning lifecycle, including experiment tracking, reproducible runs, and model sharing.

The mlflow service in this project is responsible for tracking model performance, storing artifacts, and serving models. You can access the mlflow UI by navigating to http://localhost:5000 in your browser.

### **prefect**
Prefect is a workflow management system that allows you to define, schedule, and execute data workflows.

The prefect service in this project is responsible for automating and scheduling data processing and training. It uses the train service to train the recommendation model and logs the training results to the mlflow service.

### **train**
The train service in this project is responsible for training the recommendation model using collaborative filtering. It uses the data volume to access the necessary data files and the mlflow service to log the training results.

### **api**
The api service in this project is responsible for serving the recommendation model. It uses the data volume to access the necessary data files and the mlflow service to load the trained model. You can access the api endpoints by navigating to http://localhost:8001/docs in your browser.

### **anime_application**
The anime_application service in this project is a React web application that allows users to input their user ID and view anime recommendations. It uses the api service to fetch recommendations and display them to the user. You can access the anime application by navigating to http://localhost:3000 in your browser.

### **quality_checks**
The quality_checks service in this project is responsible for performing data quality checks. It uses the data volume to access the necessary data files and logs the results to the mlflow service.
Password for jupyter is **quality**.

## Conclusion
That's it! In this project, we train a recommendation model and serve it through a FastAPI service. We use a React web application to allow users to input their user ID and view anime recommendations.