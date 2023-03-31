FROM jupyter/datascience-notebook

# Set working directory to /app
WORKDIR /app

COPY ./bin ./bin
COPY ./lib ./lib
COPY ./requirements.txt ./requirements.txt
RUN bash ./bin/install_requirements.sh
COPY . .

ENV PYTHONPATH "$PYTHONPATH:/app/lib"
ENV JUPYTER_ENABLE_LAB=yes
ENV JUPYTER_TOKEN=docker

VOLUME /app/data

# Expose port 8888 for jupyter lab
EXPOSE 8888
# Expose 8000 for data docs server
EXPOSE 8000 

# Run Jupyter Lab when container starts up
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token='quality'"]
