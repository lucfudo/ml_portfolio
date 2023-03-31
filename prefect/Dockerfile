FROM python:3.9-slim-buster

# Set working directory
WORKDIR /prefect

# Install prefect
RUN apt-get update -y && apt-get install -y tzdata python3 python3-pip
RUN pip3 install prefect

# Set environment variables
ENV API_KEY=pnu_42Hv56FdfkAcquozdSjoAdbtohJLvl1KUmqR
ENV LOCAL_DB='~/.prefect/orion.db'
ENV SWAGGER_URL=http://127.0.0.1:4200/docs
ENV REDOC_URL=http://127.0.0.1:4200/redoc

# Expose port
EXPOSE 4200

# Start prefect
# RUN prefect orion start
RUN prefect config set PREFECT_API_URL=http://localhost:4200/api 
CMD prefect server start --host 0.0.0.0
