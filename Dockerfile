# Dockerfile to serve the MNIST classifier model via REST
FROM python:3.9-slim
ENV DEBIAN_FRONTEND="noninteractive"
WORKDIR /app/src

RUN apt update -y && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/* && rm -rf /var/cache/apt/archives/*

# Install python dependencies
COPY ./requirements.txt /app
RUN python3 -m pip --no-cache-dir install --upgrade -r /app/requirements.txt

# Copy the necessary files into the docker image
COPY ./models /app/models/
COPY ./src/utils.py /app/src/
COPY ./src/server.py /app/src/
COPY ./configs/serve.yaml /app/configs/

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "6734"]
