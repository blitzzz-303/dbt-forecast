# Use an official Python image as the base image
FROM python:3.8.15-slim-buster

RUN apt-get update && apt-get install -y gcc curl

# Create a working directory in the container
WORKDIR /sf

# Upgrade pip to the latest version
RUN pip install --upgrade pip
# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the dependencies in the requirements.txt file
RUN pip install -r requirements.txt --default-timeout=100

# Copy the rest of the project files to the container
COPY . ./

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "./streamlit/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]