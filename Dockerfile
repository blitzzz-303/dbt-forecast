# Use an official Python image as the base image
FROM python:3.8.15-slim-buster

# Create a working directory in the container
WORKDIR /dbt

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the dependencies in the requirements.txt file
RUN pip install -r requirements.txt --default-timeout=100

# Copy the rest of the project files to the container
COPY . ./

# Set the default command to be run when the container starts
# This can be overridden with the `--entrypoint` flag when starting the container
ENTRYPOINT ["/bin/sh", "-c"]