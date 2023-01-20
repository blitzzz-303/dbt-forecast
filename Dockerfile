FROM python:3.8.15-slim-buster
WORKDIR /dbt

# Install dependencies:
RUN pip install --upgrade pip
COPY requirements.txt .

RUN pip install -r requirements.txt --default-timeout=100 future
COPY . ./


ENTRYPOINT ["/bin/sh", "-c"]