FROM python:3.12-slim

RUN apt update && apt install -y --no-install-recommends \
    git

COPY pr-agent /app
WORKDIR /app

RUN pip install -e .
