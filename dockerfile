FROM python:3.12-slim

RUN apt update && apt install -y --no-install-recommends \
    git

WORKDIR /app

# 8218fa6e131feabbf26c58677dd8fe2d9c1f1138 is tag v0.26 https://github.com/Codium-ai/pr-agent/releases/tag/v0.26
RUN set -ex; \
    git init; \
    git remote add origin https://github.com/Codium-ai/pr-agent; \
    git fetch --depth 1 origin 8218fa6e131feabbf26c58677dd8fe2d9c1f1138; \
    git checkout FETCH_HEAD;

RUN pip install -e .

# ENTRYPOINT [ "python", "/app/pr_agent/cli.py" ]
ENTRYPOINT [ "/bin/bash" ]
