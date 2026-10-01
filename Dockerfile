FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY configs ./configs

# Model files are downloaded at first use and cached in a mounted volume.
VOLUME ["/app/models", "/app/outputs"]
ENTRYPOINT ["vehicle-tracking"]
CMD ["--help"]
