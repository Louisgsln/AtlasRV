FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs

RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN useradd --create-home --uid 10001 atlasrv \
    && mkdir -p /app/reports \
    && chown -R atlasrv:atlasrv /app

USER atlasrv

ENTRYPOINT ["atlas-rv"]
CMD ["research", "--provider", "synthetic", "--config", "configs/universe.yml", "--output", "reports/research"]
