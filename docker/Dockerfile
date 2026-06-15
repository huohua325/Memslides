FROM node:20-bookworm-slim AS node-runtime

FROM node-runtime AS frontend-builder

WORKDIR /app

COPY frontend/package*.json ./frontend/

RUN cd frontend \
    && npm ci --no-audit --no-fund

COPY frontend ./frontend

RUN mkdir -p /app/src/memslides/web/static \
    && cd frontend \
    && npm run build

FROM python:3.11-slim-bookworm AS runtime

COPY --from=node-runtime /usr/local /usr/local

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/app/.cache \
    NODE_PATH=/app/node_modules \
    MEMSLIDES_PPTX_EXPORT_NODE_MODULES=/app/node_modules \
    MEMSLIDES_PPTX_EXPORT_AUTO_INSTALL=0 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    MEMSLIDES_PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    MEMSLIDES_PLAYWRIGHT_AUTO_INSTALL=0 \
    MEMSLIDES_CHROMIUM_EXECUTABLE=/usr/local/bin/memslides-chromium \
    PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/local/bin/memslides-chromium \
    MEMSLIDES_DEFAULT_CACHE_ROOT=/app/.cache/memslides \
    MEMSLIDES_WORKSPACE_BASE=/app/.cache/memslides/web \
    MEMSLIDES_DATA_ROOT=/app/.cache/memslides \
    MEMSLIDES_MEMORY_ROOT=/app/.cache/memslides/.memory \
    MEMSLIDES_DOCKER_SANDBOX=0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    fontconfig \
    fonts-dejavu \
    fonts-liberation \
    fonts-noto-cjk \
    git \
    imagemagick \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libsecret-1-0 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxshmfence1 \
    libxtst6 \
    libreoffice \
    poppler-utils \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY package.json pyproject.toml ./
COPY src ./src
COPY --from=frontend-builder /app/src/memslides/web/static ./src/memslides/web/static
COPY scripts ./scripts
COPY LICENSE THIRD_PARTY_NOTICES.md ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-build-isolation -e ".[web,research]" \
    && npm install --omit=dev --no-audit --no-fund \
    && python -m playwright install chromium ffmpeg \
    && chromium_path="$(find /ms-playwright -path '*/chrome-linux64/chrome' -type f | sort | tail -n 1)" \
    && test -n "${chromium_path}" \
    && ln -sf "${chromium_path}" /usr/local/bin/memslides-chromium \
    && chmod +x /usr/local/bin/memslides-chromium "${chromium_path}" \
    && python - <<'PY'
from pathlib import Path
required = [
    Path("/app/src/memslides/web/static/index.html"),
    Path("/app/node_modules/fast-glob"),
    Path("/app/node_modules/minimist"),
    Path("/app/node_modules/playwright"),
    Path("/app/node_modules/pptxgenjs"),
    Path("/app/node_modules/sharp"),
    Path("/usr/local/bin/memslides-chromium"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"Missing Docker runtime artifacts: {missing}")
PY

RUN mkdir -p /app/.cache/memslides \
    && chmod 700 /app/.cache/memslides

VOLUME ["/app/.cache/memslides"]

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import json, urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=3)); raise SystemExit(0 if data.get('status') == 'ok' else 1)"

CMD ["python", "-m", "memslides", "web", "--host", "0.0.0.0", "--port", "7860"]
