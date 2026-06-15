# Installation

MemSlides public mode is a local, single-user Web Studio plus CLI. The two supported setup paths are conda and Docker.

## Requirements

- Docker works on Linux, macOS, and Windows with Docker Desktop.
- Source installs support Linux, macOS, and Windows conda environments.
- Python 3.11.
- Node.js 20 for the frontend and PPTX export runtime.
- LibreOffice and Poppler for export/conversion utilities.
- A valid local Service Profile for generation.

On Linux, install system packages:

```bash
sudo apt-get update
sudo apt-get install -y libreoffice fontconfig fonts-noto-cjk poppler-utils
```

The CJK font package is strongly recommended even for English demos because generated decks may include multilingual source names or citations.
On Windows, install LibreOffice and Poppler separately and ensure `soffice`,
`pdfinfo`, and related Poppler tools are available on `PATH`.

## Conda Setup

From the repository root:

```bash
conda env create -f environment.yml
conda activate memslides
pip install -e ".[web,research]"
```

Build the frontend:

```bash
cd frontend
npm install
npm run build
cd ..
```

The PPTX export runtime installs its Node dependencies automatically into the MemSlides cache on first export. For offline environments, preinstall the runtime dependencies from `src/memslides/presentation_export/package.json` and set `MEMSLIDES_PPTX_EXPORT_NODE_MODULES=/path/to/node_modules`.

Install Playwright browser binaries:

```bash
python -m playwright install chromium ffmpeg
```

Run the command in the same Python environment that runs MemSlides. If export later reports that a `chromium_headless_shell-*` executable does not exist, the Playwright package and browser cache are out of sync; rerun the install command for that environment, clear the stale `PLAYWRIGHT_BROWSERS_PATH` cache, or set `MEMSLIDES_CHROMIUM_EXECUTABLE` to a compatible Chromium executable.

Run the local Studio:

```bash
python -m memslides web --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`.

## Docker Setup

Start the public local container:

```bash
docker compose up --build
```

The `.env` file is optional. Create it only when you want Docker Compose to pass
local environment overrides into the container:

```bash
cp .env.example .env
```

Open `http://127.0.0.1:7860`.

Runtime state is mounted at `./.memslides`. Delete that directory only if you want to remove local sessions, local Service Profiles, and local memory.

To validate with a private YAML file without baking credentials into the image:

```bash
docker compose -f docker-compose.yml -f docker-compose.private.yml up --build
```

This mounts `./memslides.private.yaml` read-only at
`/run/secrets/memslides.private.yaml` and sets
`MEMSLIDES_CONFIG_FILE=/run/secrets/memslides.private.yaml`. That YAML becomes
the active application config; any `${...}` placeholders inside it are expanded
from the container environment.

To build and publish the Docker Hub image:

```bash
docker build -f docker/Dockerfile -t memslides:local .
docker tag memslides:local huohua325/memslides:v2.0.1
docker tag memslides:local huohua325/memslides:latest
docker push huohua325/memslides:v2.0.1
docker push huohua325/memslides:latest
```

The Docker image must not contain `.env` or `*.private.yaml`; pass secrets at
runtime through the Web Studio Service Profile store, environment variables, or
the read-only private compose override.

## Configure A Service Profile

After starting the Studio:

1. Open `Services`.
2. Add a main LLM base URL, model, and API key.
3. Add a PDF parser key or compatible endpoint if PDF parsing is needed.
4. Optionally enable embedding, Tavily search, vision, and image generation.
5. Click validate.

The smoke script expects at least one ready Service Profile unless you provide smoke environment variables.

## Health Check

```bash
curl http://127.0.0.1:7860/api/health
```

Expected:

```json
{"status":"ok","mode":"single-user-local"}
```

The response also includes the local workspace path.

## Minimal Runtime Checks

Check Node-side PPTX export dependencies from a source checkout:

```bash
npm run check:pptx-export
```

Check the experiment suite CLI:

```bash
python -m memslides.experiment --help
python -m memslides.experiment run smoke_minimal --output-base .memslides/experiments --parallel 1
```

The built-in smoke suite uses the real MemSlides runtime. It requires working
model/service configuration before generation can complete.

Run a Web Studio smoke after configuring a Service Profile:

```bash
python scripts/local_web_smoke.py --base-url http://127.0.0.1:7860
```

The smoke performs health, generation, revision, and fresh PPTX checks. It uses real model calls, so it requires working credentials.

## Common Issues

- `Web static assets are not installed`: run `cd frontend && npm install && npm run build`.
- `pptx_export Node runtime dependencies are missing`: keep `MEMSLIDES_PPTX_EXPORT_AUTO_INSTALL=1` for first-run automatic install, or set `MEMSLIDES_PPTX_EXPORT_NODE_MODULES` to a preinstalled runtime directory.
- Browser/export failures: run `python -m playwright install chromium ffmpeg` in the active MemSlides environment; after upgrading Playwright, refresh the browser cache or set `MEMSLIDES_CHROMIUM_EXECUTABLE`.
- Missing CJK text in output: install `fonts-noto-cjk` and restart the Studio.
- Service Profile validation fails: verify the base URL, model name, API key, and PDF parser configuration.
