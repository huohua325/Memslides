# Installation And Experiment Usage

MemSlides public mode is experiment-oriented. The supported entry points are
the experiment CLI, the core Python CLI, and the Docker experiment environment.

## System Requirements

- Python 3.11.
- Node.js 20 for the PPTX export runtime.
- LibreOffice, Poppler, and CJK-capable fonts for slide export and document
  conversion.
- Playwright Chromium and ffmpeg.

On Linux:

```bash
sudo apt-get update
sudo apt-get install -y libreoffice fontconfig fonts-noto-cjk poppler-utils
```

On Windows, install LibreOffice and Poppler separately and make their command
line tools available on `PATH`.

## Source Setup

```bash
conda env create -f environment.yml
conda activate memslides
pip install -e ".[research]"
python -m playwright install chromium ffmpeg
```

For repository development, install the quality-tool extra as well:

```bash
pip install -e ".[dev,research]"
```

The PPTX export runtime installs its Node dependencies automatically into the
MemSlides cache on first export unless `MEMSLIDES_PPTX_EXPORT_AUTO_INSTALL=0`
is set. For offline environments, preinstall dependencies from the root
`package.json` or `src/memslides/presentation_export/package.json`, then set
`MEMSLIDES_PPTX_EXPORT_NODE_MODULES=/path/to/node_modules`.

## Configuration

Real generation experiments require model and service credentials. Keep
credentials outside git and provide them through environment variables, `.env`,
or a private YAML file selected with `MEMSLIDES_CONFIG_FILE` or `--config`.

The packaged public config is `src/memslides/memslides.yaml`; placeholders are
expanded from the current process environment when loaded.

## Run Experiments

```bash
python -m memslides.experiment --help
python -m memslides.experiment run smoke_minimal \
  --output-base .memslides/experiments \
  --parallel 1
```

`smoke_minimal` is a small verification suite. To run another suite, pass a
packaged suite name or a local YAML path:

```bash
python -m memslides.experiment run path/to/suite.yml \
  --output-base .memslides/experiments \
  --parallel 1
```

## Docker Environment

The Docker image is an experiment environment, not a long-running service.

```bash
docker compose build
docker compose run --rm memslides python -m memslides.experiment --help
docker compose run --rm memslides python -m memslides.experiment run smoke_minimal \
  --output-base /app/.cache/memslides/experiments \
  --parallel 1
```

For private YAML configuration:

```bash
docker compose -f docker-compose.yml -f docker-compose.private.yml run --rm memslides \
  python -m memslides.experiment run smoke_minimal \
  --output-base /app/.cache/memslides/experiments \
  --parallel 1
```

## Useful Checks

```bash
python -m compileall -q src tests
python scripts/check_source_hygiene.py
pytest -q
ruff check src tests
npm ci
npm run check:pptx-export
python -m memslides --help
python -m memslides.experiment --help
```

Generated outputs belong under `.memslides/` or another ignored directory and
must not be committed.

## Repository Layout

- `src/memslides/`: runtime code and public CLI/API entry points.
- `src/memslides/experiment/`: paper reproduction experiments and reports.
- `src/memslides/tools/`: MCP tool services and startup wrappers.
- `src/memslides/memory/`: profile, working-memory, retrieval, and storage code.
- `tests/`: automated tests.
- `.memslides/` and `~/.cache/memslides/`: ignored runtime state and caches.

The generated `.tools.json` snapshot and the unused internal modules
`memslides.memory.core.protocols` and `memslides.tools.ask_template_selection`
are no longer part of the package. Use the public CLI, API, and MCP
configuration instead.
