# MemSlides Docker Image

## What Is MemSlides?

MemSlides is a memory-aware presentation generation framework for personalized
slide generation and multi-turn local revision. The Docker image provides a
reproducible experiment environment with Python, Node.js, Playwright,
LibreOffice, Poppler, fonts, and the PPTX/PDF export runtime.

## Quick Start

Run the published image:

```bash
docker run --rm \
  -v memslides-data:/app/.cache/memslides \
  huohua325/memslides:v2.0.1 \
  python -m memslides.experiment --help
```

Run the packaged smoke suite:

```bash
docker run --rm \
  -v memslides-data:/app/.cache/memslides \
  huohua325/memslides:v2.0.1 \
  python -m memslides.experiment run smoke_minimal \
    --output-base /app/.cache/memslides/experiments \
    --parallel 1
```

`smoke_minimal` is only a verification suite. You can pass any packaged suite
name or mounted suite YAML path to `python -m memslides.experiment run`.

## Docker Compose

From the GitHub repository:

```bash
docker compose build
docker compose run --rm memslides python -m memslides.experiment --help
docker compose run --rm memslides python -m memslides.experiment run smoke_minimal \
  --output-base /app/.cache/memslides/experiments \
  --parallel 1
```

Compose stores experiment state and outputs under `./.memslides`.

## Configuration

MemSlides expects user-provided model and service credentials for real
generation experiments. Keep credentials outside the image and provide them
through environment variables, `.env`, or a private YAML file.

For private YAML configuration, mount the file read-only:

```bash
docker compose -f docker-compose.yml -f docker-compose.private.yml run --rm memslides \
  python -m memslides.experiment run smoke_minimal \
  --output-base /app/.cache/memslides/experiments \
  --parallel 1
```

## Persistent Local State

The container stores generated artifacts, caches, memory state, and experiment
reports under:

```text
/app/.cache/memslides
```

Use a Docker volume or bind mount to persist that directory. Remove the volume
only when you intentionally want to clear local state.

## Tags

- `latest`: current public experiment image.
- `v2.0.1`: pinned public release image.

## Security And Privacy

- Do not put API keys, `.env` files, or private YAML files into the image.
- Pass secrets through environment variables, `.env`, or a read-only private
  compose override.
- Network acquisition is optional and depends on user-provided search or model
  credentials.
- Review generated slides, external URLs, and downloaded assets before
  presenting.

## Links

- GitHub: https://github.com/huohua325/Memslides
- Project page: https://memslides.github.io/
- Website: https://memslides.com/
