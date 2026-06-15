# MemSlides Docker Image

## What Is MemSlides?

MemSlides is a memory-aware Web Studio for personalized slide generation and
multi-turn local revision. The Docker image packages the local Web Studio,
frontend assets, browser runtime, LibreOffice, Poppler, and PPTX/PDF export
runtime needed for local experimentation.

## Quick Start

Run the published image:

```bash
docker run --rm -p 7860:7860 \
  -v memslides-data:/app/.cache/memslides \
  huohua325/memslides:v2.0.1
```

Open `http://127.0.0.1:7860`, add a Service Profile in `Services`, validate it,
then generate or revise a deck.

## Docker Compose

From the GitHub repository:

```bash
docker compose up --build
```

This builds the local image from `docker/Dockerfile`, serves the Studio on port
`7860`, and stores local runtime state under `./.memslides`.

## Configure Services

MemSlides expects user-provided model and service credentials at runtime. The
recommended path is the Web Studio:

1. Open `Services`.
2. Add an OpenAI-compatible LLM base URL, model, and API key.
3. Add a PDF parser key or compatible endpoint if PDF parsing is needed.
4. Optionally enable embedding, Tavily search, vision, or image generation.
5. Click validate before running a demo.

For private YAML configuration, mount the file read-only instead of baking
credentials into the image:

```bash
docker compose -f docker-compose.yml -f docker-compose.private.yml up --build
```

## Persistent Local State

The container stores local sessions, Service Profiles, Memory Profiles, working
memory, and generated artifacts under:

```text
/app/.cache/memslides
```

Use a Docker volume or bind mount to persist that directory. Remove the volume
only when you intentionally want to clear local sessions and memory.

## Tags

- `latest`: current public image.
- `v2.0.1`: pinned public release image.

## Security And Privacy

- Do not put API keys, `.env` files, or private YAML files into the image.
- Pass secrets through the Web Studio Service Profile store, environment
  variables, or a read-only private compose override.
- Network acquisition is optional and depends on user-provided search or model
  credentials.
- Review generated slides, external URLs, and downloaded assets before
  presenting.

## Links

- GitHub: https://github.com/huohua325/Memslides
- Project page: https://memslides.github.io/
- Website: https://memslides.com/
