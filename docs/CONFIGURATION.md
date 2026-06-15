# Configuration

MemSlides public mode is a local single-user Studio. It has no account system and stores Service Profiles under the local data root.

## Recommended Path: Web Studio

1. Start the Studio.
2. Open `Services`.
3. Create a Service Profile.
4. Validate it before running a demo.

Service Profile keys are encrypted locally. They are not committed by default and should not be copied into public issues, screenshots, or YAML files.

## Service Profile Fields

### Main LLM

Required.

- `Model`: the main chat model used by generation, planning, revision, and QA.
- `Base URL`: an OpenAI-compatible API base URL.
- `LLM key`: API key for the main LLM.

Use a capable model for the main LLM. MemSlides makes multi-step tool decisions, writes HTML slides, and revises existing decks; very small models may pass connectivity validation but still produce weak decks.

### PDF Parser

Required by the public Web Studio validation path.

- `PDF parser key`: key for an official PDF parser-style service.
- `PDF parser URL`: optional compatible endpoint.

If you do not plan to parse PDFs, you can still use prompt-only demos, but a ready Service Profile currently expects a PDF parser key or compatible endpoint.

### Embedding

Optional.

- Enables richer memory retrieval.
- Configure model, base URL, and key only if you want API-based embedding.
- Leave disabled for a simpler first run.

### Web Search

Optional.

- Provider: Tavily.
- Enables web evidence and image/resource acquisition when the prompt or workflow needs outside information.
- Leave disabled for offline/source-only demos.

### Vision Model

Optional in the Web Studio flow.

- The main LLM is assumed to be vision-capable by default when the profile says so.
- A separate vision endpoint can be configured through private YAML or future UI extensions if your deployment needs different routing.

### Image Generation

Optional.

- Enables generated concept images as local assets.
- Configure an OpenAI-compatible image generation base URL, model, and key.
- Image generation is meant for covers, concept visuals, and explanatory scenes. It should not replace factual evidence figures from PDFs or external sources.

## Runtime Variables

Start from:

```bash
cp .env.example .env
```

The `.env.example` file intentionally contains runtime-only values. It does not
list model endpoints or API keys.

`.env` and YAML do not compete with each other. `.env` provides environment
variables; YAML is the application config. Docker Compose loads `.env`
automatically, but a bare `python -m memslides ...` process does not. For CLI
runs, either export the variables in your shell or pass a YAML file with
`--config`. YAML placeholders are expanded from the current process environment
when the YAML is loaded.

Runtime variables:

```bash
MEMSLIDES_DATA_ROOT=.memslides
MEMSLIDES_WORKSPACE_BASE=.memslides/web
MEMSLIDES_DEFAULT_CACHE_ROOT=.memslides

MEMSLIDES_PLAYWRIGHT_AUTO_INSTALL=0
MEMSLIDES_PPTX_EXPORT_AUTO_INSTALL=0
MEMSLIDES_PPTX_EXPORT_VISUAL_MODE=auto
```

For private tool manifests, set `MEMSLIDES_MCP_CONFIG_FILE` to override the
YAML `mcp_config_file` value. This is useful for local experiments that need a
private MCP manifest while keeping the packaged `src/memslides/mcp.json` public.

Model endpoints and service keys should be configured through a private YAML
file, Web Studio Service Profiles, or shell/CI secrets. If you intentionally run
in env-only mode, use `src/memslides/memslides.yaml` as the source of truth for
supported placeholder names.

## CLI YAML

The packaged public config is `src/memslides/memslides.yaml`. A custom YAML
selected with `--config` becomes the active application config. If
`MEMSLIDES_CONFIG_FILE` is set, that file is used as the default config path for
processes that do not pass `--config`.

MCP manifest precedence is `MEMSLIDES_MCP_CONFIG_FILE`, then YAML
`mcp_config_file`, then packaged `src/memslides/mcp.json`.

```bash
python -m memslides --config src/memslides/memslides.yaml generate \
  --instruction "Create a 4-slide deck about memory-aware presentation generation" \
  --num-pages 4
```

For private local use, copy the packaged YAML outside the repository or name it
with a private suffix:

```bash
cp src/memslides/memslides.yaml memslides.private.yaml
```

Files matching `*.private.yaml` are ignored by git.

For Docker validation, mount a private YAML file read-only instead of copying it
into the image:

```bash
docker compose -f docker-compose.yml -f docker-compose.private.yml up --build
```

The override sets `MEMSLIDES_CONFIG_FILE=/run/secrets/memslides.private.yaml`.
Do not add private YAML files to the Dockerfile or Docker build context.

## Local Data Layout

Default local paths:

- `MEMSLIDES_DATA_ROOT=.memslides`
- `MEMSLIDES_WORKSPACE_BASE=.memslides/web`
- Service Profiles live under the data root.
- Memory stores live under the data root.
- Session workspaces live under the workspace base.

These paths are ignored by git.

## Resource Acquisition Controls

Network and image features depend on user configuration:

- Web evidence/image search needs a Tavily key.
- PDF parsing needs a PDF parser key or compatible endpoint.
- AI image generation needs an image-generation endpoint and key.
- Offline or source-only workflows can leave search and image generation disabled.

Downloaded assets and generated images are stored inside the session workspace and tracked in manifests such as `asset_manifest.json`, `web_asset_manifest.json`, and history reports when those flows run.

## Safe Local Practice

- Do not commit `.env`, `.memslides/`, generated workspaces, or private YAML files.
- Do not paste API keys into prompts.
- Review external sources and downloaded images before presenting.
- Treat generated decks as drafts that need human review.
