<h1 align="center">
  MemSlides: A Hierarchical Memory-Driven Agent Framework for Personalized Slide Generation with Multi-turn Local Revision
</h1>

<p align="center">
  <strong>Personalized presentation agents with user profile memory, working memory, tool memory, and scoped slide-local revision.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX"><strong>Paper</strong></a> |
  <a href="https://memslides.github.io/"><strong>Project Page</strong></a> |
  <a href="#demo-video"><strong>Demo Video</strong></a> |
  <a href="https://hub.docker.com/r/huohua325/memslides"><strong>Docker Hub</strong></a> |
  <a href="https://memslides.com/"><strong>Website</strong></a>
</p>

<p align="center">
  <a href="https://hub.docker.com/r/huohua325/memslides">
    <img alt="Docker image" src="https://img.shields.io/badge/docker-huohua325%2Fmemslides-2496ED?logo=docker&logoColor=white">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="Node" src="https://img.shields.io/badge/node-20-339933?logo=node.js&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green">
</p>

## Demo Video

https://github.com/user-attachments/assets/a92ab49e-bc5c-4e90-8c0a-0f23b08a8857

## Overview

MemSlides treats presentation generation as a stateful authoring process rather
than a one-shot source-to-slides conversion task. It separates personalization
signals by lifetime: persistent user profile memory captures recurring
cross-job preferences, working memory carries active session constraints across
revision rounds, and tool memory stores reusable execution experience for
reliable localized editing.

Long-term memory stores intent-conditioned user profile memory for round-0
personalization and tool memory for reusable execution experience. Working
memory maintains active preferences, session state, and revision constraints
within the current deck. During revision, MemSlides projects user feedback onto
the smallest affected slide region and applies scoped local patches instead of
repeatedly regenerating the full deck.

<p align="center">
  <img src="assets/figures/memslides_memory_workflow.png" width="92%" alt="MemSlides hierarchical memory and localized revision overview">
</p>

## Highlights

- **Intent-conditioned user profile memory** routes personalization by
  presentation intent, then applies preferences over theme, visual style,
  layout, template use, content strategy, and general presentation habits.
- **Multi-turn working memory** preserves temporary preferences, session
  constraints, and edit-state records across feedback turns in the same deck.
- **Tool memory** retrieves prior task and tool-chain experience before similar
  edit operations to reduce repeated execution failures.
- **Scoped slide-local revision** updates the smallest affected slide region
  instead of repeatedly rewriting the full deck.

## Evidence

<p align="center">
  <img src="assets/figures/user_profile_preference_memory_lifecycle.png" width="45%" alt="User profile memory lifecycle">
  <img src="assets/figures/tool_memory.png" width="45%" alt="Tool memory flow">
</p>

<p align="center">
  <img src="assets/figures/localized_modify_example.png" width="72%" alt="Localized modify example">
</p>

- User profile memory supports persona-aware round-0 personalization by routing
  intent-matched preferences into the current job.
- Working memory carries active session constraints and temporary preferences
  across multi-turn revision.
- Tool memory stores reusable execution experience so future localized edits
  can avoid repeated failures.
- Scoped local revision keeps the edit surface close to the requested element,
  reducing unintended drift in already aligned slide content.

## Quick Start

The fastest local path is Docker:

```bash
docker compose up --build
```

Open `http://127.0.0.1:7860`, add a Service Profile in `Services`, validate it,
then create or revise a deck. Local state is mounted at `./.memslides` and is
ignored by git.

For source installation:

```bash
sudo apt-get update
sudo apt-get install -y libreoffice fontconfig fonts-noto-cjk poppler-utils

conda env create -f environment.yml
conda activate memslides
pip install -e ".[web,research]"

cd frontend && npm install && npm run build
cd ..
python -m playwright install chromium ffmpeg
python -m memslides web --host 127.0.0.1 --port 7860
```

Node.js 20 is required for the frontend and PPTX export runtime. The conda
environment installs Node 20 from conda-forge. On Windows, install LibreOffice
and Poppler separately and make their command line tools available on `PATH`
before starting the Studio.

## Configuration

The recommended path is the Web Studio:

1. Open `Services`.
2. Add an OpenAI-compatible LLM base URL, model, and API key.
3. Add a PDF parser key or compatible endpoint if you want PDF parsing.
4. Optionally add embedding, Tavily web search, vision, or image-generation
   services.
5. Click validate before running a demo.

Service Profiles are encrypted locally under the data root. Do not commit
`.env`, `.memslides/`, generated workspaces, or private YAML files.

`.env` provides environment variables; YAML is the application config. Docker
Compose loads `.env` automatically, but a bare `python -m memslides ...`
process does not. For CLI runs, either export variables in your shell or pass a
YAML file with `--config`. The packaged public config is
`src/memslides/memslides.yaml`; its placeholders are expanded from the current
process environment when the YAML is loaded.

For Docker validation with a private YAML file, mount it read-only:

```bash
docker compose -f docker-compose.yml -f docker-compose.private.yml up --build
```

The override maps `./memslides.private.yaml` to
`/run/secrets/memslides.private.yaml` and sets
`MEMSLIDES_CONFIG_FILE=/run/secrets/memslides.private.yaml` inside the
container. Private YAML files are excluded from Docker build context and must
not be pushed to GitHub or Docker Hub.

## Experiments

The suite runner is useful for reproducible local experiments:

```bash
python -m memslides.experiment run smoke_minimal --output-base .memslides/experiments --parallel 1
```

Real generation suites need working model/service configuration. Use a ready
Web Studio Service Profile, a private YAML selected with `MEMSLIDES_CONFIG_FILE`
or `--config`, and optionally a private MCP manifest selected with
`MEMSLIDES_MCP_CONFIG_FILE`.

## Security And Privacy

- Keep API keys in the Web Studio Service Profile store, a local Docker `.env`,
  exported environment variables, or a private YAML file.
- Do not commit `.env`, `.memslides/`, generated workspaces, or private config
  files.
- Network acquisition is optional and depends on user-provided search or model
  credentials.
- External URLs and downloaded assets should be reviewed before presenting.

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
