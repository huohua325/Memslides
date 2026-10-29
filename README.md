<h1 align="center">
  MemSlides: A Hierarchical Memory-Driven Agent Framework for Personalized Slide Generation with Multi-turn Local Revision
</h1>

<p align="center">
  <strong>Personalized presentation agents with user profile memory, working memory, tool memory, and scoped slide-local revision.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.17162"><img alt="Paper" src="assets/badges/paper.svg"></a>
  <a href="https://memslides.github.io/"><img alt="Project Page" src="assets/badges/project-page.svg"></a>
  <a href="#demo-video"><img alt="Demo Video" src="assets/badges/demo-video.svg"></a>
  <a href="https://hub.docker.com/r/huohua325/memslides"><img alt="Docker Hub" src="assets/badges/docker-hub.svg"></a>
  <a href="https://memslides.com/"><img alt="Website" src="assets/badges/website.svg"></a>
</p>

> **Official website notice:** [memslides.com](https://memslides.com/) is the
> official MemSlides trial website. [memslides.org](https://memslides.org/) is
> a fake website, is not operated by this project, and is not affiliated with
> this repository.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="Node" src="https://img.shields.io/badge/node-20-339933?logo=node.js&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green">
</p>

<p align="center">
  &#11088; <strong>If MemSlides is useful for your research or slide-generation workflow, please consider starring this repository to help others discover it.</strong>
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

Install from source:

```bash
sudo apt-get update
sudo apt-get install -y libreoffice fontconfig fonts-noto-cjk poppler-utils

conda env create -f environment.yml
conda activate memslides
pip install -e ".[research]"

python -m playwright install chromium ffmpeg
python -m memslides.experiment --help
```

Run the built-in smoke suite:

```bash
python -m memslides.experiment run smoke_minimal \
  --output-base .memslides/experiments \
  --parallel 1
```

The same experiment can run inside the Docker environment:

```bash
docker compose build
docker compose run --rm memslides python -m memslides.experiment run smoke_minimal \
  --output-base /app/.cache/memslides/experiments \
  --parallel 1
```

`smoke_minimal` is only a small verification suite. Users can pass any local
suite YAML path or packaged suite name to `python -m memslides.experiment run`.

## Configuration

MemSlides needs user-provided model and service credentials for real generation
experiments. Keep credentials outside git and provide them through environment
variables, `.env`, or a private YAML file selected with `MEMSLIDES_CONFIG_FILE`
or `--config`.

The packaged public config is `src/memslides/memslides.yaml`; its placeholders
are expanded from the current process environment when the YAML is loaded.
Generated outputs, caches, private YAML files, and credentials must not be
committed.

## Repository Structure

```text
src/memslides/              Runtime package and public Python entry points
src/memslides/experiment/   Paper reproduction experiments and reports
src/memslides/tools/        MCP tool services and startup wrappers
src/memslides/memory/       Working-memory, profile, retrieval, and storage code
tests/                      Automated regression and contract tests
.memslides/                 Ignored runtime workspaces and generated outputs
~/.cache/memslides/         User-level caches, logs, and shared memory state
```

The public command-line entry points are `generate`, `revise`, and
`template induct`. The experiment runner remains available through
`python -m memslides.experiment`.

The generated tool snapshot and two unused internal modules were removed from
the source tree. Use the MCP configuration and public CLI instead of importing
`memslides/.tools.json`, `memslides.memory.core.protocols`, or
`memslides.tools.ask_template_selection`.

## Development Checks

Install the development and research extras, then run the same checks used by
continuous integration:

```bash
pip install -e ".[dev,research]"
python -m compileall -q src tests
python scripts/check_source_hygiene.py
pytest -q
ruff check src tests
npm ci
npm run check:pptx-export
```

For Docker runs with a private YAML file:

```bash
docker compose -f docker-compose.yml -f docker-compose.private.yml run --rm memslides \
  python -m memslides.experiment run smoke_minimal \
  --output-base /app/.cache/memslides/experiments \
  --parallel 1
```

The override maps `./memslides.private.yaml` to
`/run/secrets/memslides.private.yaml` and sets
`MEMSLIDES_CONFIG_FILE=/run/secrets/memslides.private.yaml` inside the
container.

## Experiment CLI

The suite runner is the main public entry point:

```bash
python -m memslides.experiment run smoke_minimal --output-base .memslides/experiments --parallel 1
python -m memslides.experiment report .memslides/experiments/smoke_minimal
python -m memslides.experiment personas
```

Core generation, revision, and template induction commands remain available for
scripted local use:

```bash
python -m memslides generate --instruction "Create a one-slide project summary" --num-pages 1
python -m memslides revise --workspace .memslides/session --feedback "Tighten the title"
python -m memslides template induct --template-file template.pptx
```

## Security And Privacy

- Keep API keys in environment variables, `.env`, or private YAML files.
- Do not commit `.env`, `.memslides/`, generated workspaces, or private config
  files.
- Network acquisition is optional and depends on user-provided search or model
  credentials.
- External URLs and downloaded assets should be reviewed before presenting.

## Citation

If you find MemSlides useful, please cite our paper.

```bibtex
@misc{jin2026memslides,
  title={MemSlides: A Hierarchical Memory Driven Agent Framework for Personalized Slide Generation with Multi-turn Local Revision},
  author={Ye Jin and Yangyang Xu and Jun Zhu and Yibo Yang},
  year={2026},
  eprint={2606.17162},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2606.17162},
  url={https://arxiv.org/abs/2606.17162},
}
```

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
