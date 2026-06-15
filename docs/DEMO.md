# Demo Guide

This guide is for the public local Studio. It does not use hosted accounts or private deployment settings.

## 1. Start The Studio

```bash
python -m memslides web --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`.

## 2. Configure Services

Open `Services`, add a Service Profile, and validate it.

Minimum useful setup:

- Main LLM base URL, model, and key.
- PDF parser key or compatible endpoint.

Optional but useful:

- Tavily search key for web evidence and web images.
- Embedding endpoint for richer memory retrieval.
- Image-generation endpoint for concept visuals.

## 3. Optional Memory Profile

Open `Memory` and create a profile such as:

```json
{
  "theme": {
    "primary": "#2563eb",
    "accent": "#f97316",
    "background": "#f8fafc",
    "surface": "#ffffff",
    "text": "#111827"
  },
  "content": {
    "preference": "Start with the conclusion, then show evidence and implications."
  }
}
```

Use `memory_intent` values such as `research_demo`, `policy_brief`, or `product_brief` to test whether preferences carry across related sessions.

## Scenario A: No Attachment, Resource-Assisted Generation

Prompt:

```text
Create a 4-slide briefing for city planners about urban heat islands and practical mitigation options.
Audience: municipal planning staff. Keep it decision-oriented and visually clear.
```

What to check:

- A PPTX appears in the artifact list.
- If Tavily is configured, `.history/web_sources.md` or asset manifests may appear in the workspace.
- At least one content page should use a visual asset when useful assets are available.
- The deck should not expose debug text, implementation notes, or raw tool traces.

Revision feedback:

```text
Make slide 2 more mechanism-focused, make slide 3 a compact evidence table, and keep the deck length unchanged.
```

Expected revision artifact:

- A fresh `modification_N.pptx`, not just the original export.

## Scenario B: PDF Or URL Input

Prompt:

```text
Create a 5-slide executive summary from this source. Focus on the decision, evidence, risks, and next steps.
```

Attach a PDF, paste a PDF URL, or provide a normal webpage URL if your configured services support it.

What to check:

- The source is copied or downloaded into the workspace.
- Converted document material appears in the workspace history.
- PDF figures or useful source images enter the asset manifest when available.
- The final PPTX is exported.

Revision feedback:

```text
Rewrite the results page as three evidence cards and add a short limitations note. Do not add or delete slides.
```

## Scenario C: Multi-Round Memory And Revision

First session prompt:

```text
Create a 4-slide research demo about memory-aware slide generation. Use conclusion-first structure and evidence cards.
```

Revision 1:

```text
Make slide 1 a decision brief and slide 3 more evidence-first. Keep the slide count unchanged.
```

Revision 2:

```text
Add one final take-home slide that summarizes what the audience should remember.
```

What to check:

- The first revision keeps the same slide count.
- The second revision creates exactly one extra slide.
- Working Memory shows preferences and round history.
- Each revision creates its own fresh PPTX/PDF export.

## Scriptable Local Smoke

After starting the Studio and configuring a ready Service Profile:

```bash
python scripts/local_web_smoke.py --base-url http://127.0.0.1:7860
```

The script:

1. Checks `/api/health`.
2. Finds or creates a Service Profile.
3. Creates a session.
4. Generates a 2-slide deck.
5. Confirms a PPTX export.
6. Applies one revision.
7. Confirms a fresh revision PPTX.
8. Writes a JSON report.

To let the script create a temporary local Service Profile:

```bash
export MEMSLIDES_SMOKE_LLM_API_KEY=...
export MEMSLIDES_SMOKE_LLM_MODEL=gpt-4.1
export MEMSLIDES_SMOKE_LLM_BASE_URL=https://api.openai.com/v1
export MEMSLIDES_SMOKE_PDF_API_KEY=...
```

Optional:

```bash
export MEMSLIDES_SMOKE_TAVILY_API_KEY=...
export MEMSLIDES_SMOKE_IMAGE_API_KEY=...
export MEMSLIDES_SMOKE_IMAGE_MODEL=...
```

## Expected Artifacts

Look in the Studio `Files` tab or the session workspace:

- Slide HTML files.
- `manuscript.pptx` or equivalent initial export.
- `modification_N.pptx` after revision.
- `.history/` reports and traces.
- `asset_manifest.json` when resource acquisition or visual planning runs.

## Demo Boundaries

- Do not paste private API keys into prompts.
- Do not rely on the hosted demo or private deployment scripts for this public local workflow.
- Review generated decks before presenting.
- If a revision produces warnings but still exports a fresh PPTX, inspect the deck and history report before treating it as clean.
