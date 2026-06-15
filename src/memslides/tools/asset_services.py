import asyncio
import base64
import os
import sys
from pathlib import Path

# Ensure direct script execution imports the local repo package first.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import httpx
from fake_useragent import UserAgent
from fastmcp import FastMCP
from PIL import Image

from memslides.tools.structured_visuals import (
    render_chart_asset_impl,
    render_flowchart_asset_impl,
    render_table_asset_impl,
)
from memslides.utils.config import MemSlidesConfig
from memslides.utils.log import debug, info, set_logger, warning

FAKE_UA = UserAgent()
IMAGE_DOWNLOAD_RETRIES = 3
IMAGE_DOWNLOAD_TIMEOUT = 60.0

mcp = FastMCP(name="MemSlidesAssetTools")

LLM_CONFIG = MemSlidesConfig.load_from_file(os.getenv("MEMSLIDES_CONFIG_FILE"))


if LLM_CONFIG.t2i_model is not None:

    @mcp.tool()
    async def image_generation(prompt: str, height: int, path: str, width: int = 0) -> str:
        """
        Generate an image and save it to the specified path.

        Args:
            prompt: Text description of the image to generate. Should be detailed and specific, but do not include aspect ratio.
            height: Height of the image, in pixels
            path: Full path where the image should be saved
            width: Width of the image, in pixels. If omitted, auto-calculated from height using 16:9 ratio.
        """
        if width <= 0:
            width = round(height * 16 / 9)

        response = await LLM_CONFIG.t2i_model.generate_image(
            prompt=prompt, width=width, height=height
        )

        # Validate response structure
        if not hasattr(response, 'data') or not response.data:
            raise ValueError(f"Invalid response from image generation API: {response}")
        
        image_b64 = response.data[0].b64_json
        image_url = response.data[0].url

        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if image_b64:
            # Decode base64 image data
            image_bytes = base64.b64decode(image_b64)
        elif image_url:
            # Download image with retry and timeout
            image_bytes = None
            last_error = None
            for attempt in range(IMAGE_DOWNLOAD_RETRIES):
                try:
                    if attempt > 0:
                        await asyncio.sleep(attempt)  # Exponential backoff
                    async with httpx.AsyncClient(
                        headers={"User-Agent": FAKE_UA.random},
                        follow_redirects=True,
                        timeout=IMAGE_DOWNLOAD_TIMEOUT,
                    ) as client:
                        resp = await client.get(image_url)
                        resp.raise_for_status()
                        image_bytes = resp.content
                        break
                except Exception as e:
                    last_error = str(e)
                    warning(f"Image download attempt {attempt + 1} failed: {e}")
            
            if image_bytes is None:
                raise RuntimeError(f"Failed to download generated image after {IMAGE_DOWNLOAD_RETRIES} attempts: {last_error}")
        else:
            raise ValueError("Empty Response: no base64 data or URL returned")

        # Save image to specified path
        with open(path, "wb") as file:
            file.write(image_bytes)

        info(
            f"Image generated: prompt='{prompt}', size=({width}x{height}), saved to '{path}'"
        )
        return "Image generated successfully, saved to " + path


_CAPTION_CACHE: dict[str, dict] = {}

_CAPTION_SYSTEM = """
You are a helpful assistant that can describe the main content of the image in less than 50 words, avoiding unnecessary details or comments.
Additionally, classify the image as 'Table', 'Chart', 'Landscape', 'Diagram', 'Banner', 'Background', 'Icon', 'Logo', etc. or 'Picture' if it cannot be classified as one of the above.
Give your answer in the following format:
<type>:<description>
Example Output:
Chart: Bar graph showing quarterly revenue growth over five years. Color-coded bars represent different product lines. Notable spike in Q4 of the most recent year, with a dotted line indicating industry average for comparison
Now give your answer in one sentence only, without line breaks:
"""


@mcp.tool()
async def image_caption(image_path: str) -> dict:
    """
    Generate a caption for the image, including its type and a brief description.

    Args:
        image_path: The path to the image to caption.

    Returns:
        The caption and size for the image
    """
    if not Path(image_path).exists():
        return {"error": f"Image path {image_path} does not exist"}
    if image_path in _CAPTION_CACHE:
        info(f"Image caption cache hit: path='{image_path}'")
        return _CAPTION_CACHE[image_path]
    with open(image_path, "rb") as f:
        image_b64 = (
            f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
        )
    response = await LLM_CONFIG.vision_model.run(
        messages=[
            {"role": "system", "content": _CAPTION_SYSTEM},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_b64}}],
            },
        ],
    )

    info(
        f"Image captioned: path='{image_path}', caption='{response.choices[0].message.content}'"
    )
    with Image.open(image_path) as img:
        size = img.size
    result = {
        "size": size,
        "caption": response.choices[0].message.content,
    }
    _CAPTION_CACHE[image_path] = result
    return result


_SUMMARY_SYSTEM = """
You are a professional document analyst that generates reports based on specific tasks

Instructions:
1. Thoroughly analyze the provided document and extract key information relevant to the specified task.
2. Create a comprehensive yet concise summary report, prioritizing presenting key methodologies, critical findings, and relevant data points to support an in-depth understanding.
3. Use clear Markdown formatting with logical headers and structure.

Important: Only respond with content directly related to the task and document analysis. Do not add external information, or offer any additional advice and help.
"""


@mcp.tool()
def render_chart_asset(
    chart_type: str,
    x_field: str,
    y_fields: list[str],
    rows: list[dict[str, object]] | None = None,
    csv_text: str = "",
    csv_path: str = "",
    series_field: str = "",
    title: str = "",
    subtitle: str = "",
    x_label: str = "",
    y_label: str = "",
    note: str = "",
    width: int = 960,
    height: int = 540,
    output_format: str = "svg",
    output_stem: str = "",
    style_overrides: dict[str, object] | None = None,
) -> dict:
    """
    Render a deterministic chart asset from structured data.

    Args:
        chart_type: One of line, bar, grouped_bar, stacked_bar, area, scatter, pie, donut.
        x_field: Field name for the x/category dimension.
        y_fields: One or more value field names.
        rows: Structured row objects. Provide exactly one of rows, csv_text, or csv_path.
        csv_text: CSV content with a header row.
        csv_path: Workspace-local CSV path.
        series_field: Optional grouping field when data is already in long form.
        title: Visible chart title.
        subtitle: Visible chart subtitle.
        x_label: Axis label override.
        y_label: Axis label override.
        note: Extra note appended under the title area.
        width: Chart width in pixels.
        height: Chart height in pixels.
        output_format: svg, png, or both.
        output_stem: Optional filename stem prefix.
        style_overrides: Optional design tokens / chart style overrides.
    """
    result = render_chart_asset_impl(
        chart_type=chart_type,
        rows=rows,
        csv_text=csv_text,
        csv_path=csv_path,
        x_field=x_field,
        y_fields=y_fields,
        series_field=series_field,
        title=title,
        subtitle=subtitle,
        x_label=x_label,
        y_label=y_label,
        note=note,
        width=width,
        height=height,
        output_format=output_format,
        output_stem=output_stem,
        style_overrides=style_overrides,
        workspace=Path.cwd(),
        config=LLM_CONFIG,
    )
    info(
        "Structured chart asset rendered: type='%s', title='%s', outputs=%s",
        chart_type,
        title or chart_type,
        result.get("rendered_paths", {}),
    )
    return result


@mcp.tool()
def render_table_asset(
    rows: list[dict[str, object]] | None = None,
    columns: list[str] | None = None,
    csv_text: str = "",
    csv_path: str = "",
    markdown_table: str = "",
    style: str = "three_line",
    caption: str = "",
    title: str = "",
    footnote: str = "",
    output_mode: str = "html",
    output_format: str = "",
    width: int = 960,
    height: int = 540,
    output_stem: str = "",
    chart_type: str = "",
    x_field: str = "",
    y_fields: list[str] | str | None = None,
    series_field: str = "",
    subtitle: str = "",
    x_label: str = "",
    y_label: str = "",
    note: str = "",
    style_overrides: dict[str, object] | None = None,
) -> dict:
    """
    Render a deterministic table asset from structured data.

    Args:
        rows: Structured row objects. Provide exactly one of rows, csv_text, csv_path, or markdown_table.
        columns: Optional explicit column order for rows input.
        csv_text: CSV content with a header row.
        csv_path: Workspace-local CSV path.
        markdown_table: Markdown table source.
        style: One of three_line, simple_grid, or minimal.
        caption: Visible table caption.
        title: Alias for caption; useful when the model uses chart-like naming.
        footnote: Optional note shown below the table.
        output_mode: html, svg, png, or both.
        output_format: Alias for output_mode; also accepted for chart compatibility.
        width: Table width in pixels.
        height: Optional chart height when chart arguments are accidentally sent here.
        output_stem: Optional filename stem prefix.
        chart_type: If provided, route this mistaken table call to render_chart_asset.
        x_field: Chart x/category field when routing mistaken chart arguments.
        y_fields: Chart y/value fields when routing mistaken chart arguments.
        series_field: Optional chart grouping field when routing.
        subtitle: Optional chart subtitle when routing.
        x_label: Optional chart x-axis label when routing.
        y_label: Optional chart y-axis label when routing.
        note: Optional chart note when routing.
        style_overrides: Optional design tokens / table style overrides.
    """
    chart_like = bool(chart_type or x_field or y_fields)
    if chart_like:
        chart_output_format = (output_format or "").strip().lower()
        if chart_output_format not in {"svg", "png", "both"}:
            chart_output_format = "svg"
        normalized_y_fields: list[str]
        if isinstance(y_fields, str):
            normalized_y_fields = [part.strip() for part in y_fields.split(",") if part.strip()]
        elif isinstance(y_fields, list):
            normalized_y_fields = [str(part).strip() for part in y_fields if str(part).strip()]
        else:
            normalized_y_fields = []
        inferred_columns = list(columns or [])
        if not inferred_columns and rows:
            inferred_columns = [str(key) for key in rows[0].keys()]
        inferred_x = x_field or (inferred_columns[0] if inferred_columns else "")
        if not normalized_y_fields and len(inferred_columns) > 1:
            normalized_y_fields = inferred_columns[1:2]
        result = render_chart_asset_impl(
            chart_type=chart_type or "bar",
            rows=rows,
            csv_text=csv_text,
            csv_path=csv_path,
            x_field=inferred_x,
            y_fields=normalized_y_fields,
            series_field=series_field,
            title=title or caption,
            subtitle=subtitle,
            x_label=x_label,
            y_label=y_label,
            note=note,
            width=width,
            height=height,
            output_format=chart_output_format,
            output_stem=output_stem,
            style_overrides=style_overrides,
            workspace=Path.cwd(),
            config=LLM_CONFIG,
        )
        result.setdefault("warnings", []).append(
            "render_table_asset received chart-like arguments and routed to render_chart_asset."
        )
        info(
            "Chart-like table call routed to chart renderer: type='%s', title='%s', outputs=%s",
            chart_type or "bar",
            title or caption or "chart",
            result.get("rendered_paths", {}),
        )
        return result

    result = render_table_asset_impl(
        rows=rows,
        columns=columns,
        csv_text=csv_text,
        csv_path=csv_path,
        markdown_table=markdown_table,
        style=style,
        caption=caption or title,
        footnote=footnote,
        output_mode=output_format or output_mode or "html",
        width=width,
        output_stem=output_stem,
        style_overrides=style_overrides,
        workspace=Path.cwd(),
        config=LLM_CONFIG,
    )
    info(
        "Structured table asset rendered: style='%s', caption='%s', outputs=%s",
        style,
        caption or title or "table",
        result.get("rendered_paths", {}),
    )
    return result


@mcp.tool()
def render_flowchart_asset(
    nodes: list[object] | str,
    edges: list[object] | str | None = None,
    diagram_kind: str = "pipeline",
    title: str = "",
    subtitle: str = "",
    width: int = 960,
    height: int = 520,
    output_format: str = "svg",
    output_stem: str = "",
    style_overrides: dict[str, object] | None = None,
) -> dict:
    """
    Render a deterministic flowchart/pipeline asset from structured nodes.

    Args:
        nodes: Ordered node labels, or a delimited string like "Input -> Score -> Output".
        edges: Optional edge list. Items may be "A -> B" strings or {"from": "A", "to": "B"} objects.
        diagram_kind: One of pipeline, flowchart, branch_pipeline, architecture.
        title: Visible diagram title.
        subtitle: Optional subtitle.
        width: SVG width in pixels.
        height: SVG height in pixels.
        output_format: svg, png, or both.
        output_stem: Optional filename stem prefix.
        style_overrides: Optional design tokens / visual style overrides.
    """
    result = render_flowchart_asset_impl(
        nodes=nodes,
        edges=edges,
        diagram_kind=diagram_kind,
        title=title,
        subtitle=subtitle,
        width=width,
        height=height,
        output_format=output_format,
        output_stem=output_stem,
        style_overrides=style_overrides,
        workspace=Path.cwd(),
        config=LLM_CONFIG,
    )
    info(
        "Structured flowchart asset rendered: kind='%s', title='%s', outputs=%s",
        diagram_kind,
        title or diagram_kind,
        result.get("rendered_paths", {}),
    )
    return result


def _extractive_document_summary(task: str, document: str, *, max_chars: int = 8000) -> str:
    """Return a deterministic fallback summary when the long-context model is unavailable."""
    lines = [line.strip() for line in (document or "").splitlines()]
    headings: list[str] = []
    image_refs: list[str] = []
    table_refs: list[str] = []
    candidate_lines: list[str] = []
    keywords = [
        "abstract",
        "introduction",
        "attention",
        "transformer",
        "model",
        "training",
        "results",
        "bleu",
        "table",
        "figure",
        "conclusion",
        "architecture",
    ]
    for line in lines:
        if not line:
            continue
        lowered = line.lower()
        if line.startswith("#"):
            headings.append(line.lstrip("# ").strip())
            continue
        if "![" in line and "](" in line:
            image_refs.append(line)
            continue
        if "table" in lowered:
            table_refs.append(line)
        if any(token in lowered for token in keywords):
            candidate_lines.append(line)
    if not candidate_lines:
        candidate_lines = [line for line in lines if len(line) >= 40][:30]

    def _clip(items: list[str], limit: int) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for item in items:
            compact = " ".join(item.split())
            if not compact or compact in seen:
                continue
            seen.add(compact)
            output.append(compact)
            if len(output) >= limit:
                break
        return output

    sections = [
        "# Document Summary",
        "",
        "> Long-context LLM summary was unavailable; this is a deterministic extractive fallback so generation can continue.",
        "",
        f"## Task\n{task.strip() or 'Summarize the document for slide generation.'}",
        "",
    ]
    clipped_headings = _clip(headings, 12)
    if clipped_headings:
        sections.extend(["## Major Sections", *[f"- {item}" for item in clipped_headings], ""])
    clipped_lines = _clip(candidate_lines, 24)
    if clipped_lines:
        sections.extend(["## Key Extracted Points", *[f"- {item}" for item in clipped_lines], ""])
    clipped_images = _clip(image_refs, 10)
    if clipped_images:
        sections.extend(["## Referenced Visual Assets", *[f"- {item}" for item in clipped_images], ""])
    clipped_tables = _clip(table_refs, 8)
    if clipped_tables:
        sections.extend(["## Table Signals", *[f"- {item}" for item in clipped_tables], ""])
    summary = "\n".join(sections).strip()
    return summary[:max_chars]


def _summary_llm_candidates():
    candidates = []
    seen: set[int] = set()
    for name in ("long_context_model", "design_agent"):
        llm = getattr(LLM_CONFIG, name, None)
        if llm is None:
            continue
        ident = id(llm)
        if ident in seen:
            continue
        seen.add(ident)
        candidates.append((name, llm))
    return candidates


async def _document_summary_impl(task: str, document_path: str) -> str:
    """
    Generate a report according to the given task and long document.

    Args:
        task: The specific task or objective for the report
        document_path: Path to the pure text document to be analyzed, should be endswith like .txt or .md

    Returns:
        A structured summary report in Markdown format based on the task and document content
    """
    if not Path(document_path).exists():
        return "Document path does not exist"
    if Path(document_path).suffix.lower() not in [".txt", ".md"]:
        return "Document must be a text file with .txt or .md extension"
    with open(document_path, encoding="utf-8") as f:
        document = f.read()
    errors: list[str] = []
    for llm_name, llm in _summary_llm_candidates():
        try:
            response = await llm.run(
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Task: {task}\nDocument: {document}",
                    },
                ],
            )
            report = response.choices[0].message.content
            debug(
                "Document analyzed with %s: path='%s', task='%s', report='%s'",
                llm_name,
                document_path,
                task,
                report,
            )
            return report
        except Exception as exc:
            errors.append(f"{llm_name}: {exc}")
            warning(
                "document_summary via %s unavailable for path='%s': %s",
                llm_name,
                document_path,
                exc,
            )
    warning(
        "document_summary exhausted configured LLMs for path='%s'; returning extractive fallback: %s",
        document_path,
        " | ".join(errors),
    )
    return _extractive_document_summary(task, document)


@mcp.tool()
async def document_summary(task: str, document_path: str) -> str:
    return await _document_summary_impl(task=task, document_path=document_path)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python -m memslides.tools.asset_tools <workspace>"
    work_dir = Path(sys.argv[1])
    assert work_dir.exists(), f"Workspace {work_dir} does not exist."
    os.chdir(work_dir)
    set_logger(
        f"memslides-asset-tools-{work_dir.stem}",
        work_dir / ".history" / "memslides_asset_tools.log",
    )

    mcp.run(show_banner=False)
