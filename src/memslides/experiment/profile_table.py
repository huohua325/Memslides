from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


PROFILE_SUMMARY_FILENAME = "profile_holistic_v6_probe_summary.json"
PROFILE_MAIN_CSV = "profile_memory_v6_bestof_main_table.csv"
PROFILE_MAIN_TEX = "profile_memory_v6_bestof_main_table.tex"
PROFILE_SUMMARY_MD = "profile_memory_v6_bestof_summary.md"


@dataclass(frozen=True)
class ProfileDimensionSpec:
    dimension_id: str
    column: str
    tex_label: str


PROFILE_DIMENSIONS: tuple[ProfileDimensionSpec, ...] = (
    ProfileDimensionSpec("content_alignment", "Content", r"Content $\uparrow$"),
    ProfileDimensionSpec("structure_alignment", "Structure", r"Structure $\uparrow$"),
    ProfileDimensionSpec("visual_alignment", "Visual", r"Visual $\uparrow$"),
    ProfileDimensionSpec("profile_specificity", "Specificity", r"Specificity $\uparrow$"),
)


FRAMEWORK_ORDER = {
    "DeepPresenter": 0,
    "SlideTailor": 1,
    "MemSlides (Ours)": 2,
}
MODEL_ORDER = {
    "GPT-5": 0,
    "GLM-5": 1,
    "Gemini 3.1 Pro": 2,
}


@dataclass
class ProfileSummaryResult:
    framework: str
    model: str
    source_summary: Path
    scores: dict[str, float | None]
    probe_counts: dict[str, int] = field(default_factory=dict)
    paired_probes: int | None = None
    score_field: str = "primary_avg_score"
    warnings: list[str] = field(default_factory=list)


@dataclass
class ProfileTableRow:
    framework: str
    model: str
    scores: dict[str, float | None]
    paired_probes: int | None
    source_count: int
    source_summaries: list[Path]
    warnings: list[str] = field(default_factory=list)


def build_profile_table_report(suite_output_dir: Path | str, output_dir: Path | str) -> dict[str, Any]:
    """Build the OSS persona-alignment main table from local V6 summary files."""
    suite_output_dir = Path(suite_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results, warnings = collect_profile_results(suite_output_dir)
    rows = aggregate_profile_results(results)
    warnings.extend(warning for row in rows for warning in row.warnings)

    csv_path = output_dir / PROFILE_MAIN_CSV
    tex_path = output_dir / PROFILE_MAIN_TEX
    summary_path = output_dir / PROFILE_SUMMARY_MD

    write_profile_csv(csv_path, rows)
    write_profile_latex(tex_path, rows)
    summary_path.write_text(
        build_profile_summary_markdown(suite_output_dir, output_dir, rows, warnings),
        encoding="utf-8",
    )

    return {
        "suite_output_dir": str(suite_output_dir),
        "output_dir": str(output_dir),
        "summary_count": len(results),
        "row_count": len(rows),
        "warnings": warnings,
        "outputs": {
            "csv": str(csv_path),
            "tex": str(tex_path),
            "summary": str(summary_path),
        },
    }


def collect_profile_results(root: Path | str) -> tuple[list[ProfileSummaryResult], list[str]]:
    root = Path(root)
    warnings: list[str] = []
    summary_paths = discover_profile_summary_paths(root)
    if not summary_paths:
        return [], [f"No {PROFILE_SUMMARY_FILENAME} files found under {root}."]

    results: list[ProfileSummaryResult] = []
    for summary_path in summary_paths:
        payload = _safe_load_json(summary_path)
        if not payload:
            warnings.append(f"Skipped unreadable profile summary: {summary_path}")
            continue
        result = parse_profile_summary(summary_path, payload, root)
        results.append(result)
        warnings.extend(result.warnings)
    return results, warnings


def discover_profile_summary_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.name == PROFILE_SUMMARY_FILENAME else []
    if not root.exists():
        return []
    return sorted(root.rglob(PROFILE_SUMMARY_FILENAME))


def parse_profile_summary(summary_path: Path, payload: dict[str, Any], root: Path) -> ProfileSummaryResult:
    metadata = _profile_metadata(summary_path, payload, root)
    framework = _normalize_framework(str(metadata.get("framework") or ""))
    model = _normalize_model(str(metadata.get("model") or ""))
    warnings: list[str] = []
    if not framework:
        framework = _infer_framework(summary_path)
        if not framework:
            framework = "Unspecified"
            warnings.append(f"{summary_path}: could not infer framework label.")
    if not model:
        model = _infer_model(summary_path)
        if not model:
            model = summary_path.parent.name
            warnings.append(f"{summary_path}: could not infer model label; using directory name.")

    score_field = str(metadata.get("score_field") or "primary_avg_score")
    dimension_rows = payload.get("dimension_rows")
    if not isinstance(dimension_rows, list):
        dimension_rows = []

    by_dimension = {
        str(row.get("dimension_id") or ""): row
        for row in dimension_rows
        if isinstance(row, dict)
    }
    scores: dict[str, float | None] = {}
    probe_counts: dict[str, int] = {}
    for spec in PROFILE_DIMENSIONS:
        row = by_dimension.get(spec.dimension_id)
        if row is None:
            scores[spec.column] = None
            warnings.append(f"{summary_path}: missing dimension {spec.dimension_id}.")
            continue
        scores[spec.column] = _coerce_float(_score_value(row, score_field))
        probe_count = _coerce_int(row.get("probe_count"))
        if probe_count is not None:
            probe_counts[spec.column] = probe_count
        if scores[spec.column] is None:
            warnings.append(f"{summary_path}: no numeric {score_field} for {spec.dimension_id}.")

    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    paired_probes = _coerce_int(overall.get("paired_probe_count"))
    if paired_probes is None and probe_counts:
        paired_probes = max(probe_counts.values())

    return ProfileSummaryResult(
        framework=framework,
        model=model,
        source_summary=summary_path,
        scores=scores,
        probe_counts=probe_counts,
        paired_probes=paired_probes,
        score_field=score_field,
        warnings=warnings,
    )


def aggregate_profile_results(results: list[ProfileSummaryResult]) -> list[ProfileTableRow]:
    groups: dict[tuple[str, str], list[ProfileSummaryResult]] = {}
    for result in results:
        groups.setdefault((result.framework, result.model), []).append(result)

    rows: list[ProfileTableRow] = []
    for (framework, model), items in groups.items():
        scores: dict[str, float | None] = {}
        warnings: list[str] = []
        for spec in PROFILE_DIMENSIONS:
            values: list[tuple[float, int]] = []
            for item in items:
                value = item.scores.get(spec.column)
                if value is None:
                    continue
                weight = item.probe_counts.get(spec.column) or item.paired_probes or 1
                values.append((value, max(1, int(weight))))
            scores[spec.column] = _weighted_mean(values)
            if scores[spec.column] is None:
                warnings.append(f"{framework} / {model}: missing aggregate value for {spec.column}.")
        paired_values = [item.paired_probes for item in items if item.paired_probes is not None]
        paired_probes = sum(paired_values) if len(items) > 1 and paired_values else (paired_values[0] if paired_values else None)
        rows.append(
            ProfileTableRow(
                framework=framework,
                model=model,
                scores=scores,
                paired_probes=paired_probes,
                source_count=len(items),
                source_summaries=[item.source_summary for item in items],
                warnings=warnings,
            )
        )
    return sorted(rows, key=_profile_row_sort_key)


def write_profile_csv(path: Path, rows: list[ProfileTableRow]) -> None:
    fieldnames = [
        "framework",
        "model",
        "paired_probes",
        "source_count",
        *(spec.column for spec in PROFILE_DIMENSIONS),
        "source_summaries",
    ]
    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {
            "framework": row.framework,
            "model": row.model,
            "paired_probes": row.paired_probes if row.paired_probes is not None else "",
            "source_count": row.source_count,
            "source_summaries": ";".join(str(path) for path in row.source_summaries),
        }
        for spec in PROFILE_DIMENSIONS:
            record[spec.column] = _format_value(row.scores.get(spec.column), digits=3)
        records.append(record)
    _write_csv(path, records, fieldnames)


def write_profile_latex(path: Path, rows: list[ProfileTableRow]) -> None:
    best_by_column = _best_values(rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\caption{Persona-alignment judgments for first-pass generation. Scores are averaged over local V6 persona-alignment summaries on a 0--10 scale; higher is better. Bold marks the best score per metric among the generated rows.}",
        r"\label{tab:profile_memory_v6_bestof_main_oss}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        "Framework & Model & "
        + " & ".join(spec.tex_label for spec in PROFILE_DIMENSIONS)
        + r" \\",
        r"\midrule",
    ]
    last_framework = None
    for row in rows:
        if last_framework is not None and row.framework != last_framework:
            lines.append(r"\midrule")
        last_framework = row.framework
        cells = [tex_escape(row.framework), tex_escape(row.model)]
        for spec in PROFILE_DIMENSIONS:
            value = row.scores.get(spec.column)
            text = _format_value(value, digits=2)
            if value is not None and _is_best(value, best_by_column.get(spec.column)):
                text = rf"\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_profile_summary_markdown(
    suite_output_dir: Path,
    output_dir: Path,
    rows: list[ProfileTableRow],
    warnings: list[str],
) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "Framework": row.framework,
                "Model": row.model,
                "Paired probes": row.paired_probes if row.paired_probes is not None else "NA",
                **{spec.column: _format_value(row.scores.get(spec.column), digits=3) for spec in PROFILE_DIMENSIONS},
            }
        )
    lines = [
        "# Profile Memory V6 Main Table",
        "",
        f"- Source: `{suite_output_dir}`",
        f"- Output: `{output_dir}`",
        "- Scope: Content, Structure, Visual, and Specificity only.",
        "- Score source: `profile_holistic_v6_probe_summary.json -> dimension_rows[*].primary_avg_score` unless a summary sidecar explicitly sets `score_field`.",
        "",
        "## Main Rows",
        "",
        _markdown_table(
            table_rows,
            ["Framework", "Model", "Paired probes", "Content", "Structure", "Visual", "Specificity"],
        )
        if table_rows
        else "_No profile summaries were found._",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Excluded Metrics",
            "",
            "This OSS table layer intentionally excludes PPTEval quality, SlideTailor fair eval v3, profile span/visual span, and working-memory case-study metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def _profile_metadata(summary_path: Path, payload: dict[str, Any], root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("main_table", "table", "metadata", "suite_metadata"):
        value = payload.get(key)
        if isinstance(value, dict):
            metadata.update(value)
    for key in ("framework", "framework_label", "model", "model_label", "score_field"):
        if key in payload:
            metadata[key] = payload[key]

    for filename in ("profile_table_metadata.json", "main_table_metadata.json", "table_metadata.json"):
        sidecar = summary_path.parent / filename
        if sidecar.exists():
            value = _safe_load_json(sidecar)
            if value:
                metadata.update(value)

    manifest = root / "profile_table_manifest.json"
    if manifest.exists():
        payload = _safe_load_json(manifest)
        entries = payload.get("summaries") if isinstance(payload.get("summaries"), list) else []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            raw_path = entry.get("path") or entry.get("summary")
            if not raw_path:
                continue
            candidate = (root / str(raw_path)).resolve()
            if candidate == summary_path.resolve():
                metadata.update(entry)
                break
    return metadata


def _score_value(row: dict[str, Any], score_field: str) -> Any:
    if score_field in row:
        return row.get(score_field)
    for fallback in ("primary_avg_score", "primary_score", "mean_primary_score", "score"):
        if fallback in row:
            return row.get(fallback)
    if score_field == "mean_score_lift":
        return row.get("mean_score_lift")
    return None


def _infer_framework(path: Path) -> str:
    text = path.as_posix().lower()
    if "slidetailor" in text:
        return "SlideTailor"
    if "deeppresenter" in text:
        return "DeepPresenter"
    if "memslides" in text or "ours" in text or "profile_align" in text:
        return "MemSlides (Ours)"
    return ""


def _infer_model(path: Path) -> str:
    return _normalize_model(path.as_posix())


def _normalize_framework(value: str) -> str:
    text = value.strip()
    lowered = text.lower().replace("_", " ").replace("-", " ")
    if not text:
        return ""
    if "memslides" in lowered or lowered in {"ours", "our method"}:
        return "MemSlides (Ours)"
    if "deeppresenter" in lowered or "deep presenter" in lowered:
        return "DeepPresenter"
    if "slidetailor" in lowered or "slide tailor" in lowered:
        return "SlideTailor"
    return text


def _normalize_model(value: str) -> str:
    text = value.strip()
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    if not text:
        return ""
    if "gpt5" in compact:
        return "GPT-5"
    if "glm5" in compact:
        return "GLM-5"
    if "gemini31pro" in compact or "gemini3.1pro" in text.lower() or "gemini" in compact:
        return "Gemini 3.1 Pro"
    return text


def _profile_row_sort_key(row: ProfileTableRow) -> tuple[int, str, int, str]:
    return (
        FRAMEWORK_ORDER.get(row.framework, 99),
        row.framework,
        MODEL_ORDER.get(row.model, 99),
        row.model,
    )


def _weighted_mean(values: list[tuple[float, int]]) -> float | None:
    if not values:
        return None
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return mean(value for value, _ in values)
    return sum(value * weight for value, weight in values) / total_weight


def _best_values(rows: list[ProfileTableRow]) -> dict[str, float]:
    best: dict[str, float] = {}
    for spec in PROFILE_DIMENSIONS:
        values = [row.scores.get(spec.column) for row in rows if row.scores.get(spec.column) is not None]
        if values:
            best[spec.column] = max(value for value in values if value is not None)
    return best


def _is_best(value: float, best: float | None) -> bool:
    return best is not None and math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-9)


def _safe_load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: float | None, *, digits: int) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


__all__ = [
    "PROFILE_DIMENSIONS",
    "build_profile_table_report",
    "collect_profile_results",
    "aggregate_profile_results",
]
