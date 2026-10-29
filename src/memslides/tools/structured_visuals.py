from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import unicodedata
from html import escape
from pathlib import Path
from typing import Any

from memslides.memory.core.template_models import DesignConstraints
from memslides.utils.config import MemSlidesConfig
from memslides.utils.log import warning

try:
    import vl_convert as vlc
except ImportError:  # pragma: no cover - exercised via runtime failure path
    vlc = None


DEFAULT_FONT_STACK = "'Noto Sans CJK SC', 'Microsoft YaHei', Arial, sans-serif"
CJK_FONT_STACK = "'Noto Sans CJK SC', 'Source Han Sans SC', 'Microsoft YaHei', 'PingFang SC', Arial, sans-serif"
_CHART_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"
_REGISTERED_FONT_DIRS: set[str] = set()
_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
)


class StructuredVisualError(ValueError):
    """Raised when a chart/table request is invalid."""


_FLOWCHART_NODE_ALIASES = {
    "input": "Input",
    "输入": "Input",
    "输入 token": "Input Token",
    "输入token": "Input Token",
    "token": "Input Token",
    "tokens": "Input Token",
    "w_q": "W_Q",
    "wq": "W_Q",
    "w_k": "W_K",
    "wk": "W_K",
    "w_v": "W_V",
    "wv": "W_V",
    "q": "Q",
    "k": "K",
    "v": "V",
    "attention score": "Attention Score",
    "attention scores": "Attention Score",
    "注意力分数": "Attention Score",
    "softmax": "Softmax",
    "weighted sum": "Weighted Sum",
    "加权求和": "Weighted Sum",
    "output": "Output",
    "输出": "Output",
}


def _slugify(value: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z_-]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower() or "visual"


def _json_hash(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _workspace_path(workspace: str | Path | None = None) -> Path:
    base = Path(workspace) if workspace is not None else Path.cwd()
    return base.expanduser().resolve()


def _ensure_within_workspace(path: Path, workspace: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != workspace and workspace not in resolved.parents:
        raise StructuredVisualError(
            f"Path '{resolved}' must stay within workspace '{workspace}'."
        )
    return resolved


def _resolve_workspace_file(path: str, workspace: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    return _ensure_within_workspace(candidate, workspace)


def generated_visuals_dir(workspace: str | Path | None = None) -> Path:
    base = _workspace_path(workspace)
    target = base / "generated_visuals"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _coerce_chart_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return ""
    numeric = text.replace(",", "")
    if re.fullmatch(r"-?\d+", numeric):
        try:
            return int(numeric)
        except Exception:
            return text
    if re.fullmatch(r"-?\d+(?:\.\d+)?%?", numeric):
        try:
            return float(numeric[:-1] if numeric.endswith("%") else numeric)
        except Exception:
            return text
    return text


def _rows_from_csv_text(csv_text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise StructuredVisualError("CSV input must contain a header row.")
    rows = []
    for raw_row in reader:
        row = {str(key): _coerce_chart_scalar(value) for key, value in (raw_row or {}).items()}
        if any(str(value).strip() for value in row.values()):
            rows.append(row)
    if not rows:
        raise StructuredVisualError("CSV input produced no data rows.")
    return rows


def _normalize_rows_payload(rows: Any) -> list[dict[str, Any]]:
    if isinstance(rows, str):
        try:
            rows = json.loads(rows)
        except Exception as exc:
            raise StructuredVisualError(f"Failed to parse `rows` JSON: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise StructuredVisualError("`rows` must be a non-empty list.")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(rows, start=1):
        if isinstance(item, dict):
            normalized.append({str(key): _coerce_chart_scalar(value) for key, value in item.items()})
            continue
        raise StructuredVisualError(f"`rows[{index}]` must be an object/dict.")
    return normalized


def load_chart_rows(
    *,
    rows: Any = None,
    csv_text: str = "",
    csv_path: str = "",
    workspace: str | Path | None = None,
) -> list[dict[str, Any]]:
    provided = int(rows is not None) + int(bool(csv_text)) + int(bool(csv_path))
    if provided != 1:
        raise StructuredVisualError("Provide exactly one of `rows`, `csv_text`, or `csv_path`.")
    workspace_path = _workspace_path(workspace)
    if rows is not None:
        return _normalize_rows_payload(rows)
    if csv_text:
        return _rows_from_csv_text(csv_text)
    resolved = _resolve_workspace_file(csv_path, workspace_path)
    if not resolved.exists():
        raise StructuredVisualError(f"CSV path does not exist: {resolved}")
    return _rows_from_csv_text(resolved.read_text(encoding="utf-8"))


def _normalize_table_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isfinite(value):
        if value.is_integer():
            return str(int(value))
    return str(value)


def _parse_markdown_table(markdown_table: str) -> tuple[list[str], list[dict[str, Any]]]:
    lines = [line.rstrip() for line in str(markdown_table or "").splitlines() if line.strip()]
    if len(lines) < 2:
        raise StructuredVisualError("Markdown table must contain at least a header and one row.")

    def _split(line: str) -> list[str]:
        content = line.strip().strip("|")
        return [cell.strip() for cell in content.split("|")]

    header = _split(lines[0])
    divider = _split(lines[1])
    if len(header) == 0 or not all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in divider):
        raise StructuredVisualError("Markdown table alignment row is invalid.")
    rows: list[dict[str, Any]] = []
    for raw_line in lines[2:]:
        cells = _split(raw_line)
        if len(cells) < len(header):
            cells.extend([""] * (len(header) - len(cells)))
        row = {header[idx]: _normalize_table_cell(cells[idx]) for idx in range(len(header))}
        rows.append(row)
    if not rows:
        raise StructuredVisualError("Markdown table produced no data rows.")
    return header, rows


def load_table_rows(
    *,
    rows: Any = None,
    columns: list[str] | None = None,
    csv_text: str = "",
    csv_path: str = "",
    markdown_table: str = "",
    workspace: str | Path | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    provided = (
        int(rows is not None)
        + int(bool(csv_text))
        + int(bool(csv_path))
        + int(bool(markdown_table))
    )
    if provided != 1:
        raise StructuredVisualError(
            "Provide exactly one table input source: `rows`, `csv_text`, `csv_path`, or `markdown_table`."
        )
    workspace_path = _workspace_path(workspace)
    if markdown_table:
        return _parse_markdown_table(markdown_table)
    if csv_text or csv_path:
        chart_rows = load_chart_rows(
            rows=None,
            csv_text=csv_text,
            csv_path=csv_path,
            workspace=workspace_path,
        )
        inferred_columns = list(chart_rows[0].keys())
        normalized_rows = [
            {name: _normalize_table_cell(row.get(name, "")) for name in inferred_columns}
            for row in chart_rows
        ]
        return inferred_columns, normalized_rows

    normalized_rows = _normalize_rows_payload(rows)
    column_names = [str(value) for value in (columns or []) if str(value).strip()]
    if not column_names:
        column_names = list(normalized_rows[0].keys())
    if not column_names:
        raise StructuredVisualError("Unable to infer table columns from empty row objects.")
    shaped_rows = [
        {name: _normalize_table_cell(row.get(name, "")) for name in column_names}
        for row in normalized_rows
    ]
    return column_names, shaped_rows


def _display_width(text: str) -> float:
    total = 0.0
    for char in str(text):
        total += 2.0 if unicodedata.east_asian_width(char) in {"F", "W"} else 1.0
    return total


def _contains_cjk(value: Any) -> bool:
    return bool(_CJK_RE.search(str(value or "")))


def _contains_cjk_payload(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_cjk_payload(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_cjk_payload(item) for item in value)
    return _contains_cjk(value)


def _empty_cell_placeholder(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return text or "-"


def _wrap_display_text(value: Any, max_units: float, max_lines: int = 3) -> list[str]:
    text = " ".join(_empty_cell_placeholder(value).split())
    if not text:
        return ["-"]
    if _display_width(text) <= max_units:
        return [text]
    tokens = text.split(" ")
    if len(tokens) == 1:
        tokens = list(text)
        joiner = ""
    else:
        joiner = " "
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = token if not current else f"{current}{joiner}{token}"
        if _display_width(candidate) <= max_units or not current:
            current = candidate
            continue
        lines.append(current)
        current = token
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if not lines:
        lines = [text]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if len(lines) == max_lines and _display_width(lines[-1]) > max_units:
        clipped = ""
        for char in lines[-1]:
            suffix = "..."
            if _display_width(clipped + char + suffix) > max_units:
                break
            clipped += char
        lines[-1] = (clipped.rstrip() or lines[-1][:1]) + "..."
    elif len(lines) == max_lines and "".join(lines).replace(" ", "") != text.replace(" ", ""):
        clipped = lines[-1]
        while clipped and _display_width(clipped + "...") > max_units:
            clipped = clipped[:-1]
        lines[-1] = (clipped.rstrip() or lines[-1][:1]) + "..."
    return lines


def _svg_text_block(
    *,
    lines: list[str],
    x: float,
    y: float,
    font: str,
    font_size: float,
    fill: str,
    line_gap: float,
    anchor: str = "start",
    weight: str = "400",
    extra: str = "",
) -> str:
    parts = []
    for index, line in enumerate(lines):
        parts.append(
            f'<text x="{x:.1f}" y="{y + index * line_gap:.1f}" text-anchor="{anchor}" '
            f'font-family="{escape(font)}" font-size="{font_size:g}" font-weight="{escape(weight)}" '
            f'fill="{escape(fill)}" {extra}>{escape(line)}</text>'
        )
    return "".join(parts)


def _to_design_constraints(style_overrides: dict[str, Any] | None) -> DesignConstraints:
    payload = dict(style_overrides or {})
    nested = payload.get("design_constraints")
    if isinstance(nested, dict):
        payload = nested
    if any(key in payload for key in ("color_palette", "typography", "chart_style", "spacing")):
        try:
            return DesignConstraints.from_dict(payload)
        except Exception:
            pass
    return DesignConstraints()


def _normalize_font_stack(font_name: str) -> str:
    candidate = str(font_name or "").strip()
    if not candidate:
        return DEFAULT_FONT_STACK
    if "sans-serif" in candidate.lower():
        return candidate
    return f"{candidate}, {DEFAULT_FONT_STACK}"


def resolve_visual_theme(
    *,
    config: MemSlidesConfig | None = None,
    style_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = dict(style_overrides or {})
    visual_cfg = getattr(config, "visual_assets", None)
    constraints = _to_design_constraints(overrides)
    palette = constraints.color_palette
    typography = constraints.typography
    chart_style = constraints.chart_style

    font_family = (
        overrides.get("font_family")
        or overrides.get("body_font")
        or typography.body_font
        or getattr(visual_cfg, "default_font_family", "")
        or DEFAULT_FONT_STACK
    )
    title_font = (
        overrides.get("title_font")
        or typography.title_font
        or getattr(visual_cfg, "default_font_family", "")
        or DEFAULT_FONT_STACK
    )
    caption_font = (
        overrides.get("caption_font")
        or typography.caption_font
        or typography.body_font
        or getattr(visual_cfg, "default_font_family", "")
        or DEFAULT_FONT_STACK
    )
    colors = list(overrides.get("color_scheme") or chart_style.color_scheme or palette.additional or [])
    if not colors:
        colors = [
            palette.primary or "#2563eb",
            palette.accent or "#0d9488",
            palette.secondary or "#d97706",
            "#9333ea",
            "#dc2626",
            "#475569",
        ]
    return {
        "font_family": _normalize_font_stack(font_family),
        "title_font": _normalize_font_stack(title_font),
        "caption_font": _normalize_font_stack(caption_font),
        "title_size": int(overrides.get("title_size") or typography.title_size or 24),
        "body_size": int(overrides.get("body_size") or typography.body_size or 16),
        "caption_size": int(overrides.get("caption_size") or typography.caption_size or 13),
        "line_height": float(overrides.get("line_height") or typography.line_height or 1.4),
        "text_color": str(overrides.get("text_color") or palette.text or palette.primary_text or "#0f172a"),
        "muted_text_color": str(overrides.get("muted_text_color") or palette.muted_text or "#475569"),
        "background_color": str(overrides.get("background_color") or palette.background or "#ffffff"),
        "surface_color": str(overrides.get("surface_color") or palette.surface or "#ffffff"),
        "border_color": str(overrides.get("border_color") or palette.border or "#cbd5e1"),
        "accent_color": str(overrides.get("accent_color") or palette.accent or colors[0]),
        "inverse_text_color": str(
            overrides.get("inverse_text_color") or palette.inverse_text or "#ffffff"
        ),
        "color_scheme": colors,
        "show_legend": bool(overrides.get("show_legend", chart_style.show_legend)),
        "show_grid": bool(overrides.get("show_grid", chart_style.show_grid)),
    }


def _ensure_vl_convert() -> Any:
    if vlc is None:
        raise StructuredVisualError(
            "vl-convert-python is required for structured chart rendering but is not installed."
        )
    return vlc


def register_visual_font_dirs(
    *,
    config: MemSlidesConfig | None = None,
    workspace: str | Path | None = None,
) -> list[str]:
    module = _ensure_vl_convert()
    visual_cfg = getattr(config, "visual_assets", None)
    configured_dirs = list(getattr(visual_cfg, "font_dirs", []) or [])
    if not configured_dirs:
        return []
    workspace_path = _workspace_path(workspace)
    registered: list[str] = []
    for raw_dir in configured_dirs:
        if not str(raw_dir or "").strip():
            continue
        path = Path(str(raw_dir)).expanduser()
        if not path.is_absolute():
            path = workspace_path / path
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if not resolved.exists() or not resolved.is_dir():
            warning("Structured visual font dir missing or not a directory: %s", resolved)
            continue
        key = str(resolved)
        if key in _REGISTERED_FONT_DIRS:
            registered.append(key)
            continue
        try:
            module.register_font_directory(key)
            _REGISTERED_FONT_DIRS.add(key)
            registered.append(key)
        except Exception as exc:  # pragma: no cover - depends on runtime fonts
            warning("Failed to register structured visual font dir %s: %s", key, exc)
    return registered


def _infer_vega_type(values: list[Any]) -> str:
    non_empty = [value for value in values if value not in {"", None}]
    if not non_empty:
        return "nominal"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in non_empty):
        return "quantitative"
    if all(
        isinstance(value, str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ t]\d{2}:\d{2}(?::\d{2})?)?", value.strip())
        for value in non_empty
    ):
        return "temporal"
    return "nominal"


def _prepare_chart_dataset(
    *,
    chart_type: str,
    source_rows: list[dict[str, Any]],
    x_field: str,
    y_fields: list[str],
    series_field: str = "",
) -> tuple[list[dict[str, Any]], str, str, str]:
    missing = {field for field in [x_field, *y_fields] if field and field not in source_rows[0]}
    if missing:
        raise StructuredVisualError(
            f"Missing required chart field(s): {', '.join(sorted(missing))}."
        )

    if series_field:
        if series_field not in source_rows[0]:
            raise StructuredVisualError(f"Series field '{series_field}' does not exist.")
        if len(y_fields) != 1:
            raise StructuredVisualError(
                "When `series_field` is provided, `y_fields` must contain exactly one value field."
            )
        return source_rows, x_field, y_fields[0], series_field

    if len(y_fields) == 1:
        return source_rows, x_field, y_fields[0], ""

    long_rows: list[dict[str, Any]] = []
    for row in source_rows:
        for y_field in y_fields:
            long_rows.append(
                {
                    x_field: row.get(x_field),
                    "__value__": row.get(y_field),
                    "__series__": y_field,
                }
            )
    derived_series_field = "__series__"
    if chart_type in {"pie", "donut"}:
        raise StructuredVisualError("Pie and donut charts accept exactly one `y_field`.")
    return long_rows, x_field, "__value__", derived_series_field


def build_chart_spec(
    *,
    chart_type: str,
    source_rows: list[dict[str, Any]],
    x_field: str,
    y_fields: list[str],
    series_field: str = "",
    title: str = "",
    subtitle: str = "",
    x_label: str = "",
    y_label: str = "",
    note: str = "",
    width: int = 960,
    height: int = 540,
    theme: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chart_kind = str(chart_type or "").strip().lower()
    if chart_kind not in {"line", "bar", "grouped_bar", "stacked_bar", "area", "scatter", "pie", "donut"}:
        raise StructuredVisualError(f"Unsupported chart_type '{chart_type}'.")
    if not x_field:
        raise StructuredVisualError("`x_field` is required.")
    if not y_fields:
        raise StructuredVisualError("`y_fields` must contain at least one field.")
    theme = dict(theme or {})
    dataset, resolved_x_field, resolved_y_field, resolved_series_field = _prepare_chart_dataset(
        chart_type=chart_kind,
        source_rows=source_rows,
        x_field=x_field,
        y_fields=y_fields,
        series_field=series_field,
    )
    x_type = _infer_vega_type([row.get(resolved_x_field) for row in dataset])
    y_type = _infer_vega_type([row.get(resolved_y_field) for row in dataset])
    title_payload: str | dict[str, Any] | None = None
    if title or subtitle:
        title_payload = {
            "text": title or chart_kind.replace("_", " ").title(),
            "subtitle": [text for text in [subtitle, note] if text],
            "anchor": "start",
            "color": theme.get("text_color", "#0f172a"),
            "font": theme.get("title_font", DEFAULT_FONT_STACK),
            "fontSize": theme.get("title_size", 24),
            "fontWeight": 700,
            "subtitleFont": theme.get("caption_font", DEFAULT_FONT_STACK),
            "subtitleFontSize": theme.get("caption_size", 13),
            "subtitleColor": theme.get("muted_text_color", "#475569"),
            "offset": 14,
        }

    spec: dict[str, Any] = {
        "$schema": _CHART_SCHEMA,
        "width": int(width),
        "height": int(height),
        "data": {"values": dataset},
        "config": {
            "background": theme.get("background_color", "#ffffff"),
            "view": {"stroke": None},
            "title": {
                "font": theme.get("title_font", DEFAULT_FONT_STACK),
                "fontSize": theme.get("title_size", 24),
                "color": theme.get("text_color", "#0f172a"),
            },
            "axis": {
                "labelFont": theme.get("font_family", DEFAULT_FONT_STACK),
                "labelFontSize": theme.get("body_size", 16),
                "labelColor": theme.get("muted_text_color", "#475569"),
                "titleFont": theme.get("font_family", DEFAULT_FONT_STACK),
                "titleFontSize": theme.get("body_size", 16),
                "titleColor": theme.get("text_color", "#0f172a"),
                "grid": bool(theme.get("show_grid", True)),
                "gridColor": theme.get("border_color", "#cbd5e1"),
                "domainColor": theme.get("border_color", "#cbd5e1"),
                "tickColor": theme.get("border_color", "#cbd5e1"),
            },
            "legend": {
                "labelFont": theme.get("font_family", DEFAULT_FONT_STACK),
                "labelFontSize": theme.get("caption_size", 13),
                "labelColor": theme.get("muted_text_color", "#475569"),
                "titleFont": theme.get("font_family", DEFAULT_FONT_STACK),
                "titleFontSize": theme.get("caption_size", 13),
                "titleColor": theme.get("text_color", "#0f172a"),
                "disable": not bool(theme.get("show_legend", True)),
            },
            "range": {
                "category": list(theme.get("color_scheme") or []),
            },
        },
    }
    if title_payload:
        spec["title"] = title_payload

    base_encode: dict[str, Any] = {
        "tooltip": [
            {"field": resolved_x_field, "title": x_label or resolved_x_field},
            {"field": resolved_y_field, "title": y_label or resolved_y_field},
        ]
    }
    if resolved_series_field:
        base_encode["tooltip"].append({"field": resolved_series_field, "title": resolved_series_field})

    if chart_kind in {"line", "area", "scatter"}:
        spec["mark"] = {
            "type": "line" if chart_kind == "line" else "area" if chart_kind == "area" else "point",
            "line": chart_kind == "scatter",
            "point": chart_kind in {"line", "area"},
            "filled": chart_kind == "scatter",
            "opacity": 0.85 if chart_kind == "area" else 1,
        }
        encode = {
            **base_encode,
            "x": {"field": resolved_x_field, "type": x_type, "title": x_label or resolved_x_field},
            "y": {"field": resolved_y_field, "type": y_type, "title": y_label or resolved_y_field},
        }
        if resolved_series_field:
            encode["color"] = {"field": resolved_series_field, "type": "nominal", "legend": None if not theme.get("show_legend", True) else {}}
        spec["encoding"] = encode
        return spec

    if chart_kind in {"bar", "grouped_bar", "stacked_bar"}:
        spec["mark"] = {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3}
        encode = {
            **base_encode,
            "x": {"field": resolved_x_field, "type": x_type, "title": x_label or resolved_x_field},
            "y": {"field": resolved_y_field, "type": "quantitative", "title": y_label or resolved_y_field},
        }
        if resolved_series_field:
            encode["color"] = {"field": resolved_series_field, "type": "nominal", "legend": None if not theme.get("show_legend", True) else {}}
        if chart_kind == "grouped_bar" and resolved_series_field:
            encode["xOffset"] = {"field": resolved_series_field}
        spec["encoding"] = encode
        return spec

    spec["mark"] = {
        "type": "arc",
        "innerRadius": 80 if chart_kind == "donut" else 0,
    }
    encode = {
        **base_encode,
        "theta": {"field": resolved_y_field, "type": "quantitative"},
        "color": {
            "field": resolved_x_field if not resolved_series_field else resolved_series_field,
            "type": "nominal",
            "legend": None if not theme.get("show_legend", True) else {},
        },
    }
    spec["encoding"] = encode
    return spec


def _structured_chart_summary(
    *,
    source_rows: list[dict[str, Any]],
    x_field: str,
    y_fields: list[str],
    series_field: str = "",
) -> dict[str, Any]:
    numeric_fields: list[str] = []
    for field in source_rows[0].keys():
        values = [row.get(field) for row in source_rows]
        if values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values if value not in {"", None}):
            numeric_fields.append(field)
    return {
        "row_count": len(source_rows),
        "field_names": list(source_rows[0].keys()),
        "numeric_fields": numeric_fields,
        "x_field": x_field,
        "y_fields": list(y_fields),
        "series_field": series_field or "",
        "preview": source_rows[:5],
    }


def _truncate_svg_text(value: Any, max_chars: int = 42) -> str:
    text = str(value if value is not None else "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)].rstrip() + "..."


def _build_chart_fallback_svg_markup(
    *,
    chart_type: str,
    source_rows: list[dict[str, Any]],
    x_field: str,
    y_fields: list[str],
    title: str,
    subtitle: str,
    width: int,
    height: int,
    theme: dict[str, Any],
) -> str:
    canvas_width = max(480, int(width or 960))
    row_height = 38
    header_height = 118 if subtitle else 92
    visible_rows = source_rows[:8]
    columns = [x_field, *list(y_fields or [])]
    if not columns:
        columns = list(source_rows[0].keys())[:4]
    body_height = row_height * (len(visible_rows) + 1) + 42
    canvas_height = max(int(height or 540), header_height + body_height)
    table_x = 36
    table_y = header_height
    table_width = canvas_width - table_x * 2
    col_w = table_width / max(1, len(columns))
    accent = str(theme.get("accent_color") or "#2563eb")
    border = str(theme.get("border_color") or "#cbd5e1")
    text = str(theme.get("text_color") or "#0f172a")
    muted = str(theme.get("muted_text_color") or "#475569")
    background = str(theme.get("background_color") or "#ffffff")
    surface = str(theme.get("surface_color") or "#ffffff")
    font = str(theme.get("font_family") or DEFAULT_FONT_STACK)
    title_text = title or chart_type.replace("_", " ").title()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" '
        f'viewBox="0 0 {canvas_width} {canvas_height}" role="img" '
        f'aria-label="{escape(title_text)}">',
        f'<rect width="100%" height="100%" fill="{escape(background)}" />',
        f'<text x="{table_x}" y="44" font-family="{escape(font)}" font-size="26" '
        f'font-weight="700" fill="{escape(text)}">{escape(_truncate_svg_text(title_text, 70))}</text>',
        f'<text x="{table_x}" y="72" font-family="{escape(font)}" font-size="13" '
        f'fill="{escape(muted)}">Fallback structured visual: {escape(chart_type)}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="{table_x}" y="98" font-family="{escape(font)}" font-size="15" '
            f'fill="{escape(muted)}">{escape(_truncate_svg_text(subtitle, 88))}</text>'
        )
    parts.extend(
        [
            f'<rect x="{table_x}" y="{table_y}" width="{table_width}" height="{row_height}" '
            f'rx="8" fill="{escape(accent)}" opacity="0.12" />',
            f'<line x1="{table_x}" y1="{table_y + row_height}" x2="{table_x + table_width}" '
            f'y2="{table_y + row_height}" stroke="{escape(accent)}" stroke-width="2" />',
        ]
    )
    for col_index, column in enumerate(columns):
        x = table_x + col_index * col_w + 12
        parts.append(
            f'<text x="{x}" y="{table_y + 25}" font-family="{escape(font)}" font-size="14" '
            f'font-weight="700" fill="{escape(text)}">{escape(_truncate_svg_text(column, 24))}</text>'
        )
    y = table_y + row_height
    for row_index, row in enumerate(visible_rows, start=1):
        fill = surface if row_index % 2 else background
        parts.append(
            f'<rect x="{table_x}" y="{y}" width="{table_width}" height="{row_height}" '
            f'fill="{escape(fill)}" stroke="{escape(border)}" stroke-width="1" />'
        )
        for col_index, column in enumerate(columns):
            x = table_x + col_index * col_w + 12
            value = row.get(column, "")
            parts.append(
                f'<text x="{x}" y="{y + 24}" font-family="{escape(font)}" font-size="13" '
                f'fill="{escape(text)}">{escape(_truncate_svg_text(value, 26))}</text>'
            )
        y += row_height
    if len(source_rows) > len(visible_rows):
        parts.append(
            f'<text x="{table_x}" y="{y + 26}" font-family="{escape(font)}" font-size="12" '
            f'fill="{escape(muted)}">+{len(source_rows) - len(visible_rows)} rows summarized in spec JSON</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _normalize_flowchart_node(value: Any) -> str:
    text = " ".join(str(value if value is not None else "").split()).strip()
    if not text:
        return ""
    key = text.lower().replace("／", "/")
    return _FLOWCHART_NODE_ALIASES.get(key, text)


def _normalize_flowchart_nodes(nodes: Any) -> list[str]:
    if isinstance(nodes, str):
        try:
            parsed = json.loads(nodes)
            nodes = parsed
        except Exception:
            nodes = re.split(r"\s*(?:->|→|,|，|、|到)\s*", nodes)
    if not isinstance(nodes, list):
        raise StructuredVisualError("`nodes` must be a list or a delimited string.")
    normalized: list[str] = []
    for item in nodes:
        if isinstance(item, dict):
            label = _normalize_flowchart_node(item.get("label") or item.get("text") or item.get("id"))
        else:
            label = _normalize_flowchart_node(item)
        if label and label not in normalized:
            normalized.append(label)
    if len(normalized) < 2:
        raise StructuredVisualError("A flowchart needs at least two nodes.")
    return normalized[:16]


def _normalize_flowchart_edges(edges: Any) -> list[dict[str, str]]:
    if not edges:
        return []
    if isinstance(edges, str):
        try:
            edges = json.loads(edges)
        except Exception:
            parts = [part.strip() for part in re.split(r"\s*(?:,|，|;|；)\s*", edges) if part.strip()]
            edges = parts
    if not isinstance(edges, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in edges:
        source = ""
        target = ""
        label = ""
        if isinstance(item, dict):
            source = _normalize_flowchart_node(item.get("from") or item.get("source") or item.get("src"))
            target = _normalize_flowchart_node(item.get("to") or item.get("target") or item.get("dst"))
            label = " ".join(str(item.get("label") or item.get("text") or item.get("condition") or "").split())
        else:
            text = str(item or "")
            labeled = re.match(r"^\s*(.*?)\s*(?:--|==|\|)\s*(.*?)\s*(?:-->|==>|->|→|=>|到)\s*(.*?)\s*$", text)
            if labeled:
                source = _normalize_flowchart_node(labeled.group(1))
                label = " ".join(str(labeled.group(2) or "").strip("| ").split())
                target = _normalize_flowchart_node(labeled.group(3))
                if source and target:
                    normalized.append({"source": source, "target": target, "label": label})
                continue
            match = re.split(r"\s*(?:->|→|=>|到)\s*", text, maxsplit=1)
            if len(match) == 2:
                source = _normalize_flowchart_node(match[0])
                target = _normalize_flowchart_node(match[1])
        if source and target:
            normalized.append({"source": source, "target": target, "label": label})
    return normalized[:24]


def _flowchart_theme(style_overrides: dict[str, Any] | None, config: MemSlidesConfig | None = None) -> dict[str, Any]:
    theme = resolve_visual_theme(config=config, style_overrides=style_overrides)
    return {
        "background": str(theme.get("background_color") or "#ffffff"),
        "surface": str(theme.get("surface_color") or "#f8fafc"),
        "text": str(theme.get("text_color") or "#111827"),
        "muted": str(theme.get("muted_text_color") or "#64748b"),
        "primary": str((style_overrides or {}).get("primary_color") or theme.get("accent_color") or "#1d3557"),
        "accent": str((style_overrides or {}).get("accent_color") or "#0d9488"),
        "border": str(theme.get("border_color") or "#cbd5e1"),
        "font": str(theme.get("font_family") or DEFAULT_FONT_STACK),
    }


def _svg_text_lines(text: str, max_chars: int = 18) -> list[str]:
    return _wrap_display_text(text, max_chars, max_lines=2)


def _arrow_svg(x1: float, y1: float, x2: float, y2: float, color: str) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{escape(color)}" stroke-width="3" stroke-linecap="round" marker-end="url(#arrowhead)" />'
    )


def _orthogonal_arrow_svg(
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    width: float = 3.0,
    label: str = "",
    font: str = DEFAULT_FONT_STACK,
    muted: str = "#64748b",
    waypoints: list[tuple[float, float]] | None = None,
) -> str:
    x1, y1 = start
    x2, y2 = end
    if waypoints:
        points = [start, *waypoints, end]
        d = " ".join(
            ("M" if index == 0 else "L") + f"{x:.1f},{y:.1f}"
            for index, (x, y) in enumerate(points)
        )
        segments = list(zip(points, points[1:]))
        label_segment = max(
            segments,
            key=lambda segment: abs(segment[1][0] - segment[0][0]) + abs(segment[1][1] - segment[0][1]),
        )
        label_x = (label_segment[0][0] + label_segment[1][0]) / 2
        label_y = (label_segment[0][1] + label_segment[1][1]) / 2 - 8
    elif abs(y1 - y2) < 1:
        d = f"M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}"
        label_x = (x1 + x2) / 2
        label_y = y1 - 8
    else:
        mid_x = (x1 + x2) / 2
        d = f"M{x1:.1f},{y1:.1f} L{mid_x:.1f},{y1:.1f} L{mid_x:.1f},{y2:.1f} L{x2:.1f},{y2:.1f}"
        label_x = mid_x
        label_y = min(y1, y2) + abs(y2 - y1) / 2 - 8
    parts = [
        f'<path d="{d}" fill="none" stroke="{escape(color)}" stroke-width="{width:g}" '
        'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrowhead)" />'
    ]
    if label:
        clipped = _truncate_svg_text(label, 22)
        label_w = max(48, min(170, _display_width(clipped) * 7.2 + 22))
        parts.append(
            f'<rect x="{label_x - label_w / 2:.1f}" y="{label_y - 17:.1f}" '
            f'width="{label_w:.1f}" height="22" rx="11" fill="#ffffff" '
            f'stroke="{escape(muted)}" stroke-width="1" opacity="0.92" />'
        )
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle" '
            f'font-family="{escape(font)}" font-size="12" font-weight="700" '
            f'fill="{escape(muted)}">{escape(clipped)}</text>'
        )
    return "".join(parts)


def _node_svg(
    *,
    label: str,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    stroke: str,
    text_color: str,
    font: str,
    accent: str,
    index: int,
    emphasis: str = "normal",
) -> str:
    lines = _svg_text_lines(label)
    text_y = y + height / 2 - (len(lines) - 1) * 9 + 5
    stroke_width = 2.8 if emphasis in {"start", "end", "critical"} else 1.6
    shadow = (
        f'<rect x="{x + 3:.1f}" y="{y + 4:.1f}" width="{width:.1f}" height="{height:.1f}" rx="14" '
        'fill="#0f172a" opacity="0.08" />'
    )
    parts = [
        shadow,
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="14" '
        f'fill="{escape(fill)}" stroke="{escape(stroke)}" stroke-width="{stroke_width:g}" />',
        f'<circle cx="{x + 18:.1f}" cy="{y + 18:.1f}" r="8" fill="{escape(accent)}" opacity="0.95" />',
        f'<text x="{x + 18:.1f}" y="{y + 21:.1f}" font-family="{escape(font)}" font-size="9" '
        f'font-weight="700" text-anchor="middle" fill="#ffffff">{index}</text>',
    ]
    for line_index, line in enumerate(lines):
        parts.append(
            f'<text x="{x + width / 2:.1f}" y="{text_y + line_index * 18:.1f}" '
            f'font-family="{escape(font)}" font-size="16" font-weight="700" '
            f'text-anchor="middle" fill="{escape(text_color)}">{escape(line)}</text>'
        )
    return "".join(parts)


def _build_branch_pipeline_svg(
    *,
    nodes: list[str],
    title: str,
    subtitle: str,
    width: int,
    height: int,
    theme: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    canvas_w = max(760, int(width or 960))
    canvas_h = max(420, int(height or 520))
    font = theme["font"]
    title_text = title or "Attention Pipeline"
    primary = theme["primary"]
    accent = theme["accent"]
    surface = theme["surface"]
    border = theme["border"]
    text = theme["text"]
    muted = theme["muted"]
    background = theme["background"]
    node_w, node_h = 150, 66
    pos = {
        "Input Token": (44, 200),
        "W_Q": (245, 96),
        "W_K": (245, 200),
        "W_V": (245, 304),
        "Q": (430, 96),
        "K": (430, 200),
        "V": (430, 304),
        "Attention Score": (626, 148),
        "Softmax": (626, 252),
        "Weighted Sum": (812, 252),
        "Output": (812, 96),
    }
    if canvas_w < 1020:
        scale = (canvas_w - 88) / 918
        pos = {key: (44 + (x - 44) * scale, y) for key, (x, y) in pos.items()}
        node_w = min(node_w, 130)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" '
        f'viewBox="0 0 {canvas_w} {canvas_h}" role="img" aria-label="{escape(title_text)}" '
        f'data-visual-kind="flowchart" data-preferred-pptx-export="raster" '
        f'data-contains-cjk="{str(_contains_cjk_payload([nodes, title, subtitle])).lower()}">',
        f'<rect width="100%" height="100%" fill="{escape(background)}" />',
        f'<style>text,tspan{{font-family:{escape(CJK_FONT_STACK)};}}</style>',
        '<defs><marker id="arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="'
        + escape(accent)
        + '" /></marker></defs>',
        f'<text x="36" y="44" font-family="{escape(font)}" font-size="26" font-weight="800" fill="{escape(primary)}">{escape(_truncate_svg_text(title_text, 72))}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="36" y="72" font-family="{escape(font)}" font-size="14" fill="{escape(muted)}">{escape(_truncate_svg_text(subtitle, 96))}</text>'
        )
    arrows = [
        ("Input Token", "W_Q"),
        ("Input Token", "W_K"),
        ("Input Token", "W_V"),
        ("W_Q", "Q"),
        ("W_K", "K"),
        ("W_V", "V"),
        ("Q", "Attention Score"),
        ("K", "Attention Score"),
        ("Attention Score", "Softmax"),
        ("Softmax", "Weighted Sum"),
        ("V", "Weighted Sum"),
        ("Weighted Sum", "Output"),
    ]
    for source, target in arrows:
        x1, y1 = pos[source]
        x2, y2 = pos[target]
        parts.append(_arrow_svg(x1 + node_w, y1 + node_h / 2, x2 - 8, y2 + node_h / 2, accent))
    ordered = [node for node in pos if node in nodes or node in {"Input Token", "W_Q", "W_K", "W_V", "Q", "K", "V", "Attention Score", "Softmax", "Weighted Sum", "Output"}]
    for index, node in enumerate(ordered, start=1):
        x, y = pos[node]
        fill = "#ffffff" if node in {"Q", "K", "V"} else surface
        stroke = accent if node in {"Attention Score", "Softmax", "Weighted Sum"} else border
        parts.append(
            _node_svg(
                label=node,
                x=x,
                y=y,
                width=node_w,
                height=node_h,
                fill=fill,
                stroke=stroke,
                text_color=text,
                font=font,
                accent=accent if index % 2 else primary,
                index=index,
                emphasis="critical" if node in {"Attention Score", "Softmax", "Weighted Sum"} else "normal",
            )
        )
    parts.append("</svg>")
    node_bounds = {
        node: {"x": round(pos[node][0], 2), "y": round(pos[node][1], 2), "w": node_w, "h": node_h}
        for node in ordered
    }
    layout_meta = {
        "node_bounds": node_bounds,
        "edges": [
            {"source": source, "target": target, "label": ""}
            for source, target in arrows
        ],
        "recommended_width": canvas_w,
        "recommended_height": canvas_h,
        "contains_cjk": _contains_cjk_payload([nodes, title, subtitle]),
        "uses_edges": True,
        "layout": "branch_pipeline",
    }
    return "".join(parts), layout_meta


def _flowchart_edges_for_layout(nodes: list[str], edges: list[dict[str, str]]) -> list[dict[str, str]]:
    node_set = set(nodes)
    normalized = [
        edge
        for edge in edges
        if edge.get("source") in node_set
        and edge.get("target") in node_set
        and edge.get("source") != edge.get("target")
    ]
    if normalized:
        return normalized
    return [
        {"source": source, "target": target, "label": ""}
        for source, target in zip(nodes, nodes[1:])
    ]


def _flowchart_rank_nodes(nodes: list[str], edges: list[dict[str, str]]) -> dict[str, int]:
    ranks = {node: 0 for node in nodes}
    for _ in range(len(nodes)):
        changed = False
        for edge in edges:
            src = edge.get("source", "")
            dst = edge.get("target", "")
            next_rank = min(len(nodes) - 1, ranks.get(src, 0) + 1)
            if next_rank > ranks.get(dst, 0):
                ranks[dst] = next_rank
                changed = True
        if not changed:
            break
    return ranks


def _build_linear_flowchart_svg(
    *,
    nodes: list[str],
    edges: list[dict[str, str]],
    title: str,
    subtitle: str,
    diagram_kind: str,
    width: int,
    height: int,
    theme: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    edges = _flowchart_edges_for_layout(nodes, edges)
    canvas_w = max(760, int(width or 960))
    font = theme["font"]
    primary = theme["primary"]
    accent = theme["accent"]
    surface = theme["surface"]
    border = theme["border"]
    text = theme["text"]
    muted = theme["muted"]
    background = theme["background"]
    title_text = title or diagram_kind.replace("_", " ").title()
    top = 128 if subtitle else 104
    left = 46
    node_w = min(206, max(150, (canvas_w - 118) / max(3, min(5, len(nodes)))))
    node_h = 82
    gap_x = max(42, (canvas_w - left * 2 - node_w * min(4, len(nodes))) / max(1, min(4, len(nodes)) - 1))
    gap_y = 52
    phase_pad_x = 16
    phase_pad_top = 34
    phase_pad_bottom = 14
    ranks = _flowchart_rank_nodes(nodes, edges)
    grouped: dict[int, list[str]] = {}
    for node in nodes:
        grouped.setdefault(ranks.get(node, 0), []).append(node)
    columns = sorted(grouped)
    max_cols_per_band = 4 if len(columns) > 5 else max(1, len(columns))
    node_w = min(206, max(156, (canvas_w - left * 2 - gap_x * (max_cols_per_band - 1)) / max_cols_per_band))
    rank_bands = [
        columns[index : index + max_cols_per_band]
        for index in range(0, len(columns), max_cols_per_band)
    ]
    band_heights = [
        max(
            (
                len(grouped[rank]) * node_h + max(0, len(grouped[rank]) - 1) * gap_y
                for rank in band
            ),
            default=node_h,
        )
        for band in rank_bands
    ]
    band_gap_y = 104 if len(rank_bands) > 1 else 0
    canvas_h = max(
        int(height or 520),
        top + sum(band_heights) + band_gap_y * max(0, len(rank_bands) - 1) + 52,
    )
    positions_by_node: dict[str, tuple[float, float]] = {}
    band_top = top
    for band_index, band in enumerate(rank_bands):
        band_height = band_heights[band_index]
        band_width = len(band) * node_w + max(0, len(band) - 1) * gap_x
        band_left = left + max(0, (canvas_w - left * 2 - band_width) / 2)
        for col_index, rank in enumerate(band):
            column_nodes = grouped[rank]
            col_h = len(column_nodes) * node_h + max(0, len(column_nodes) - 1) * gap_y
            y0 = band_top + max(0, (band_height - col_h) / 2)
            x = band_left + col_index * (node_w + gap_x)
            for row_index, node in enumerate(column_nodes):
                positions_by_node[node] = (x, y0 + row_index * (node_h + gap_y))
        band_top += band_height + band_gap_y
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" '
        f'viewBox="0 0 {canvas_w} {canvas_h}" role="img" aria-label="{escape(title_text)}" '
        f'data-visual-kind="flowchart" data-preferred-pptx-export="raster" '
        f'data-contains-cjk="{str(_contains_cjk_payload([nodes, title, subtitle])).lower()}">',
        f'<rect width="100%" height="100%" fill="{escape(background)}" />',
        f'<style>text,tspan{{font-family:{escape(CJK_FONT_STACK)};}}</style>',
        '<defs><marker id="arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="'
        + escape(accent)
        + '" /></marker></defs>',
        f'<rect x="0" y="0" width="7" height="{canvas_h}" fill="{escape(accent)}" />',
        f'<text x="36" y="44" font-family="{escape(font)}" font-size="26" font-weight="800" fill="{escape(primary)}">{escape(_truncate_svg_text(title_text, 72))}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="36" y="72" font-family="{escape(font)}" font-size="14" fill="{escape(muted)}">{escape(_truncate_svg_text(subtitle, 96))}</text>'
        )
    for rank in columns:
        x_values = [positions_by_node[node][0] for node in grouped[rank] if node in positions_by_node]
        if not x_values:
            continue
        x = min(x_values) - phase_pad_x
        node_top = min(positions_by_node[node][1] for node in grouped[rank])
        y0 = node_top - phase_pad_top
        y1 = max(positions_by_node[node][1] + node_h for node in grouped[rank]) + 14
        parts.append(
            f'<rect x="{x:.1f}" y="{y0:.1f}" width="{node_w + phase_pad_x * 2:.1f}" height="{y1 - y0:.1f}" '
            f'rx="18" fill="{escape(surface)}" stroke="{escape(border)}" stroke-width="1" opacity="0.34" />'
        )
        parts.append(
            f'<text x="{x + 14:.1f}" y="{node_top - 12:.1f}" font-family="{escape(font)}" font-size="11" '
            f'font-weight="800" fill="{escape(muted)}">PHASE {columns.index(rank) + 1}</text>'
        )
    edge_routes: list[dict[str, Any]] = []
    for edge in edges:
        src = edge.get("source", "")
        dst = edge.get("target", "")
        if src not in positions_by_node or dst not in positions_by_node:
            continue
        x1, y1 = positions_by_node[src]
        x2, y2 = positions_by_node[dst]
        start = (x1 + node_w, y1 + node_h / 2)
        end = (x2 - 10, y2 + node_h / 2)
        waypoints = None
        route = "forward"
        if x2 <= x1:
            start = (x1 + node_w / 2, y1 + node_h)
            end = (x2 + node_w / 2, y2 - 10)
            if y2 > y1:
                source_group_bottom = y1 + node_h + phase_pad_bottom
                target_group_top = y2 - phase_pad_top
                lane_y = (source_group_bottom + target_group_top) / 2
                lane_y = max(start[1] + 28, min(lane_y, end[1] - 28))
            else:
                lane_y = max(y1 + node_h, y2 + node_h) + phase_pad_bottom + 28
            waypoints = [(start[0], lane_y), (end[0], lane_y)]
            route = "wrap_lane"
        edge_routes.append(
            {
                "source": src,
                "target": dst,
                "route": route,
                "points": [
                    {"x": round(x, 2), "y": round(y, 2)}
                    for x, y in [start, *(waypoints or []), end]
                ],
            }
        )
        parts.append(
            _orthogonal_arrow_svg(
                start=start,
                end=end,
                color=accent,
                width=3,
                label=edge.get("label", ""),
                font=font,
                muted=muted,
                waypoints=waypoints,
            )
        )
    for index, node in enumerate(nodes, start=1):
        x, y = positions_by_node[node]
        is_start = index == 1
        is_end = index == len(nodes)
        parts.append(
            _node_svg(
                label=node,
                x=x,
                y=y,
                width=node_w,
                height=node_h,
                fill="#ffffff" if is_start or is_end else surface,
                stroke=accent if index in {1, len(nodes)} else border,
                text_color=text,
                font=font,
                accent=accent if index % 2 else primary,
                index=index,
                emphasis="start" if is_start else "end" if is_end else "normal",
            )
        )
    parts.append("</svg>")
    node_bounds = {
        node: {
            "x": round(pos[0], 2),
            "y": round(pos[1], 2),
            "w": round(node_w, 2),
            "h": node_h,
            "rank": ranks.get(node, 0),
        }
        for node, pos in positions_by_node.items()
    }
    layout_meta = {
        "node_bounds": node_bounds,
        "edges": [
            {
                "source": edge.get("source", ""),
                "target": edge.get("target", ""),
                "label": edge.get("label", ""),
            }
            for edge in edges
        ],
        "recommended_width": canvas_w,
        "recommended_height": canvas_h,
        "contains_cjk": _contains_cjk_payload([nodes, title, subtitle]),
        "uses_edges": bool(edges),
        "rank_bands": rank_bands,
        "edge_routes": edge_routes,
        "layout": "layered_wrapped_edges" if len(rank_bands) > 1 else "layered_edges",
    }
    return "".join(parts), layout_meta


def render_flowchart_asset_impl(
    *,
    nodes: Any,
    edges: Any = None,
    diagram_kind: str = "pipeline",
    title: str = "",
    subtitle: str = "",
    width: int = 960,
    height: int = 520,
    output_format: str = "svg",
    output_stem: str = "",
    style_overrides: dict[str, Any] | None = None,
    workspace: str | Path | None = None,
    config: MemSlidesConfig | None = None,
) -> dict[str, Any]:
    workspace_path = _workspace_path(workspace)
    normalized_nodes = _normalize_flowchart_nodes(nodes)
    normalized_edges = _normalize_flowchart_edges(edges)
    kind = str(diagram_kind or "pipeline").strip().lower().replace("-", "_")
    if kind not in {"pipeline", "flowchart", "branch_pipeline", "architecture"}:
        raise StructuredVisualError("`diagram_kind` must be one of: pipeline, flowchart, branch_pipeline, architecture.")
    requested_format = str(output_format or "svg").strip().lower()
    if requested_format not in {"svg", "png", "both"}:
        raise StructuredVisualError("`output_format` must be one of: svg, png, both.")
    theme = _flowchart_theme(style_overrides, config=config)
    cache_payload = {
        "kind": "flowchart",
        "diagram_kind": kind,
        "nodes": normalized_nodes,
        "edges": normalized_edges,
        "title": title,
        "subtitle": subtitle,
        "width": int(width),
        "height": int(height),
        "style": theme,
    }
    spec_hash = _json_hash(cache_payload)
    stem = f"flowchart_{_slugify(output_stem or title or kind)}_{spec_hash}"
    target_dir = generated_visuals_dir(workspace_path)
    spec_path = target_dir / f"{stem}.spec.json"
    meta_path = target_dir / f"{stem}.meta.json"
    svg_path = target_dir / f"{stem}.svg"
    png_path = target_dir / f"{stem}.png"
    spec_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if kind == "branch_pipeline" or any(node in normalized_nodes for node in {"W_Q", "W_K", "W_V", "Q", "K", "V"}):
        svg_markup, layout_meta = _build_branch_pipeline_svg(
            nodes=normalized_nodes,
            title=title,
            subtitle=subtitle,
            width=width,
            height=height,
            theme=theme,
        )
    else:
        svg_markup, layout_meta = _build_linear_flowchart_svg(
            nodes=normalized_nodes,
            edges=normalized_edges,
            title=title,
            subtitle=subtitle,
            diagram_kind=kind,
            width=width,
            height=height,
            theme=theme,
        )
    if not svg_path.exists():
        svg_path.write_text(svg_markup, encoding="utf-8")
    warnings: list[str] = []
    if not png_path.exists():
        if vlc is None:
            warnings.append("vl-convert-python is unavailable; PNG output was skipped, SVG is available.")
        else:
            module = _ensure_vl_convert()
            register_visual_font_dirs(config=config, workspace=workspace_path)
            png_path.write_bytes(module.svg_to_png(svg_markup, scale=2))
    rendered_paths: dict[str, str] = {}
    if svg_path.exists():
        rendered_paths["svg"] = str(svg_path.resolve())
    if png_path.exists():
        rendered_paths["png"] = str(png_path.resolve())
    primary_path = rendered_paths.get("png") or rendered_paths.get("svg") or ""
    recommended_width = int(layout_meta.get("recommended_width") or width or 960)
    recommended_height = int(layout_meta.get("recommended_height") or height or 520)
    edge_labels = [
        edge.get("label", "")
        for edge in layout_meta.get("edges", [])
        if isinstance(edge, dict) and edge.get("label")
    ]
    result = {
        "kind": "flowchart",
        "renderer": "deterministic-svg",
        "spec_hash": spec_hash,
        "diagram_kind": kind,
        "title": title or kind.replace("_", " ").title(),
        "nodes": normalized_nodes,
        "edges": [
            (
                f"{edge.get('source', '')} -- {edge.get('label', '')} -> {edge.get('target', '')}"
                if edge.get("label")
                else f"{edge.get('source', '')} -> {edge.get('target', '')}"
            )
            for edge in normalized_edges
        ],
        "edge_labels": edge_labels,
        "svg_path": rendered_paths.get("svg"),
        "png_path": rendered_paths.get("png"),
        "spec_path": str(spec_path.resolve()),
        "meta_path": str(meta_path.resolve()),
        "rendered_paths": rendered_paths,
        "primary_path": primary_path,
        "requested_output_format": requested_format,
        "recommended_width": recommended_width,
        "recommended_height": recommended_height,
        "recommended_aspect_ratio": round(recommended_width / max(1, recommended_height), 4),
        "contains_cjk": bool(layout_meta.get("contains_cjk")),
        "visual_type": "flowchart",
        "preferred_pptx_export": "raster",
        "layout": layout_meta,
        "warnings": warnings,
        "embed_html": (
            f'<img class="flowchart-asset" src="{escape(primary_path)}" alt="{escape(title or kind)}" '
            'style="width:100%;height:100%;object-fit:contain;display:block;" />'
            if primary_path
            else ""
        ),
        "style": {
            "font_family": theme.get("font"),
            "primary_color": theme.get("primary"),
            "accent_color": theme.get("accent"),
        },
        "source": "structured_visual_tool",
        "within_workspace": True,
    }
    meta_payload = {
        "kind": "flowchart",
        "renderer": result["renderer"],
        "spec_hash": spec_hash,
        "title": result["title"],
        "diagram_kind": kind,
        "nodes": normalized_nodes,
        "edges": result["edges"],
        "edge_labels": edge_labels,
        "style": result["style"],
        "rendered_paths": rendered_paths,
        "primary_path": primary_path,
        "requested_output_format": requested_format,
        "recommended_width": recommended_width,
        "recommended_height": recommended_height,
        "recommended_aspect_ratio": result["recommended_aspect_ratio"],
        "contains_cjk": result["contains_cjk"],
        "visual_type": "flowchart",
        "preferred_pptx_export": "raster",
        "layout": layout_meta,
        "warnings": warnings,
        "meta_path": str(meta_path.resolve()),
        "source": "structured_visual_tool",
        "within_workspace": True,
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def render_chart_asset_impl(
    *,
    chart_type: str,
    rows: Any = None,
    csv_text: str = "",
    csv_path: str = "",
    x_field: str,
    y_fields: list[str],
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
    style_overrides: dict[str, Any] | None = None,
    workspace: str | Path | None = None,
    config: MemSlidesConfig | None = None,
) -> dict[str, Any]:
    workspace_path = _workspace_path(workspace)
    source_rows = load_chart_rows(rows=rows, csv_text=csv_text, csv_path=csv_path, workspace=workspace_path)
    theme = resolve_visual_theme(config=config, style_overrides=style_overrides)
    spec = build_chart_spec(
        chart_type=chart_type,
        source_rows=source_rows,
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
        theme=theme,
    )
    requested_format = str(output_format or "svg").strip().lower()
    if requested_format not in {"svg", "png", "both"}:
        raise StructuredVisualError("`output_format` must be one of: svg, png, both.")

    cache_payload = {
        "kind": "chart",
        "chart_type": chart_type,
        "rows": source_rows,
        "x_field": x_field,
        "y_fields": list(y_fields),
        "series_field": series_field,
        "title": title,
        "subtitle": subtitle,
        "x_label": x_label,
        "y_label": y_label,
        "note": note,
        "width": int(width),
        "height": int(height),
        "style": theme,
    }
    spec_hash = _json_hash(cache_payload)
    stem = f"chart_{_slugify(output_stem or title or chart_type)}_{spec_hash}"
    target_dir = generated_visuals_dir(workspace_path)
    spec_path = target_dir / f"{stem}.spec.json"
    meta_path = target_dir / f"{stem}.meta.json"
    svg_path = target_dir / f"{stem}.svg"
    png_path = target_dir / f"{stem}.png"

    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    warnings: list[str] = []

    if vlc is None:
        warnings.append(
            "vl-convert-python is not installed; generated a fallback SVG data visual instead of Vega-rendered outputs."
        )
        if not svg_path.exists():
            svg_markup = _build_chart_fallback_svg_markup(
                chart_type=chart_type,
                source_rows=source_rows,
                x_field=x_field,
                y_fields=y_fields,
                title=title,
                subtitle=subtitle,
                width=width,
                height=height,
                theme=theme,
            )
            svg_path.write_text(svg_markup, encoding="utf-8")
    else:
        module = _ensure_vl_convert()
        register_visual_font_dirs(config=config, workspace=workspace_path)
        if requested_format in {"svg", "both"} and not svg_path.exists():
            svg_markup = module.vegalite_to_svg(spec)
            svg_path.write_text(svg_markup, encoding="utf-8")
        if requested_format in {"png", "both"} and not png_path.exists():
            png_bytes = module.vegalite_to_png(spec, scale=2)
            png_path.write_bytes(png_bytes)

    rendered_paths: dict[str, str] = {}
    if svg_path.exists():
        rendered_paths["svg"] = str(svg_path.resolve())
    if png_path.exists():
        rendered_paths["png"] = str(png_path.resolve())
    primary_path = rendered_paths.get("svg") or rendered_paths.get("png") or ""
    summary = _structured_chart_summary(
        source_rows=source_rows,
        x_field=x_field,
        y_fields=y_fields,
        series_field=series_field,
    )
    result = {
        "kind": "chart",
        "renderer": "vega-lite+vl-convert" if vlc is not None else "vega-lite+fallback-svg",
        "spec_hash": spec_hash,
        "title": title or chart_type.replace("_", " ").title(),
        "svg_path": rendered_paths.get("svg"),
        "png_path": rendered_paths.get("png"),
        "spec_path": str(spec_path.resolve()),
        "meta_path": str(meta_path.resolve()),
        "data_summary": summary,
        "rendered_paths": rendered_paths,
        "primary_path": primary_path,
        "warnings": warnings,
        "embed_html": (
            f'<img src="{escape(primary_path)}" alt="{escape(title or chart_type)}" '
            f'style="width:{int(width)}px;height:auto;display:block;" />'
            if primary_path
            else ""
        ),
        "style": {
            "color_scheme": theme.get("color_scheme"),
            "font_family": theme.get("font_family"),
            "show_grid": theme.get("show_grid"),
            "show_legend": theme.get("show_legend"),
        },
    }
    meta_payload = {
        "kind": "chart",
        "renderer": result["renderer"],
        "spec_hash": spec_hash,
        "title": result["title"],
        "data_fields": {
            "x_field": x_field,
            "y_fields": list(y_fields),
            "series_field": series_field or "",
            "field_names": summary["field_names"],
        },
        "style": result["style"],
        "rendered_paths": rendered_paths,
        "primary_path": primary_path,
        "warnings": warnings,
        "spec_path": str(spec_path.resolve()),
        "meta_path": str(meta_path.resolve()),
        "source": "structured_visual_tool",
        "within_workspace": True,
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _is_numeric_column(rows: list[dict[str, Any]], column: str) -> bool:
    seen = False
    for row in rows:
        value = str(row.get(column, "") or "").strip().replace(",", "")
        if not value:
            continue
        seen = True
        if not re.fullmatch(r"-?\d+(?:\.\d+)?%?", value):
            return False
    return seen


def _table_style_css(style: str, theme: dict[str, Any]) -> tuple[str, str, str]:
    style_name = str(style or "three_line").strip().lower()
    header_rule = ""
    row_rule = ""
    table_rule = (
        f"width:100%;border-collapse:separate;border-spacing:0;color:{theme['text_color']};"
        f"font-family:{theme['font_family']};font-size:{theme['body_size']}px;"
        f"background:{theme['surface_color']};border:1px solid {theme['border_color']};"
        "border-radius:10px;overflow:hidden;"
    )
    if style_name == "simple_grid":
        header_rule = (
            f"border-bottom:1px solid {theme['border_color']};padding:12px 14px;"
            f"font-weight:700;background:{theme['background_color']};"
        )
        row_rule = f"border-bottom:1px solid {theme['border_color']};padding:12px 14px;"
    elif style_name == "minimal":
        header_rule = (
            f"border-bottom:1.5px solid {theme['border_color']};padding:12px 14px;font-weight:700;"
        )
        row_rule = f"padding:12px 14px;border-bottom:1px solid {theme['border_color']};"
    else:
        header_rule = (
            f"border-bottom:1.5px solid {theme['accent_color']};padding:12px 14px;"
            f"font-weight:800;background:{theme['background_color']};"
        )
        row_rule = f"padding:12px 14px;border-bottom:1px solid {theme['border_color']};"
    return style_name, table_rule, header_rule or row_rule


def _table_layout_model(
    *,
    columns: list[str],
    rows: list[dict[str, Any]],
    caption: str = "",
    footnote: str = "",
    width: int,
    theme: dict[str, Any],
) -> dict[str, Any]:
    body_size = max(12, int(theme.get("body_size") or 16))
    caption_size = max(11, int(theme.get("caption_size") or 13))
    table_width = max(520, int(width or 960))
    numeric_columns = {column for column in columns if _is_numeric_column(rows, column)}
    column_widths = _estimate_column_widths(columns, rows, table_width - 56, body_size)
    padding_x = 16
    line_gap = max(17, body_size + 4)

    header_lines: dict[str, list[str]] = {}
    wrapped_rows: list[dict[str, list[str]]] = []
    for index, column in enumerate(columns):
        available_units = max(6.0, (column_widths[index] - padding_x * 2) / (body_size * 0.52))
        header_lines[column] = _wrap_display_text(column, available_units, max_lines=2)
    for row in rows:
        shaped: dict[str, list[str]] = {}
        for index, column in enumerate(columns):
            available_units = max(6.0, (column_widths[index] - padding_x * 2) / (body_size * 0.52))
            shaped[column] = _wrap_display_text(row.get(column, ""), available_units, max_lines=3)
        wrapped_rows.append(shaped)

    header_height = max(48, 22 + max((len(header_lines[column]) for column in columns), default=1) * line_gap)
    row_heights: list[int] = []
    for shaped in wrapped_rows:
        line_count = max((len(shaped[column]) for column in columns), default=1)
        row_heights.append(max(48, 20 + line_count * line_gap))
    title_line_count = len(_wrap_display_text(caption, max(30, table_width / 14), max_lines=2)) if caption else 0
    note_line_count = len(_wrap_display_text(footnote, max(36, table_width / 12), max_lines=2)) if footnote else 0
    title_block_height = 34 + title_line_count * (caption_size + 11) + note_line_count * (caption_size + 5)
    title_block_height = max(58, title_block_height)
    footnote_height = 34
    total_height = title_block_height + header_height + sum(row_heights) + footnote_height + 28
    return {
        "table_width": table_width,
        "inner_x": 28,
        "column_widths": column_widths,
        "numeric_columns": numeric_columns,
        "header_lines": header_lines,
        "wrapped_rows": wrapped_rows,
        "row_heights": row_heights,
        "header_height": header_height,
        "line_gap": line_gap,
        "body_size": body_size,
        "caption_size": caption_size,
        "title_block_height": title_block_height,
        "total_height": total_height,
        "contains_cjk": _contains_cjk_payload([columns, rows]),
        "density": "presentation",
    }


def _build_table_html_snippet(
    *,
    columns: list[str],
    rows: list[dict[str, Any]],
    style: str,
    caption: str,
    footnote: str,
    width: int,
    theme: dict[str, Any],
) -> str:
    style_name, table_rule, header_rule_or_row_rule = _table_style_css(style, theme)
    if style_name == "simple_grid":
        row_rule = f"border-bottom:1px solid {theme['border_color']};padding:12px 14px;"
        header_rule = header_rule_or_row_rule
    elif style_name == "minimal":
        row_rule = f"padding:12px 14px;border-bottom:1px solid {theme['border_color']};"
        header_rule = header_rule_or_row_rule
    else:
        row_rule = f"padding:12px 14px;border-bottom:1px solid {theme['border_color']};"
        header_rule = header_rule_or_row_rule
    numeric_columns = {column for column in columns if _is_numeric_column(rows, column)}
    parts = [
        (
            f'<div class="generated-table" style="width:{int(width)}px;margin:0 auto;'
            f'font-family:{theme["font_family"]};color:{theme["text_color"]};'
            f'background:{theme["background_color"]};padding:20px 22px;'
            f'border-left:6px solid {theme["accent_color"]};border-radius:12px;'
            'box-sizing:border-box;">'
        )
    ]
    if caption:
        parts.append(
            f'<div style="font-family:{theme["title_font"]};font-size:{theme["caption_size"] + 5}px;'
            f'font-weight:800;color:{theme["text_color"]};margin-bottom:4px;line-height:1.18;">'
            f'{escape(caption)}</div>'
        )
    if footnote:
        parts.append(
            f'<div style="font-family:{theme["caption_font"]};font-size:{theme["caption_size"]}px;'
            f'color:{theme["muted_text_color"]};margin-bottom:12px;line-height:1.32;">{escape(footnote)}</div>'
        )
    parts.append(f'<table style="{table_rule}">')
    parts.append("<thead><tr>")
    for column in columns:
        align = "right" if column in numeric_columns else "left"
        parts.append(
            f'<th style="{header_rule}text-align:{align};vertical-align:bottom;'
            'line-height:1.25;white-space:normal;">'
            f'{escape(column)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    for row_index, row in enumerate(rows):
        row_bg = theme["background_color"] if row_index % 2 else theme["surface_color"]
        parts.append("<tr>")
        for column in columns:
            align = "right" if column in numeric_columns else "left"
            value = _empty_cell_placeholder(row.get(column, ""))
            parts.append(
                f'<td style="{row_rule}text-align:{align};vertical-align:top;line-height:1.34;'
                f'background:{row_bg};word-break:break-word;">{escape(value)}</td>'
            )
        parts.append("</tr>")
    parts.append("</tbody></table>")
    parts.append("</div>")
    return "".join(parts)


def _estimate_column_widths(
    columns: list[str],
    rows: list[dict[str, Any]],
    width: int,
    font_size: int,
) -> list[float]:
    weights: list[float] = []
    for column in columns:
        values = [column, *[_empty_cell_placeholder(row.get(column, "")) for row in rows]]
        max_units = max((_display_width(value) for value in values), default=4.0)
        base_weight = math.sqrt(max(4.0, max_units)) * 4.4
        if _is_numeric_column(rows, column):
            base_weight = min(base_weight, 12.0)
        weights.append(max(6.0, min(26.0, base_weight)))
    total_weight = sum(weights) or len(columns) or 1
    raw = [width * (weight / total_weight) for weight in weights]
    min_width = max(76.0, font_size * 5.2)
    if sum(max(value, min_width) for value in raw) <= width:
        return [max(value, min_width) for value in raw]
    return raw


def _build_table_svg_markup(
    *,
    columns: list[str],
    rows: list[dict[str, Any]],
    style: str,
    caption: str,
    footnote: str,
    width: int,
    theme: dict[str, Any],
) -> tuple[str, int, dict[str, Any]]:
    layout = _table_layout_model(
        columns=columns,
        rows=rows,
        caption=caption,
        footnote=footnote,
        width=width,
        theme=theme,
    )
    table_width = int(layout["table_width"])
    total_height = int(layout["total_height"])
    inner_x = float(layout["inner_x"])
    col_widths = list(layout["column_widths"])
    numeric_columns = set(layout["numeric_columns"])
    header_lines = dict(layout["header_lines"])
    wrapped_rows = list(layout["wrapped_rows"])
    row_heights = list(layout["row_heights"])
    header_height = int(layout["header_height"])
    line_gap = float(layout["line_gap"])
    body_size = int(layout["body_size"])
    caption_size = int(layout["caption_size"])
    top_y = int(layout["title_block_height"]) + 12
    y = top_y
    style_name = str(style or "three_line").strip().lower()
    accent = str(theme["accent_color"])
    text_color = str(theme["text_color"])
    muted = str(theme["muted_text_color"])
    border = str(theme["border_color"])
    background = str(theme["background_color"])
    surface = str(theme["surface_color"])
    header_fill = "#f8fafc" if background.lower() == "#ffffff" else background
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{table_width}" height="{total_height}" '
            f'viewBox="0 0 {table_width} {total_height}" role="img" data-visual-kind="table" '
            f'data-preferred-pptx-export="raster" data-contains-cjk="{str(layout["contains_cjk"]).lower()}">'
        ),
        f'<rect width="{table_width}" height="{total_height}" rx="14" fill="{escape(background)}" />',
        f'<rect x="0" y="0" width="7" height="{total_height}" fill="{escape(accent)}" />',
        f'<style>text,tspan{{font-family:{escape(CJK_FONT_STACK)};}}</style>',
    ]
    if caption:
        title_lines = _wrap_display_text(caption, max(30, table_width / 14), max_lines=2)
        parts.append(
            _svg_text_block(
                lines=title_lines,
                x=inner_x,
                y=28,
                font=str(theme["title_font"]),
                font_size=caption_size + 7,
                fill=text_color,
                line_gap=caption_size + 11,
                weight="800",
            )
        )
    if footnote:
        title_lines_for_offset = _wrap_display_text(caption, max(30, table_width / 14), max_lines=2) if caption else []
        note_y = 34 + len(title_lines_for_offset) * (caption_size + 11)
        note_lines = _wrap_display_text(footnote, max(36, table_width / 12), max_lines=2)
        parts.append(
            _svg_text_block(
                lines=note_lines,
                x=inner_x,
                y=note_y,
                font=str(theme["caption_font"]),
                font_size=caption_size,
                fill=muted,
                line_gap=caption_size + 5,
                weight="500",
            )
        )
    line_color = accent if style_name == "three_line" else border
    table_x = inner_x
    table_right = table_x + sum(col_widths)
    parts.append(
        f'<rect x="{table_x:.1f}" y="{y:.1f}" width="{sum(col_widths):.1f}" height="{header_height}" '
        f'rx="10" fill="{escape(header_fill)}" stroke="{escape(border)}" stroke-width="1" />'
    )
    parts.append(
        f'<line x1="{table_x:.1f}" y1="{y + header_height:.1f}" x2="{table_right:.1f}" '
        f'y2="{y + header_height:.1f}" stroke="{escape(line_color)}" stroke-width="2" />'
    )
    x = table_x
    for index, column in enumerate(columns):
        col_w = col_widths[index]
        text_x = x + 16 if column not in numeric_columns else x + col_w - 16
        anchor = "start" if column not in numeric_columns else "end"
        parts.append(
            _svg_text_block(
                lines=header_lines[column],
                x=text_x,
                y=y + 25,
                font=str(theme["font_family"]),
                font_size=body_size,
                fill=text_color,
                line_gap=line_gap,
                anchor=anchor,
                weight="800",
            )
        )
        if style_name == "simple_grid":
            parts.append(
                f'<line x1="{x + col_w:.1f}" y1="{y:.1f}" x2="{x + col_w:.1f}" y2="{y + header_height:.1f}" '
                f'stroke="{escape(border)}" stroke-width="1" />'
            )
        x += col_w
    y += header_height
    for row_index, shaped in enumerate(wrapped_rows):
        row_height = row_heights[row_index]
        fill = surface if row_index % 2 == 0 else background
        parts.append(
            f'<rect x="{table_x:.1f}" y="{y:.1f}" width="{sum(col_widths):.1f}" height="{row_height}" '
            f'fill="{escape(fill)}" stroke="{escape(border)}" stroke-width="1" opacity="0.98" />'
        )
        x = table_x
        for index, column in enumerate(columns):
            col_w = col_widths[index]
            is_numeric = column in numeric_columns
            text_x = x + 16 if not is_numeric else x + col_w - 16
            anchor = "start" if not is_numeric else "end"
            parts.append(
                _svg_text_block(
                    lines=shaped[column],
                    x=text_x,
                    y=y + 25,
                    font=str(theme["font_family"]),
                    font_size=body_size - 1,
                    fill=text_color,
                    line_gap=line_gap,
                    anchor=anchor,
                    weight="600" if is_numeric else "500",
                    extra='font-variant-numeric="tabular-nums"',
                )
            )
            if style_name == "simple_grid":
                parts.append(
                    f'<line x1="{x + col_w:.1f}" y1="{y:.1f}" x2="{x + col_w:.1f}" y2="{y + row_height:.1f}" '
                    f'stroke="{escape(border)}" stroke-width="1" />'
                )
            x += col_w
        if style_name == "minimal":
            parts.append(
                f'<line x1="{table_x:.1f}" y1="{y + row_height:.1f}" x2="{table_right:.1f}" y2="{y + row_height:.1f}" '
                f'stroke="{escape(border)}" stroke-width="1" />'
            )
        y += row_height
    parts.append(
        f'<line x1="{table_x:.1f}" y1="{y:.1f}" x2="{table_right:.1f}" y2="{y:.1f}" '
        f'stroke="{escape(line_color)}" stroke-width="2" />'
    )
    if footnote:
        parts.append(
            f'<text x="{table_right:.1f}" y="{total_height - 15}" text-anchor="end" '
            f'font-family="{escape(theme["caption_font"])}" font-size="{caption_size - 1}" '
            f'fill="{escape(muted)}">{escape(str(len(rows)))} rows</text>'
        )
    parts.append("</svg>")
    layout_meta = {
        "column_widths": [round(value, 2) for value in col_widths],
        "row_heights": row_heights,
        "header_height": header_height,
        "wrapped_cells": {
            column: max((len(row[column]) for row in wrapped_rows), default=1)
            for column in columns
        },
        "contains_cjk": bool(layout["contains_cjk"]),
        "density": layout["density"],
    }
    return "".join(parts), total_height, layout_meta


def render_table_asset_impl(
    *,
    rows: Any = None,
    columns: list[str] | None = None,
    csv_text: str = "",
    csv_path: str = "",
    markdown_table: str = "",
    style: str = "three_line",
    caption: str = "",
    footnote: str = "",
    output_mode: str = "html",
    width: int = 960,
    output_stem: str = "",
    style_overrides: dict[str, Any] | None = None,
    workspace: str | Path | None = None,
    config: MemSlidesConfig | None = None,
) -> dict[str, Any]:
    workspace_path = _workspace_path(workspace)
    column_names, table_rows = load_table_rows(
        rows=rows,
        columns=columns,
        csv_text=csv_text,
        csv_path=csv_path,
        markdown_table=markdown_table,
        workspace=workspace_path,
    )
    theme = resolve_visual_theme(config=config, style_overrides=style_overrides)
    mode = str(output_mode or "html").strip().lower()
    if mode not in {"html", "svg", "png", "both"}:
        raise StructuredVisualError("`output_mode` must be one of: html, svg, png, both.")

    cache_payload = {
        "kind": "table",
        "columns": column_names,
        "rows": table_rows,
        "style": style,
        "caption": caption,
        "footnote": footnote,
        "width": int(width),
        "theme": theme,
    }
    spec_hash = _json_hash(cache_payload)
    stem = f"table_{_slugify(output_stem or caption or 'table')}_{spec_hash}"
    target_dir = generated_visuals_dir(workspace_path)
    fragment_path = target_dir / f"{stem}.fragment.html"
    svg_path = target_dir / f"{stem}.svg"
    png_path = target_dir / f"{stem}.png"
    meta_path = target_dir / f"{stem}.meta.json"

    html_snippet = _build_table_html_snippet(
        columns=column_names,
        rows=table_rows,
        style=style,
        caption=caption,
        footnote=footnote,
        width=width,
        theme=theme,
    )
    svg_markup, svg_height, layout_meta = _build_table_svg_markup(
        columns=column_names,
        rows=table_rows,
        style=style,
        caption=caption,
        footnote=footnote,
        width=width,
        theme=theme,
    )

    warnings: list[str] = []
    needs_png_fallback = vlc is None
    if needs_png_fallback:
        warnings.append(
            "vl-convert-python is not installed; generated HTML/SVG table outputs instead of PNG."
        )

    if mode in {"html", "both"} or needs_png_fallback:
        fragment_path.write_text(html_snippet, encoding="utf-8")
    if not svg_path.exists():
        svg_path.write_text(svg_markup, encoding="utf-8")
    if vlc is not None and not png_path.exists():
        module = _ensure_vl_convert()
        register_visual_font_dirs(config=config, workspace=workspace_path)
        png_path.write_bytes(module.svg_to_png(svg_markup, scale=2))

    rendered_paths: dict[str, str] = {}
    if fragment_path.exists():
        rendered_paths["html"] = str(fragment_path.resolve())
    if svg_path.exists():
        rendered_paths["svg"] = str(svg_path.resolve())
    if png_path.exists():
        rendered_paths["png"] = str(png_path.resolve())
    primary_path = (
        rendered_paths.get("png")
        or rendered_paths.get("svg")
        or rendered_paths.get("html")
        or ""
    )
    recommended_width = int(width)
    recommended_height = int(svg_height)

    result = {
        "kind": "table",
        "renderer": "html+svg+png" if rendered_paths.get("png") else "html+svg",
        "spec_hash": spec_hash,
        "title": caption or "Table",
        "html_snippet": html_snippet if fragment_path.exists() else "",
        "fragment_path": rendered_paths.get("html"),
        "svg_path": rendered_paths.get("svg"),
        "png_path": rendered_paths.get("png"),
        "meta_path": str(meta_path.resolve()),
        "rendered_paths": rendered_paths,
        "primary_path": primary_path,
        "requested_output_mode": mode,
        "warnings": warnings,
        "svg_height": svg_height,
        "recommended_width": recommended_width,
        "recommended_height": recommended_height,
        "recommended_aspect_ratio": round(recommended_width / max(1, recommended_height), 4),
        "contains_cjk": bool(layout_meta.get("contains_cjk")),
        "visual_type": "table",
        "preferred_pptx_export": "raster",
        "layout": layout_meta,
        "data_summary": {
            "row_count": len(table_rows),
            "column_count": len(column_names),
            "columns": column_names,
            "numeric_columns": [column for column in column_names if _is_numeric_column(table_rows, column)],
            "preview": table_rows[:5],
        },
        "embed_html": (
            f'<img src="{escape(primary_path)}" '
            f'alt="{escape(caption or "table")}" style="width:{int(width)}px;height:auto;display:block;" />'
        ),
        "style": {
            "table_style": str(style or "three_line"),
            "font_family": theme.get("font_family"),
            "text_color": theme.get("text_color"),
            "border_color": theme.get("border_color"),
            "accent_color": theme.get("accent_color"),
        },
    }
    meta_payload = {
        "kind": "table",
        "renderer": result["renderer"],
        "spec_hash": spec_hash,
        "title": caption or "Table",
        "data_fields": {
            "columns": column_names,
            "row_count": len(table_rows),
        },
        "style": result["style"],
        "rendered_paths": rendered_paths,
        "primary_path": primary_path,
        "requested_output_mode": mode,
        "recommended_width": recommended_width,
        "recommended_height": recommended_height,
        "recommended_aspect_ratio": result["recommended_aspect_ratio"],
        "contains_cjk": result["contains_cjk"],
        "visual_type": "table",
        "preferred_pptx_export": "raster",
        "layout": layout_meta,
        "warnings": warnings,
        "meta_path": str(meta_path.resolve()),
        "source": "structured_visual_tool",
        "within_workspace": True,
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def collect_generated_visual_entries(workspace: str | Path | None = None) -> list[dict[str, Any]]:
    workspace_path = _workspace_path(workspace)
    target_dir = workspace_path / "generated_visuals"
    if not target_dir.exists():
        return []

    collected: list[dict[str, Any]] = []
    for meta_path in sorted(target_dir.glob("*.meta.json")):
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warning("Failed to parse generated visual metadata %s: %s", meta_path, exc)
            continue
        if not isinstance(payload, dict):
            continue
        rendered_paths = payload.get("rendered_paths", {})
        if not isinstance(rendered_paths, dict):
            rendered_paths = {}
        preferred = (
            payload.get("primary_path")
            or rendered_paths.get("html")
            or rendered_paths.get("svg")
            or rendered_paths.get("png")
            or ""
        )
        if not preferred:
            continue
        preferred_path = Path(str(preferred)).expanduser()
        if not preferred_path.is_absolute():
            preferred_path = workspace_path / preferred_path
        try:
            resolved = preferred_path.resolve()
        except Exception:
            resolved = preferred_path
        within_workspace = resolved == workspace_path or workspace_path in resolved.parents
        collected.append(
            {
                "path": str(resolved),
                "kind": str(payload.get("kind", "") or "figure"),
                "caption": str(payload.get("title", "") or ""),
                "exists": resolved.exists(),
                "within_workspace": within_workspace,
                "generated_by_tool": True,
                "renderer": str(payload.get("renderer", "") or ""),
                "meta_path": str(meta_path.resolve()),
                "rendered_paths": rendered_paths,
                "recommended_width": payload.get("recommended_width"),
                "recommended_height": payload.get("recommended_height"),
                "recommended_aspect_ratio": payload.get("recommended_aspect_ratio"),
                "contains_cjk": bool(payload.get("contains_cjk")),
                "preferred_pptx_export": str(payload.get("preferred_pptx_export", "") or ""),
                "visual_type": str(payload.get("visual_type", "") or ""),
            }
        )
    return collected
