from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


PROFILE_HOLISTIC_V6_VISUAL_DIMENSION_FILENAME = "profile_holistic_v6_visual_dimension_records.json"
PROFILE_HOLISTIC_V6_VISUAL_OVERALL_FILENAME = "profile_holistic_v6_visual_overall_records.json"
PROFILE_HOLISTIC_V6_SUMMARY_FILENAME = "profile_holistic_v6_probe_summary.json"
PROFILE_HOLISTIC_V6_REPORT_FILENAME = "profile_holistic_v6_report.md"
PROFILE_HOLISTIC_V6_SCOPE = "profile_only_visual_full_deck"


@dataclass(frozen=True)
class ProfileHolisticV6DimensionSpec:
    dimension_id: str
    label: str
    focus: str


PROFILE_HOLISTIC_V6_DIMENSIONS: tuple[ProfileHolisticV6DimensionSpec, ...] = (
    ProfileHolisticV6DimensionSpec(
        "role_decision_fit",
        "Role-Decision Fit",
        "Whether the deck is organized around the target persona's role, responsibility, and decision frame.",
    ),
    ProfileHolisticV6DimensionSpec(
        "narrative_priority",
        "Narrative Priority",
        "Whether the deck prioritizes and sequences information in the way the target persona would naturally use it.",
    ),
    ProfileHolisticV6DimensionSpec(
        "visual_manifestation",
        "Visual Manifestation",
        "Whether layout, hierarchy, whitespace, diagrams, and card/table choices visibly reflect persona preferences.",
    ),
    ProfileHolisticV6DimensionSpec(
        "action_support",
        "Action Support",
        "Whether the deck helps the target persona reach a judgment, restate the conclusion, and act on it.",
    ),
)

_DIMENSION_LABEL_FALLBACKS = {
    "role_decision_fit": "角色决策取向",
    "narrative_priority": "叙事与重点组织",
    "visual_manifestation": "视觉落地与版式体现",
    "action_support": "决策支持与行动导向",
}
_POSITIVE_EFFECTS = {"clear_positive", "weak_positive"}


def rebuild_profile_holistic_v6_summary(
    eval_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    llm_ref: str | None = None,
) -> dict[str, Any]:
    """Rebuild the V6 main-table summary from persisted visual judge records."""
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir) if output_dir else eval_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    visual_dimension_records = _load_record_list(eval_dir / PROFILE_HOLISTIC_V6_VISUAL_DIMENSION_FILENAME)
    visual_overall_records = _load_record_list(eval_dir / PROFILE_HOLISTIC_V6_VISUAL_OVERALL_FILENAME)
    source_summary = _safe_load_json(eval_dir / PROFILE_HOLISTIC_V6_SUMMARY_FILENAME)
    skipped_probes = _skipped_probes(source_summary)

    probe_rows = build_profile_holistic_v6_probe_rows(visual_dimension_records, visual_overall_records)
    dimension_rows = build_profile_holistic_v6_dimension_rows(visual_dimension_records)
    overall = build_profile_holistic_v6_overall_row(
        probe_rows=probe_rows,
        visual_dimension_records=visual_dimension_records,
        visual_overall_records=visual_overall_records,
        skipped_probes=skipped_probes,
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(),
        "llm_ref": llm_ref or str(source_summary.get("llm_ref") or "unknown"),
        "evaluation_scope": PROFILE_HOLISTIC_V6_SCOPE,
        "source_eval_dir": str(eval_dir),
        "probe_rows": probe_rows,
        "dimension_rows": dimension_rows,
        "overall": overall,
    }

    summary_path = output_dir / PROFILE_HOLISTIC_V6_SUMMARY_FILENAME
    report_path = output_dir / PROFILE_HOLISTIC_V6_REPORT_FILENAME
    _write_json(summary_path, payload)
    report_path.write_text(
        build_profile_holistic_v6_report(
            source_eval_dir=eval_dir,
            llm_ref=str(payload["llm_ref"]),
            probe_rows=probe_rows,
            dimension_rows=dimension_rows,
            overall=overall,
            visual_dimension_records=visual_dimension_records,
        ),
        encoding="utf-8",
    )

    return {
        "source_eval_dir": str(eval_dir),
        "output_dir": str(output_dir),
        "probe_count": len(probe_rows),
        "dimension_count": len(dimension_rows),
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
        },
        "overall": overall,
    }


def validate_profile_holistic_v6_summary(eval_dir: Path | str) -> dict[str, Any]:
    """Validate that the persisted V6 summary matches the detailed judge records."""
    eval_dir = Path(eval_dir)
    existing = _safe_load_json(eval_dir / PROFILE_HOLISTIC_V6_SUMMARY_FILENAME)
    if not existing:
        return {
            "eval_dir": str(eval_dir),
            "ok": False,
            "issues": [f"Missing or unreadable {PROFILE_HOLISTIC_V6_SUMMARY_FILENAME}."],
        }

    dimension_path = eval_dir / PROFILE_HOLISTIC_V6_VISUAL_DIMENSION_FILENAME
    overall_path = eval_dir / PROFILE_HOLISTIC_V6_VISUAL_OVERALL_FILENAME
    if not _path_readable(dimension_path) or not _path_readable(overall_path):
        return _validate_summary_shape(eval_dir, existing)

    visual_dimension_records = _load_record_list(dimension_path)
    visual_overall_records = _load_record_list(overall_path)

    rebuilt_probe_rows = build_profile_holistic_v6_probe_rows(visual_dimension_records, visual_overall_records)
    rebuilt_dimension_rows = build_profile_holistic_v6_dimension_rows(visual_dimension_records)
    rebuilt_overall = build_profile_holistic_v6_overall_row(
        probe_rows=rebuilt_probe_rows,
        visual_dimension_records=visual_dimension_records,
        visual_overall_records=visual_overall_records,
        skipped_probes=_skipped_probes(existing),
    )

    issues: list[str] = []
    _compare_probe_rows(existing.get("probe_rows"), rebuilt_probe_rows, issues)
    _compare_dimension_rows(existing.get("dimension_rows"), rebuilt_dimension_rows, issues)
    _compare_overall(existing.get("overall"), rebuilt_overall, issues)
    return {
        "eval_dir": str(eval_dir),
        "ok": not issues,
        "issues": issues,
        "existing_probe_count": len(existing.get("probe_rows") or []),
        "rebuilt_probe_count": len(rebuilt_probe_rows),
        "existing_dimension_count": len(existing.get("dimension_rows") or []),
        "rebuilt_dimension_count": len(rebuilt_dimension_rows),
    }


def build_profile_holistic_v6_probe_rows(
    visual_dimension_records: list[dict[str, Any]],
    visual_overall_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_probe_uid = {
        str(item.get("probe_uid") or ""): item
        for item in visual_overall_records
        if isinstance(item, dict) and item.get("probe_uid")
    }
    for probe_uid in sorted(by_probe_uid):
        overall = by_probe_uid[probe_uid]
        dimensions = [
            item
            for item in visual_dimension_records
            if isinstance(item, dict) and str(item.get("probe_uid") or "") == probe_uid
        ]
        primary_win_dims = len([item for item in dimensions if item.get("winner") == "primary_better"])
        vote_consistency_rate = _coerce_float(overall.get("vote_consistency_rate")) or 0.0
        strict_positive = all(
            [
                overall.get("winner") == "primary_better",
                overall.get("profile_effect") in _POSITIVE_EFFECTS,
                bool(overall.get("order_consistent")),
                vote_consistency_rate >= (2 / 3),
            ]
        )
        rows.append(
            {
                "persona": str(overall.get("persona") or ""),
                "probe_id": str(overall.get("probe_id") or ""),
                "probe_uid": probe_uid,
                "overall_lift": _coerce_float(overall.get("score_lift")),
                "primary_win_dims": primary_win_dims,
                "overall_vote_consistency_rate": _coerce_float(overall.get("vote_consistency_rate")),
                "overall_order_consistent": bool(overall.get("order_consistent")),
                "strict_positive": strict_positive,
                "overall_winner": str(overall.get("winner") or ""),
                "overall_profile_effect": str(overall.get("profile_effect") or ""),
            }
        )
    return rows


def build_profile_holistic_v6_dimension_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PROFILE_HOLISTIC_V6_DIMENSIONS:
        items = [item for item in records if isinstance(item, dict) and item.get("dimension_id") == spec.dimension_id]
        if not items:
            continue
        primary_wins = [item for item in items if item.get("winner") == "primary_better"]
        rows.append(
            {
                "dimension_id": spec.dimension_id,
                "dimension_label": _dimension_label(spec, items),
                "dimension_focus": spec.focus,
                "probe_count": len(items),
                "primary_avg_score": _mean(_coerce_float(item.get("primary_score")) for item in items),
                "control_avg_score": _mean(_coerce_float(item.get("control_score")) for item in items),
                "mean_score_lift": _mean(_coerce_float(item.get("score_lift")) for item in items),
                "primary_win_rate": _safe_divide(len(primary_wins), len(items)),
            }
        )
    return rows


def build_profile_holistic_v6_overall_row(
    *,
    probe_rows: list[dict[str, Any]],
    visual_dimension_records: list[dict[str, Any]],
    visual_overall_records: list[dict[str, Any]],
    skipped_probes: list[dict[str, Any]],
) -> dict[str, Any]:
    strict_positive = [item for item in probe_rows if item.get("strict_positive")]
    dimension_lifts: dict[str, float | None] = {}
    for spec in PROFILE_HOLISTIC_V6_DIMENSIONS:
        matched = [
            _coerce_float(item.get("score_lift"))
            for item in visual_dimension_records
            if isinstance(item, dict) and item.get("dimension_id") == spec.dimension_id
        ]
        dimension_lifts[f"mean_{spec.dimension_id}_lift"] = _mean(matched)
    return {
        "paired_probe_count": len(probe_rows),
        "skipped_probe_count": len(skipped_probes),
        "skipped_probes": skipped_probes,
        "strict_positive_persona_rate": _safe_divide(len(strict_positive), len(probe_rows)),
        "mean_overall_lift": _mean(_coerce_float(item.get("score_lift")) for item in visual_overall_records),
        **dimension_lifts,
    }


def build_profile_holistic_v6_report(
    *,
    source_eval_dir: Path,
    llm_ref: str,
    probe_rows: list[dict[str, Any]],
    dimension_rows: list[dict[str, Any]],
    overall: dict[str, Any],
    visual_dimension_records: list[dict[str, Any]],
) -> str:
    summary_columns = [
        ("paired_probes", str(overall.get("paired_probe_count", 0))),
        ("strict_positive_persona_rate", _format_metric(overall.get("strict_positive_persona_rate"))),
        ("mean_overall_lift", _format_metric(overall.get("mean_overall_lift"))),
        *[
            (
                f"mean_{spec.dimension_id}_lift",
                _format_metric(overall.get(f"mean_{spec.dimension_id}_lift")),
            )
            for spec in PROFILE_HOLISTIC_V6_DIMENSIONS
        ],
        ("skipped_probes", str(overall.get("skipped_probe_count", 0))),
    ]
    lines = [
        "# Profile Memory Holistic Evaluation V6",
        "",
        f"Generated at: `{datetime.now().isoformat()}`",
        f"Judge model ref: `{llm_ref}`",
        f"Evaluation scope: `{PROFILE_HOLISTIC_V6_SCOPE}`",
        f"Source: `{source_eval_dir}`",
        "",
        "## Method",
        "",
        "- Profile-only full-deck visual blind judge.",
        "- Judge sees persona overview plus page-aligned comparison images covering the full round-0 deck.",
        "- Judge does not receive the original prompt, task intent, or textual deck summary.",
        "- Four fixed dimensions: role-decision fit, narrative priority, visual manifestation, and action support.",
        "- Each comparison is judged three times with order swapping and rubric paraphrasing.",
        "- A persona is strict positive only when the profile-conditioned deck wins overall, profile effect is positive, order-swapped voting is consistent, and vote consistency is at least 2/3.",
        "",
        "## Overall Summary",
        "",
        "| " + " | ".join(key for key, _ in summary_columns) + " |",
        "| " + " | ".join("---:" for _ in summary_columns) + " |",
        "| " + " | ".join(value for _, value in summary_columns) + " |",
        "",
        "## Persona-Level Summary",
        "",
        "| persona | probe_id | overall_lift | primary_win_dims | vote_consistency | order_consistent | strict_positive |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in probe_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("persona")),
                    _md(row.get("probe_id")),
                    _format_metric(row.get("overall_lift")),
                    str(row.get("primary_win_dims", "")),
                    _format_metric(row.get("overall_vote_consistency_rate")),
                    str(row.get("overall_order_consistent")),
                    str(row.get("strict_positive")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Dimension Summary",
            "",
            "| dimension | primary_avg | control_avg | lift | primary_win_rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dimension_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("dimension_label")),
                    _format_metric(row.get("primary_avg_score")),
                    _format_metric(row.get("control_avg_score")),
                    _format_metric(row.get("mean_score_lift")),
                    _format_metric(row.get("primary_win_rate")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Detailed Dimension Rows",
            "",
            "| persona | probe_id | dimension | primary_score | control_score | lift | winner | reason |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for item in visual_dimension_records:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(item.get("persona")),
                    _md(item.get("probe_id")),
                    _md(item.get("dimension_label") or item.get("dimension_id")),
                    _format_metric(item.get("primary_score")),
                    _format_metric(item.get("control_score")),
                    _format_metric(item.get("score_lift")),
                    _md(item.get("winner")),
                    _md(item.get("reason")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _validate_summary_shape(eval_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    dimension_ids = {
        row.get("dimension_id")
        for row in payload.get("dimension_rows", [])
        if isinstance(row, dict)
    }
    for spec in PROFILE_HOLISTIC_V6_DIMENSIONS:
        if spec.dimension_id not in dimension_ids:
            issues.append(f"Summary missing dimension `{spec.dimension_id}`.")
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    for key in ("paired_probe_count", "strict_positive_persona_rate", "mean_overall_lift"):
        if key not in overall:
            issues.append(f"Summary overall missing `{key}`.")
    return {
        "eval_dir": str(eval_dir),
        "ok": not issues,
        "issues": issues,
        "existing_probe_count": len(payload.get("probe_rows") or []),
        "existing_dimension_count": len(payload.get("dimension_rows") or []),
        "detail_records_available": False,
    }


def _compare_probe_rows(existing: Any, rebuilt: list[dict[str, Any]], issues: list[str]) -> None:
    existing_rows = existing if isinstance(existing, list) else []
    if len(existing_rows) != len(rebuilt):
        issues.append(f"Probe row count mismatch: existing={len(existing_rows)} rebuilt={len(rebuilt)}.")
        return
    existing_by_uid = {str(item.get("probe_uid") or ""): item for item in existing_rows if isinstance(item, dict)}
    for row in rebuilt:
        probe_uid = str(row.get("probe_uid") or "")
        other = existing_by_uid.get(probe_uid)
        if other is None:
            issues.append(f"Missing existing probe row `{probe_uid}`.")
            continue
        for key in ("overall_lift", "primary_win_dims", "overall_vote_consistency_rate", "overall_order_consistent", "strict_positive"):
            _compare_value(f"probe[{probe_uid}].{key}", other.get(key), row.get(key), issues)


def _compare_dimension_rows(existing: Any, rebuilt: list[dict[str, Any]], issues: list[str]) -> None:
    existing_rows = existing if isinstance(existing, list) else []
    existing_by_id = {str(item.get("dimension_id") or ""): item for item in existing_rows if isinstance(item, dict)}
    for row in rebuilt:
        dimension_id = str(row.get("dimension_id") or "")
        other = existing_by_id.get(dimension_id)
        if other is None:
            issues.append(f"Missing existing dimension row `{dimension_id}`.")
            continue
        for key in ("probe_count", "primary_avg_score", "control_avg_score", "mean_score_lift", "primary_win_rate"):
            _compare_value(f"dimension[{dimension_id}].{key}", other.get(key), row.get(key), issues)


def _compare_overall(existing: Any, rebuilt: dict[str, Any], issues: list[str]) -> None:
    existing_overall = existing if isinstance(existing, dict) else {}
    for key, value in rebuilt.items():
        if key == "skipped_probes":
            continue
        _compare_value(f"overall.{key}", existing_overall.get(key), value, issues)


def _compare_value(name: str, existing: Any, rebuilt: Any, issues: list[str]) -> None:
    existing_float = _coerce_float(existing)
    rebuilt_float = _coerce_float(rebuilt)
    if existing_float is not None or rebuilt_float is not None:
        if existing_float is None or rebuilt_float is None or round(existing_float, 4) != round(rebuilt_float, 4):
            issues.append(f"{name} mismatch: existing={existing!r} rebuilt={rebuilt!r}.")
        return
    if existing != rebuilt:
        issues.append(f"{name} mismatch: existing={existing!r} rebuilt={rebuilt!r}.")


def _load_record_list(path: Path) -> list[dict[str, Any]]:
    payload = _safe_load_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("records", "visual_dimension_records", "visual_overall_records"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _safe_load_json(path: Path) -> Any:
    try:
        text = _read_text(path)
    except FileNotFoundError:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _path_readable(path: Path) -> bool:
    if path.exists():
        return True
    if not _is_windows_absolute_path(path):
        return False
    try:
        with open(_long_windows_path(path), encoding="utf-8"):
            return True
    except FileNotFoundError:
        return False


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if not _is_windows_absolute_path(path):
            raise
        with open(_long_windows_path(path), encoding="utf-8") as handle:
            return handle.read()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_windows_absolute_path(path: Path) -> bool:
    return len(str(path)) >= 240 and path.drive.endswith(":")


def _long_windows_path(path: Path) -> str:
    text = str(path)
    if text.startswith("\\\\?\\"):
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text.lstrip("\\")
    return "\\\\?\\" + text


def _skipped_probes(payload: Any) -> list[dict[str, Any]]:
    overall = payload.get("overall") if isinstance(payload, dict) and isinstance(payload.get("overall"), dict) else {}
    skipped = overall.get("skipped_probes")
    return [item for item in skipped if isinstance(item, dict)] if isinstance(skipped, list) else []


def _dimension_label(spec: ProfileHolisticV6DimensionSpec, items: list[dict[str, Any]]) -> str:
    for item in items:
        label = item.get("dimension_label")
        if label:
            return str(label)
    return _DIMENSION_LABEL_FALLBACKS.get(spec.dimension_id, spec.label)


def _mean(values: Iterable[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(mean(numeric), 4)


def _safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.4f}"


def _md(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "/").replace("\n", " ")


__all__ = [
    "PROFILE_HOLISTIC_V6_DIMENSIONS",
    "PROFILE_HOLISTIC_V6_SCOPE",
    "build_profile_holistic_v6_dimension_rows",
    "build_profile_holistic_v6_overall_row",
    "build_profile_holistic_v6_probe_rows",
    "rebuild_profile_holistic_v6_summary",
    "validate_profile_holistic_v6_summary",
]
