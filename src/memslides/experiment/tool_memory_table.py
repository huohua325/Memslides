from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from memslides.experiment.profile_table import tex_escape


TOOL_MEMORY_MAIN_CSV = "tool_memory_main_table.csv"
TOOL_MEMORY_PAIR_CSV = "tool_memory_pair_metrics.csv"
TOOL_MEMORY_RUN_CSV = "tool_memory_run_metrics.csv"
TOOL_MEMORY_MAIN_TEX = "tool_memory_main_table.tex"
TOOL_MEMORY_SUMMARY_MD = "tool_memory_summary.md"

CHANGE_TOOLS = {
    "write_html_file",
    "apply_slide_patch",
    "batch_update_css_rule",
    "batch_update_semantic_style",
    "patch_semantic_inline_style",
    "insert_slide",
    "delete_slide",
    "write_markdown_file",
}
PROCESS_VERIFY_TOOLS = {"inspect_slide", "inspect_manuscript", "read_slide_snapshot"}
STRICT_VERIFY_TOOLS = {"inspect_slide", "inspect_manuscript"}
CORE_TIME_EXCLUDED_TOOLS = {"inspect_slide", "inspect_manuscript", "convert_to_markdown"}
TFSE_EXCLUDED_TOOLS = {"convert_to_markdown"}
VERIFY_WINDOW = 3


@dataclass(frozen=True)
class ToolMetricSpec:
    key: str
    label: str
    tex_label: str
    higher_is_better: bool
    value_kind: str


TOOL_METRICS: tuple[ToolMetricSpec, ...] = (
    ToolMetricSpec(
        "closed_loop_completion",
        "Closed-Loop Completion",
        r"\shortstack{Closed-Loop\\Completion $\uparrow$}",
        True,
        "rate",
    ),
    ToolMetricSpec(
        "strict_verify",
        "Strict Verify",
        r"\shortstack{Strict\\Verify $\uparrow$}",
        True,
        "rate",
    ),
    ToolMetricSpec(
        "first_correct_edit_s",
        "First Correct Edit (s)",
        r"\shortstack{First Correct\\Edit (s) $\downarrow$}",
        False,
        "seconds",
    ),
    ToolMetricSpec(
        "core_tool_time_ratio",
        "Core Tool Time Ratio",
        r"\shortstack{Core Tool\\Time Ratio $\downarrow$}",
        False,
        "ratio",
    ),
)


MODEL_ORDER = {
    "GPT-5": 0,
    "GLM-5": 1,
    "Gemini 3.1 Pro": 2,
}


@dataclass
class TaskBehavior:
    task_name: str
    is_modify: bool
    has_success_edit: bool
    has_verify_after_last_success_edit: bool
    has_safe_finalize: bool
    first_success_edit_s: float | None


@dataclass
class ToolRunMetrics:
    model: str
    scenario_id: str
    memory_mode: str
    run_id: str
    run_dir: Path
    modify_task_count: int
    process_task_count: int
    closed_loop_completion: float | None
    strict_verify: float | None
    first_correct_edit_s: float | None
    core_tool_time_s: float | None
    successful_change_ops: int


@dataclass
class ToolPairMetrics:
    model: str
    scenario_id: str
    tool_run: ToolRunMetrics
    no_injection_run: ToolRunMetrics
    core_tool_time_ratio: float | None


@dataclass
class MetricSummary:
    tool_value: float | None
    no_injection_value: float | None
    wins: int
    losses: int
    ties: int
    na: int
    valid_pairs: int


@dataclass
class ToolGroupSummary:
    subset: str
    pairs: int
    metric_summaries: dict[str, MetricSummary]


def build_tool_memory_report(suite_output_dir: Path | str, output_dir: Path | str) -> dict[str, Any]:
    """Build the OSS tool-memory main table from local run tool-call logs."""
    suite_output_dir = Path(suite_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metrics, warnings = collect_tool_memory_runs(suite_output_dir)
    pairs, pair_warnings = build_tool_pairs(run_metrics)
    warnings.extend(pair_warnings)
    summaries = summarize_tool_pairs(pairs)

    run_csv = output_dir / TOOL_MEMORY_RUN_CSV
    pair_csv = output_dir / TOOL_MEMORY_PAIR_CSV
    main_csv = output_dir / TOOL_MEMORY_MAIN_CSV
    tex_path = output_dir / TOOL_MEMORY_MAIN_TEX
    summary_path = output_dir / TOOL_MEMORY_SUMMARY_MD

    write_run_csv(run_csv, run_metrics)
    write_pair_csv(pair_csv, pairs)
    write_main_csv(main_csv, summaries)
    write_main_latex(tex_path, summaries)
    summary_path.write_text(
        build_tool_summary_markdown(suite_output_dir, output_dir, summaries, warnings),
        encoding="utf-8",
    )

    return {
        "suite_output_dir": str(suite_output_dir),
        "output_dir": str(output_dir),
        "run_count": len(run_metrics),
        "pair_count": len(pairs),
        "warnings": warnings,
        "outputs": {
            "main_csv": str(main_csv),
            "pair_csv": str(pair_csv),
            "run_csv": str(run_csv),
            "tex": str(tex_path),
            "summary": str(summary_path),
        },
    }


def collect_tool_memory_runs(root: Path | str) -> tuple[list[ToolRunMetrics], list[str]]:
    root = Path(root)
    warnings: list[str] = []
    index = _load_suite_experiment_index(root)
    run_dirs = discover_tool_run_dirs(root)
    if not run_dirs:
        return [], [f"No run directories with .memory/rounds found under {root}."]

    rows: list[ToolRunMetrics] = []
    for run_dir in run_dirs:
        metadata = _run_metadata(run_dir, root, index)
        mode = _normalize_memory_mode(
            str(metadata.get("memory_mode") or metadata.get("arm") or metadata.get("condition") or "")
        )
        if not mode:
            mode = _infer_memory_mode(run_dir, metadata)
        if mode not in {"tool_only", "no_injection"}:
            warnings.append(f"{run_dir}: skipped because memory_mode is not tool_only/no_injection.")
            continue
        model = _normalize_model(str(metadata.get("model") or metadata.get("model_label") or ""))
        if not model:
            model = _infer_model(run_dir, metadata)
        if not model:
            model = "unknown"
            warnings.append(f"{run_dir}: could not infer model; using unknown.")
        scenario_id = str(metadata.get("scenario_id") or metadata.get("scenario") or "").strip()
        if not scenario_id:
            scenario_id = _infer_scenario_id(run_dir, metadata, mode)
        rows.append(compute_tool_run_metrics(run_dir, model=model, scenario_id=scenario_id, memory_mode=mode))
    return sorted(rows, key=lambda item: (item.model, item.scenario_id, item.memory_mode, str(item.run_dir))), warnings


def discover_tool_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if (root / ".memory" / "rounds").is_dir():
        return [root]
    dirs: list[Path] = []
    for memory_dir in sorted(root.rglob(".memory")):
        run_dir = memory_dir.parent
        if (memory_dir / "rounds").is_dir():
            dirs.append(run_dir)
    return dirs


def compute_tool_run_metrics(run_dir: Path, *, model: str, scenario_id: str, memory_mode: str) -> ToolRunMetrics:
    task_behaviors: list[TaskBehavior] = []
    process_task_count = 0
    core_duration_ms = 0.0
    successful_change_ops = 0
    strict_verify_after_change_count = 0

    for task_dir in _iter_task_dirs(run_dir):
        raw_path = task_dir / "raw_tool_calls.json"
        if not raw_path.exists():
            continue
        raw_payload = _safe_load_json_or_list(raw_path)
        tool_calls = _tool_calls_from_payload(raw_payload)
        task_meta = _safe_load_json(task_dir / "task_meta.json")
        process_task_count += 1

        for index, tool_call in enumerate(tool_calls):
            tool_name = _tool_name(tool_call)
            duration_ms = _duration_ms(tool_call)
            if tool_name not in CORE_TIME_EXCLUDED_TOOLS:
                core_duration_ms += duration_ms
            if tool_name not in CHANGE_TOOLS or _is_error(tool_call):
                continue
            successful_change_ops += 1
            lookahead = tool_calls[index + 1 : index + 1 + VERIFY_WINDOW]
            if any(_tool_name(next_call) in STRICT_VERIFY_TOOLS and not _is_error(next_call) for next_call in lookahead):
                strict_verify_after_change_count += 1

        task_behaviors.append(build_task_behavior(task_dir, task_meta, tool_calls))

    modify_behaviors = [item for item in task_behaviors if item.is_modify]
    closed_loop_completion = _safe_divide(
        sum(item.has_safe_finalize for item in modify_behaviors),
        len(modify_behaviors),
    )
    strict_verify = _safe_divide(strict_verify_after_change_count, successful_change_ops)
    first_edits = [
        item.first_success_edit_s
        for item in modify_behaviors
        if item.first_success_edit_s is not None
    ]
    first_correct_edit_s = mean(first_edits) if first_edits else None
    core_tool_time_s = core_duration_ms / 1000.0 if process_task_count else None

    return ToolRunMetrics(
        model=model,
        scenario_id=scenario_id,
        memory_mode=memory_mode,
        run_id=run_dir.name,
        run_dir=run_dir,
        modify_task_count=len(modify_behaviors),
        process_task_count=process_task_count,
        closed_loop_completion=closed_loop_completion,
        strict_verify=strict_verify,
        first_correct_edit_s=first_correct_edit_s,
        core_tool_time_s=core_tool_time_s,
        successful_change_ops=successful_change_ops,
    )


def build_task_behavior(task_dir: Path, task_meta: dict[str, Any], tool_calls: list[dict[str, Any]]) -> TaskBehavior:
    cumulative_duration_ms = 0.0
    first_success_edit_ms: float | None = None
    successful_edit_indices: list[int] = []
    finalize_indices: list[int] = []

    for index, tool_call in enumerate(tool_calls):
        tool_name = _tool_name(tool_call)
        if first_success_edit_ms is None and tool_name not in TFSE_EXCLUDED_TOOLS:
            cumulative_duration_ms += _duration_ms(tool_call)

        if tool_name in CHANGE_TOOLS and not _is_error(tool_call):
            successful_edit_indices.append(index)
            if first_success_edit_ms is None:
                first_success_edit_ms = cumulative_duration_ms

        if tool_name == "finalize":
            finalize_indices.append(index)

    has_success_edit = bool(successful_edit_indices)
    has_verify_after_last_success_edit = False
    has_safe_finalize = False
    if has_success_edit:
        last_success_edit = successful_edit_indices[-1]
        verify_indices = [
            index
            for index in range(last_success_edit + 1, len(tool_calls))
            if _tool_name(tool_calls[index]) in STRICT_VERIFY_TOOLS and not _is_error(tool_calls[index])
        ]
        has_verify_after_last_success_edit = bool(verify_indices)
        if verify_indices:
            last_verify = verify_indices[-1]
            has_safe_finalize = any(
                finalize_index > last_verify and not _is_error(tool_calls[finalize_index])
                for finalize_index in finalize_indices
            )

    return TaskBehavior(
        task_name=task_dir.name,
        is_modify=_is_modify_task(task_dir, task_meta),
        has_success_edit=has_success_edit,
        has_verify_after_last_success_edit=has_verify_after_last_success_edit,
        has_safe_finalize=has_safe_finalize,
        first_success_edit_s=(first_success_edit_ms / 1000.0 if first_success_edit_ms is not None else None),
    )


def build_tool_pairs(run_metrics: list[ToolRunMetrics]) -> tuple[list[ToolPairMetrics], list[str]]:
    warnings: list[str] = []
    grouped: dict[tuple[str, str], dict[str, list[ToolRunMetrics]]] = {}
    for row in run_metrics:
        grouped.setdefault((row.model, row.scenario_id), {}).setdefault(row.memory_mode, []).append(row)

    pairs: list[ToolPairMetrics] = []
    for (model, scenario_id), arms in sorted(grouped.items()):
        tool_rows = arms.get("tool_only") or []
        no_rows = arms.get("no_injection") or []
        if not tool_rows or not no_rows:
            missing = "tool_only" if not tool_rows else "no_injection"
            warnings.append(f"{model} / {scenario_id}: missing paired {missing} run.")
            continue
        if len(tool_rows) > 1:
            warnings.append(f"{model} / {scenario_id}: multiple tool_only runs; selected latest by mtime.")
        if len(no_rows) > 1:
            warnings.append(f"{model} / {scenario_id}: multiple no_injection runs; selected latest by mtime.")
        tool = _latest_run(tool_rows)
        no_injection = _latest_run(no_rows)
        pairs.append(
            ToolPairMetrics(
                model=model,
                scenario_id=scenario_id,
                tool_run=tool,
                no_injection_run=no_injection,
                core_tool_time_ratio=_ratio(tool.core_tool_time_s, no_injection.core_tool_time_s),
            )
        )
    return pairs, warnings


def summarize_tool_pairs(pairs: list[ToolPairMetrics]) -> list[ToolGroupSummary]:
    model_names = sorted({pair.model for pair in pairs}, key=lambda value: (MODEL_ORDER.get(value, 99), value))
    groups: list[tuple[str, list[ToolPairMetrics]]] = [(model, [pair for pair in pairs if pair.model == model]) for model in model_names]
    if pairs:
        groups.append(("Overall", pairs))
    return [ToolGroupSummary(label, len(group_pairs), _summarize_group(group_pairs)) for label, group_pairs in groups]


def write_run_csv(path: Path, rows: list[ToolRunMetrics]) -> None:
    fieldnames = [
        "model",
        "scenario_id",
        "memory_mode",
        "run_id",
        "run_dir",
        "modify_task_count",
        "process_task_count",
        "closed_loop_completion",
        "strict_verify",
        "first_correct_edit_s",
        "core_tool_time_s",
        "successful_change_ops",
    ]
    _write_csv(
        path,
        [
            {
                "model": row.model,
                "scenario_id": row.scenario_id,
                "memory_mode": row.memory_mode,
                "run_id": row.run_id,
                "run_dir": str(row.run_dir),
                "modify_task_count": row.modify_task_count,
                "process_task_count": row.process_task_count,
                "closed_loop_completion": _format_float(row.closed_loop_completion, 4),
                "strict_verify": _format_float(row.strict_verify, 4),
                "first_correct_edit_s": _format_float(row.first_correct_edit_s, 4),
                "core_tool_time_s": _format_float(row.core_tool_time_s, 4),
                "successful_change_ops": row.successful_change_ops,
            }
            for row in rows
        ],
        fieldnames,
    )


def write_pair_csv(path: Path, pairs: list[ToolPairMetrics]) -> None:
    fieldnames = [
        "model",
        "scenario_id",
        "tool_run_id",
        "no_injection_run_id",
        "tool_closed_loop_completion",
        "no_injection_closed_loop_completion",
        "closed_loop_completion_verdict",
        "tool_strict_verify",
        "no_injection_strict_verify",
        "strict_verify_verdict",
        "tool_first_correct_edit_s",
        "no_injection_first_correct_edit_s",
        "first_correct_edit_s_verdict",
        "core_tool_time_ratio",
        "core_tool_time_ratio_verdict",
    ]
    records = []
    for pair in pairs:
        records.append(
            {
                "model": pair.model,
                "scenario_id": pair.scenario_id,
                "tool_run_id": pair.tool_run.run_id,
                "no_injection_run_id": pair.no_injection_run.run_id,
                "tool_closed_loop_completion": _format_float(pair.tool_run.closed_loop_completion, 4),
                "no_injection_closed_loop_completion": _format_float(pair.no_injection_run.closed_loop_completion, 4),
                "closed_loop_completion_verdict": _compare_metric(
                    pair.tool_run.closed_loop_completion,
                    pair.no_injection_run.closed_loop_completion,
                    TOOL_METRICS[0],
                ),
                "tool_strict_verify": _format_float(pair.tool_run.strict_verify, 4),
                "no_injection_strict_verify": _format_float(pair.no_injection_run.strict_verify, 4),
                "strict_verify_verdict": _compare_metric(pair.tool_run.strict_verify, pair.no_injection_run.strict_verify, TOOL_METRICS[1]),
                "tool_first_correct_edit_s": _format_float(pair.tool_run.first_correct_edit_s, 4),
                "no_injection_first_correct_edit_s": _format_float(pair.no_injection_run.first_correct_edit_s, 4),
                "first_correct_edit_s_verdict": _compare_metric(
                    pair.tool_run.first_correct_edit_s,
                    pair.no_injection_run.first_correct_edit_s,
                    TOOL_METRICS[2],
                ),
                "core_tool_time_ratio": _format_float(pair.core_tool_time_ratio, 4),
                "core_tool_time_ratio_verdict": _compare_metric(pair.core_tool_time_ratio, 1.0 if pair.core_tool_time_ratio is not None else None, TOOL_METRICS[3]),
            }
        )
    _write_csv(path, records, fieldnames)


def write_main_csv(path: Path, summaries: list[ToolGroupSummary]) -> None:
    fieldnames = [
        "subset",
        "pairs",
        "metric",
        "tool_only",
        "no_injection",
        "wins",
        "losses",
        "ties",
        "na",
        "w_l_t_na",
        "valid_pairs",
    ]
    records: list[dict[str, Any]] = []
    for group in summaries:
        for metric in TOOL_METRICS:
            summary = group.metric_summaries[metric.key]
            records.append(
                {
                    "subset": group.subset,
                    "pairs": group.pairs,
                    "metric": metric.label,
                    "tool_only": format_metric_value(summary.tool_value, metric),
                    "no_injection": format_metric_value(summary.no_injection_value, metric),
                    "wins": summary.wins,
                    "losses": summary.losses,
                    "ties": summary.ties,
                    "na": summary.na,
                    "w_l_t_na": f"{summary.wins}-{summary.losses}-{summary.ties}-{summary.na}",
                    "valid_pairs": summary.valid_pairs,
                }
            )
    _write_csv(path, records, fieldnames)


def write_main_latex(path: Path, summaries: list[ToolGroupSummary]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{0.9}",
        r"\caption{Tool-memory ablation on paired local modify runs. Completion and verification are higher-is-better; time and ratio are lower-is-better.}",
        r"\label{tab:tool_memory_main_oss}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        "Model & Memory Mode & " + " & ".join(metric.tex_label for metric in TOOL_METRICS) + r" \\",
        r"\midrule",
    ]
    for group_index, group in enumerate(summaries):
        if group_index:
            lines.append(r"\midrule")
        tool_cells = [tex_escape(group.subset), r"\texttt{tool\_only}"]
        base_cells = ["", r"\texttt{no\_injection}"]
        for metric in TOOL_METRICS:
            summary = group.metric_summaries[metric.key]
            verdict = _compare_metric(summary.tool_value, summary.no_injection_value, metric)
            tool_text = format_metric_value(summary.tool_value, metric, tex=True)
            base_text = format_metric_value(summary.no_injection_value, metric, tex=True)
            if verdict == "win":
                tool_text = rf"\textbf{{{tool_text}}}"
            elif verdict == "loss":
                base_text = rf"\textbf{{{base_text}}}"
            tool_cells.append(tool_text)
            base_cells.append(base_text)
        lines.append(" & ".join(tool_cells) + r" \\")
        lines.append(" & ".join(base_cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_tool_summary_markdown(
    suite_output_dir: Path,
    output_dir: Path,
    summaries: list[ToolGroupSummary],
    warnings: list[str],
) -> str:
    rows: list[dict[str, Any]] = []
    for group in summaries:
        row: dict[str, Any] = {"Subset": group.subset, "Pairs": group.pairs}
        for metric in TOOL_METRICS:
            summary = group.metric_summaries[metric.key]
            row[metric.label] = (
                f"{format_metric_value(summary.tool_value, metric)} / "
                f"{format_metric_value(summary.no_injection_value, metric)}; "
                f"{summary.wins}-{summary.losses}-{summary.ties}-{summary.na}"
            )
        rows.append(row)
    lines = [
        "# Tool-Memory Main Table",
        "",
        f"- Source: `{suite_output_dir}`",
        f"- Output: `{output_dir}`",
        "- Pairing key: `(model, scenario_id)` with `tool_only` and `no_injection` arms.",
        "- Each metric cell below is `tool_only / no_injection; W-L-T-NA` from the tool-memory perspective.",
        "",
        "## Main Summary",
        "",
        _markdown_table(
            rows,
            [
                "Subset",
                "Pairs",
                "Closed-Loop Completion",
                "Strict Verify",
                "First Correct Edit (s)",
                "Core Tool Time Ratio",
            ],
        )
        if rows
        else "_No complete tool_only/no_injection pairs were found._",
        "",
        "## Metric Definitions",
        "",
        "- `Closed-Loop Completion`: modify tasks with a successful edit followed by strict verification and a non-error `finalize`.",
        "- `Strict Verify`: successful change-tool calls followed within three tool calls by non-error `inspect_slide` or `inspect_manuscript`.",
        "- `First Correct Edit (s)`: cumulative tool time to the first successful slide-changing edit in modify tasks, excluding pre-edit `convert_to_markdown` time.",
        "- `Core Tool Time Ratio`: pair-level `tool_only / no_injection` core tool time, excluding `inspect_slide`, `inspect_manuscript`, and `convert_to_markdown`.",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(["", "## Excluded Metrics", "", "This OSS table layer intentionally excludes PPTEval quality, SlideTailor fair eval v3, profile span/visual span, and working-memory case-study metrics.", ""])
    return "\n".join(lines)


def _summarize_group(pairs: list[ToolPairMetrics]) -> dict[str, MetricSummary]:
    summaries: dict[str, MetricSummary] = {}
    for metric in TOOL_METRICS:
        values: list[tuple[float, float]] = []
        wins = losses = ties = na = 0
        for pair in pairs:
            tool_value, base_value = _pair_metric_values(pair, metric)
            verdict = _compare_metric(tool_value, base_value, metric)
            if verdict == "win":
                wins += 1
            elif verdict == "loss":
                losses += 1
            elif verdict == "tie":
                ties += 1
            else:
                na += 1
            if tool_value is not None and base_value is not None:
                values.append((tool_value, base_value))
        if not values:
            tool_agg = None
            base_agg = None
        elif metric.key == "core_tool_time_ratio":
            ratios = [tool for tool, _ in values]
            tool_agg = _geomean(ratios)
            base_agg = 1.0
        else:
            tool_agg = mean(tool for tool, _ in values)
            base_agg = mean(base for _, base in values)
        summaries[metric.key] = MetricSummary(tool_agg, base_agg, wins, losses, ties, na, len(values))
    return summaries


def _pair_metric_values(pair: ToolPairMetrics, metric: ToolMetricSpec) -> tuple[float | None, float | None]:
    if metric.key == "core_tool_time_ratio":
        return pair.core_tool_time_ratio, 1.0 if pair.core_tool_time_ratio is not None else None
    return getattr(pair.tool_run, metric.key), getattr(pair.no_injection_run, metric.key)


def _compare_metric(tool_value: float | None, base_value: float | None, metric: ToolMetricSpec) -> str:
    if tool_value is None or base_value is None:
        return "NA"
    if math.isclose(tool_value, base_value, rel_tol=1e-9, abs_tol=1e-9):
        return "tie"
    if metric.higher_is_better:
        return "win" if tool_value > base_value else "loss"
    return "win" if tool_value < base_value else "loss"


def format_metric_value(value: float | None, metric: ToolMetricSpec, *, tex: bool = False) -> str:
    if value is None:
        return "NA"
    if metric.value_kind == "ratio":
        suffix = r"$\times$" if tex else "x"
        return f"{value:.3f}{suffix}"
    if metric.value_kind == "seconds":
        return f"{value:.1f}"
    return f"{value:.3f}"


def _iter_task_dirs(run_dir: Path) -> list[Path]:
    rounds_dir = run_dir / ".memory" / "rounds"
    if not rounds_dir.is_dir():
        return []
    return sorted(
        [path for path in rounds_dir.iterdir() if path.is_dir() and path.name.startswith("task_")],
        key=lambda path: (_task_index(path), path.name),
    )


def _is_modify_task(task_dir: Path, task_meta: dict[str, Any]) -> bool:
    for key in ("task_type", "task_kind", "phase", "intent"):
        text = str(task_meta.get(key) or "").strip().lower()
        if not text:
            continue
        if any(token in text for token in ("modify", "revision", "revise", "edit")):
            return True
        if any(token in text for token in ("research", "generate", "design", "plan")):
            return False
    return _task_index(task_dir) >= 2


def _task_index(task_dir: Path) -> int:
    try:
        return int(task_dir.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _tool_calls_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw = payload.get("tool_calls") or payload.get("calls") or []
    else:
        raw = payload
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _tool_name(tool_call: dict[str, Any]) -> str:
    return str(tool_call.get("tool_name") or tool_call.get("name") or tool_call.get("tool") or "").strip()


def _duration_ms(tool_call: dict[str, Any]) -> float:
    for key in ("duration_ms", "elapsed_ms", "latency_ms"):
        if key in tool_call:
            return _float(tool_call.get(key)) or 0.0
    seconds = _float(tool_call.get("duration_s") or tool_call.get("elapsed_s"))
    if seconds is not None:
        return seconds * 1000.0
    return _float(tool_call.get("duration")) or 0.0


def _is_error(tool_call: dict[str, Any]) -> bool:
    value = tool_call.get("is_error", tool_call.get("error", False))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "error"}
    return bool(value)


def _load_suite_experiment_index(root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    suite_config = _safe_load_json(root / "suite_config.json")
    experiments = suite_config.get("experiments") if isinstance(suite_config.get("experiments"), list) else []
    for item in experiments:
        if not isinstance(item, dict):
            continue
        experiment_id = str(item.get("experiment_id") or "").strip()
        if experiment_id:
            index[f"id:{experiment_id}"] = dict(item)
        output_dir = item.get("output_dir")
        if output_dir:
            index[f"path:{(root / str(output_dir)).resolve()}"] = dict(item)

    manifest = _safe_load_json(root / "run_manifest.json")
    manifest_experiments = manifest.get("experiments") if isinstance(manifest.get("experiments"), list) else []
    for item in manifest_experiments:
        if not isinstance(item, dict):
            continue
        experiment_id = str(item.get("experiment_id") or "").strip()
        output_dir = item.get("output_dir")
        base = dict(index.get(f"id:{experiment_id}", {}))
        base.update(item)
        if experiment_id:
            index[f"id:{experiment_id}"] = base
        if output_dir:
            index[f"path:{Path(str(output_dir)).resolve()}"] = base
    return index


def _run_metadata(run_dir: Path, root: Path, index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    experiment_id = ""
    for filename in ("experiment_config.json", "metrics_summary.json", "result.json", ".input_request.json", "table_metadata.json"):
        payload = _safe_load_json(run_dir / filename)
        if not payload:
            continue
        extra_info = payload.get("extra_info") if isinstance(payload.get("extra_info"), dict) else {}
        metadata.update(payload)
        metadata.update(extra_info)
        experiment_id = experiment_id or str(payload.get("experiment_id") or "").strip()
    if not experiment_id:
        experiment_id = run_dir.name
    metadata.update(index.get(f"id:{experiment_id}", {}))
    metadata.update(index.get(f"path:{run_dir.resolve()}", {}))
    extra_info = metadata.get("extra_info") if isinstance(metadata.get("extra_info"), dict) else {}
    metadata.update(extra_info)
    metadata.setdefault("experiment_id", experiment_id)
    if not metadata.get("scenario_id"):
        for key in ("scenario_id", "memory_bucket_id", "memory_intent", "main_table_scenario"):
            if extra_info.get(key):
                metadata["scenario_id"] = extra_info[key]
                break
    return metadata


def _infer_scenario_id(run_dir: Path, metadata: dict[str, Any], mode: str) -> str:
    candidates = [
        str(metadata.get("experiment_id") or "").strip(),
        run_dir.parent.name,
        run_dir.name,
    ]
    for candidate in candidates:
        text = _clean_scenario_candidate(candidate, mode)
        if text and not _looks_like_trial_id(text):
            return text
    return _clean_scenario_candidate(run_dir.name, mode) or run_dir.name


def _clean_scenario_candidate(value: str, mode: str) -> str:
    text = value.strip()
    if not text:
        return ""
    suffixes = (
        f"_{mode}",
        "-tool-only",
        "_tool_only",
        "-no-injection",
        "_no_injection",
        "-noinj",
        "_noinj",
        "-no-inject",
        "_no_inject",
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                changed = True
                break
    return text.removeprefix("eval_")


def _looks_like_trial_id(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}_[0-9a-f]{6,16}", value.strip().lower()))


def _normalize_memory_mode(value: str) -> str:
    text = value.strip().lower().replace("-", "_")
    if text in {"tool_only", "toolmemory", "tool_memory", "tool"}:
        return "tool_only"
    if text in {"no_injection", "noinj", "no_inject", "baseline", "none"}:
        return "no_injection"
    return ""


def _infer_memory_mode(run_dir: Path, metadata: dict[str, Any]) -> str:
    candidates = [
        str(metadata.get("experiment_id") or ""),
        run_dir.name,
        run_dir.parent.name,
    ]
    for text in (_mode_text(candidate) for candidate in candidates):
        if _has_no_injection_marker(text):
            return "no_injection"
        if _has_tool_only_marker(text):
            return "tool_only"
    if metadata.get("memory_enabled") is False:
        return "no_injection"
    return ""


def _mode_text(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _has_no_injection_marker(text: str) -> bool:
    return any(marker in text for marker in ("no_injection", "noinj", "no_inject"))


def _has_tool_only_marker(text: str) -> bool:
    return "tool_only" in text or text in {"toolmemory", "tool_memory"}


def _infer_model(run_dir: Path, metadata: dict[str, Any]) -> str:
    for key in ("llm_model", "model_name", "main_table_model"):
        model = _normalize_model(str(metadata.get(key) or ""))
        if model:
            return model
    return _normalize_model(run_dir.as_posix())


def _normalize_model(value: str) -> str:
    text = value.strip()
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    if not text:
        return ""
    if "gpt5" in compact:
        return "GPT-5"
    if "glm5" in compact:
        return "GLM-5"
    if "gemini31pro" in compact or "gemini" in compact:
        return "Gemini 3.1 Pro"
    if "claudesonnet46" in compact or ("claude" in compact and "sonnet" in compact):
        return "Claude Sonnet 4.6"
    if "qwen35plus" in compact or "qwen" in compact:
        return "Qwen 3.5 Plus"
    if "kimik26" in compact or "kimi" in compact:
        return "Kimi K2.6"
    return text


def _latest_run(rows: list[ToolRunMetrics]) -> ToolRunMetrics:
    return max(rows, key=lambda row: (row.run_dir.stat().st_mtime if row.run_dir.exists() else 0.0, str(row.run_dir)))


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or math.isclose(denominator, 0.0, abs_tol=1e-12):
        return None
    return numerator / denominator


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def _geomean(values: list[float]) -> float | None:
    if not values:
        return None
    if any(value < 0 for value in values):
        return None
    if any(math.isclose(value, 0.0, abs_tol=1e-12) for value in values):
        return 0.0
    return math.exp(mean(math.log(value) for value in values))


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _safe_load_json_or_list(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: float | None, digits: int) -> str:
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


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


__all__ = [
    "TOOL_METRICS",
    "build_tool_memory_report",
    "collect_tool_memory_runs",
    "compute_tool_run_metrics",
    "build_tool_pairs",
    "summarize_tool_pairs",
]
