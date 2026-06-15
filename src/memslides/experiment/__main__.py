from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import select
import sys
import time
from pathlib import Path

from memslides.experiment.config import ExperimentConfig, ExperimentSuite, UserModelProfile
from memslides.experiment.induct_template import induct_template
from memslides.experiment.profile_table import build_profile_table_report
from memslides.experiment.runner import ExperimentSuiteRunner, summarize_suite_output
from memslides.experiment.seed import BUILTIN_PERSONAS, get_user_model_profile
from memslides.experiment.tool_memory_table import build_tool_memory_report
from memslides.utils.constants import DEFAULT_CACHE_BASE


def _write_stdout_text(text: str) -> None:
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except (BlockingIOError, BrokenPipeError):
            return
        return

    data = text.encode(sys.stdout.encoding or "utf-8", errors="replace")
    offset = 0
    while offset < len(data):
        try:
            written = os.write(fd, data[offset:])
        except BlockingIOError:
            try:
                select.select([], [fd], [], 1.0)
            except OSError:
                time.sleep(0.05)
            continue
        except BrokenPipeError:
            return
        if not written:
            time.sleep(0.05)
            continue
        offset += written


def _print_json(payload: object) -> None:
    _write_stdout_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _setup_file_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(str(log_path), encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(levelname)-4s %(asctime)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG)


async def _cmd_run(args: argparse.Namespace, *, resume: bool = False) -> None:
    suite = ExperimentSuite.from_yaml(args.suite)
    if args.output_base:
        suite.output_base = Path(args.output_base).expanduser().resolve()
    if args.parallel is not None:
        suite.max_parallel = int(args.parallel)

    should_resume = resume or bool(getattr(args, "resume", False))
    suite_output_dir = suite.suite_output_dir(reuse_existing=should_resume)
    if args.log_file:
        _setup_file_logging(Path(args.log_file))
    else:
        _setup_file_logging(suite_output_dir / ".history" / f"{suite.suite_id}.log")

    runner = ExperimentSuiteRunner(suite)
    results = await runner.run_all(resume=should_resume)
    summary = {
        "suite_id": suite.suite_id,
        "output_dir": str(suite_output_dir),
        "total": len(results),
        "successful": sum(1 for item in results if item.success),
        "results": [item.to_dict() for item in results],
    }
    (suite_output_dir / "suite_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print_json(summary)


async def _cmd_report(args: argparse.Namespace) -> None:
    summary = summarize_suite_output(Path(args.output_dir))
    report_path = Path(args.output_dir) / "suite_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_json(summary)


def _cmd_profile_table(args: argparse.Namespace) -> None:
    report = build_profile_table_report(Path(args.suite_output_dir), Path(args.output_dir))
    _print_json(report)


def _cmd_tool_memory_report(args: argparse.Namespace) -> None:
    report = build_tool_memory_report(Path(args.suite_output_dir), Path(args.output_dir))
    _print_json(report)


async def _cmd_ab(args: argparse.Namespace) -> None:
    profile = UserModelProfile(
        strictness=args.strictness,
        focus=["layout", "content", "visual"],
        style="directive",
        language=args.language,
        satisfaction_threshold=args.threshold,
        persona=args.user_id,
    )
    output_base = Path(args.output_base) if args.output_base else DEFAULT_CACHE_BASE
    suite = ExperimentSuite(
        suite_id=f"ab_{args.user_id}",
        output_base=output_base,
        max_parallel=1,
        experiments=[
            ExperimentConfig(
                experiment_id=f"ab_with_{args.user_id}",
                user_id=args.user_id,
                instruction=args.prompt,
                attachments=[Path(path) for path in args.attachments],
                max_rounds=args.rounds,
                memory_enabled=True,
                memory_mode="global",
                user_model_profile=profile,
                extra_info={"arm": "with_memory"},
            ),
            ExperimentConfig(
                experiment_id=f"ab_cold_{args.user_id}",
                user_id=f"{args.user_id}_cold",
                instruction=args.prompt,
                attachments=[Path(path) for path in args.attachments],
                max_rounds=args.rounds,
                memory_enabled=True,
                memory_mode="cold",
                user_model_profile=profile,
                extra_info={"arm": "cold"},
            ),
        ],
    )
    runner = ExperimentSuiteRunner(suite)
    results = await runner.run_all()
    print(
        json.dumps(
            {"suite_id": suite.suite_id, "results": [item.to_dict() for item in results]},
            ensure_ascii=False,
            indent=2,
        )
    )


async def _cmd_template_induct(args: argparse.Namespace) -> None:
    result = await induct_template(
        Path(args.template_file),
        user_id=args.user_id,
        narrative_style=args.narrative_style,
        auto_infer_style=not args.disable_auto_infer,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        workspace=Path(args.workspace) if args.workspace else None,
        config_file=Path(args.config_file) if args.config_file else None,
        memory_db_dir=Path(args.memory_db_dir) if args.memory_db_dir else None,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memslides.experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run experiments from a suite YAML file")
    run.add_argument("suite", help="Suite YAML path or built-in suite name")
    run.add_argument("--output-base")
    run.add_argument("--parallel", type=int)
    run.add_argument("--log-file")
    run.add_argument("--resume", action="store_true")
    run.set_defaults(func=_cmd_run)

    resume = sub.add_parser("resume", help="Resume incomplete experiments in a suite")
    resume.add_argument("suite", help="Suite YAML path or built-in suite name")
    resume.add_argument("--output-base")
    resume.add_argument("--parallel", type=int)
    resume.add_argument("--log-file")
    resume.set_defaults(func=lambda args: _cmd_run(args, resume=True))

    report = sub.add_parser("report", help="Summarize finished experiments")
    report.add_argument("output_dir")
    report.set_defaults(func=_cmd_report)

    profile_table = sub.add_parser("profile-table", help="Build the profile-alignment main table")
    profile_table.add_argument("suite_output_dir")
    profile_table.add_argument("--output-dir", required=True)
    profile_table.set_defaults(func=_cmd_profile_table)

    tool_memory = sub.add_parser("tool-memory", help="Build tool-memory reproducibility reports")
    tool_sub = tool_memory.add_subparsers(dest="tool_memory_command", required=True)
    tool_report = tool_sub.add_parser("report", help="Build the tool-memory main table")
    tool_report.add_argument("suite_output_dir")
    tool_report.add_argument("--output-dir", required=True)
    tool_report.set_defaults(func=_cmd_tool_memory_report)

    ab = sub.add_parser("ab", help="Run a two-arm A/B experiment")
    ab.add_argument("--prompt", required=True)
    ab.add_argument("--attachments", nargs="+", required=True)
    ab.add_argument("--user-id", default="default")
    ab.add_argument("--rounds", type=int, default=5)
    ab.add_argument("--strictness", type=float, default=0.7)
    ab.add_argument("--threshold", type=float, default=0.85)
    ab.add_argument("--language", default="zh")
    ab.add_argument("--output-base")
    ab.set_defaults(func=_cmd_ab)

    personas = sub.add_parser("personas", help="List built-in personas")
    personas.set_defaults(func=lambda args: print(
        json.dumps({k: v["description"] for k, v in BUILTIN_PERSONAS.items()}, ensure_ascii=False, indent=2)
    ))

    template = sub.add_parser("template", help="Template utilities")
    template_sub = template.add_subparsers(dest="template_command", required=True)
    induct = template_sub.add_parser("induct", help="Induct a template into the MemSlides store")
    induct.add_argument("--template-file", required=True)
    induct.add_argument("--user-id", default="default")
    induct.add_argument("--narrative-style")
    induct.add_argument("--disable-auto-infer", action="store_true")
    induct.add_argument("--output-dir")
    induct.add_argument("--workspace")
    induct.add_argument("--config-file")
    induct.add_argument("--memory-db-dir")
    induct.set_defaults(func=_cmd_template_induct)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = args.func(args)
    if asyncio.iscoroutine(result):
        asyncio.run(result)


if __name__ == "__main__":
    main()
