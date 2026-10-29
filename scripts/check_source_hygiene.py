"""Check that Python maintenance text remains English-only."""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "memslides"
DOC_ROOTS = (ROOT / "README.md", ROOT / "docs", ROOT / "docker")
HAN_RANGE = ("\u4e00", "\u9fff")


def contains_han(text: str) -> bool:
    return any(HAN_RANGE[0] <= char <= HAN_RANGE[1] for char in text)


def docstring_spans(source: str) -> set[tuple[int, int, int, int]]:
    """Return exact source spans for module, class, and function docstrings."""
    tree = ast.parse(source)
    spans: set[tuple[int, int, int, int]] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            spans.add(
                (
                    first.lineno,
                    first.col_offset,
                    first.end_lineno,
                    first.end_col_offset,
                )
            )
    return spans


def check_python(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    docstrings = docstring_spans(source)
    failures: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and contains_han(token.string):
            failures.append(f"{path.relative_to(ROOT)}:{token.start[0]}: Chinese comment")
        elif (
            token.type == tokenize.STRING
            and (
                token.start[0],
                token.start[1],
                token.end[0],
                token.end[1],
            )
            in docstrings
            and contains_han(token.string)
        ):
            failures.append(f"{path.relative_to(ROOT)}:{token.start[0]}: Chinese docstring")
    return failures


def iter_doc_paths() -> list[Path]:
    paths: list[Path] = [DOC_ROOTS[0]]
    for root in DOC_ROOTS[1:]:
        if root.exists():
            paths.extend(root.rglob("*.md"))
    return paths


def main() -> int:
    failures: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        failures.extend(check_python(path))
    for path in iter_doc_paths():
        text = path.read_text(encoding="utf-8")
        if contains_han(text):
            failures.append(f"{path.relative_to(ROOT)}: Chinese maintenance text")

    if failures:
        print("Source hygiene check failed:")
        print("\n".join(failures))
        return 1
    print("Source hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
