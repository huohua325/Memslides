from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from memslides.tools.document_conversion import mcp
from memslides.utils.log import set_logger


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("Usage: python -m memslides.tools.document_tools <workspace>")

    work_dir = Path(args[0])
    if not work_dir.exists():
        raise FileNotFoundError(f"Workspace {work_dir} does not exist.")
    os.chdir(work_dir)
    context_logger = set_logger(
        f"memslides-document-tools-{work_dir.stem}",
        work_dir / ".history" / "memslides_document_tools.log",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    for handler in context_logger.handlers:
        if handler not in root_logger.handlers:
            root_logger.addHandler(handler)
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
