from __future__ import annotations

import os
import sys
from pathlib import Path

from memslides.tools.asset_services import mcp
from memslides.utils.log import set_logger


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("Usage: python -m memslides.tools.asset_tools <workspace>")

    work_dir = Path(args[0])
    if not work_dir.exists():
        raise FileNotFoundError(f"Workspace {work_dir} does not exist.")
    os.chdir(work_dir)
    set_logger(
        f"memslides-asset-tools-{work_dir.stem}",
        work_dir / ".history" / "memslides_asset_tools.log",
    )
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
