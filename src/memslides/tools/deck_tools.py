from __future__ import annotations

import sys

from memslides.tools.deck_server import main as _deck_server_main


def main(argv: list[str] | None = None) -> None:
    _deck_server_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
