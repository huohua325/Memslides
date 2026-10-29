from __future__ import annotations

from memslides.experiment import __main__ as experiment_cli


def test_print_json_retries_nonblocking_stdout(monkeypatch) -> None:
    attempts = {"count": 0}
    chunks: list[bytes] = []

    def fake_write(_fd: int, data: bytes) -> int:
        if attempts["count"] == 0:
            attempts["count"] += 1
            raise BlockingIOError(11, "temporarily unavailable")
        chunks.append(bytes(data))
        return len(data)

    monkeypatch.setattr(experiment_cli.os, "write", fake_write)
    monkeypatch.setattr(experiment_cli.select, "select", lambda *_args, **_kwargs: ([], [], []))

    experiment_cli._print_json({"success": True, "rounds_completed": 2})

    assert attempts["count"] == 1
    output = b"".join(chunks).decode("utf-8")
    assert '"success": true' in output
    assert '"rounds_completed": 2' in output


def test_print_json_ignores_broken_pipe(monkeypatch) -> None:
    def fake_write(_fd: int, _data: bytes) -> int:
        raise BrokenPipeError

    monkeypatch.setattr(experiment_cli.os, "write", fake_write)

    experiment_cli._print_json({"success": True})
